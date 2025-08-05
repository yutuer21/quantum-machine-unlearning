import numpy as np
import tensorcircuit as tc
from models import *
from datar import *
from poison_unlearn import *


K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")

xs0,ys0,xs_val,ys_val = mnist_data(500)

# mnist phase
resultsqnny = qnn_y_flip(xs0,ys0,xs_val,ys_val,ave=5,dep=6,epo=100,bat=256,lr=0.005)
resultsmlpy = mlp_y_flip(xs0,ys0,xs_val,ys_val,ave=5,hid=[64, 16],epo=100,bat=256,lr=0.005)
resultsqnnx = qnn_x_poison(xs0,ys0,xs_val,ys_val,ave=5,dep=6,epo=100,bat=256,lr=0.005)
resultsmlpx = mlp_x_poison(xs0,ys0,xs_val,ys_val,ave=5,hid=[64, 16],epo=100,bat=256,lr=0.005)

resultsmlpy2 = mlp_y_flip(xs0,ys0,xs_val,ys_val,ave=5,hid=[16, 4],epo=100,bat=256,lr=0.005)
resultsmlpx2 = mlp_x_poison(xs0,ys0,xs_val,ys_val,ave=5,hid=[16, 4],epo=100,bat=256,lr=0.005)
np.save("mnist_phase.npy",[resultsqnny,resultsmlpy,resultsqnnx,resultsmlpx,resultsmlpy2,resultsmlpx2])

#unlearn_y
alpha = 0.3
ys_confused, confused_indices = flip_labels(ys0, alpha=alpha)
xs_forget, xs_retain = select_complement_indices(xs0, confused_indices)
ys_forget, ys_retain = select_complement_indices(ys0, confused_indices)
ys_forget_pollu, ys_retain = select_complement_indices(ys_confused, confused_indices)

xs, ys_confused = shuffle_zip(xs0, ys_confused)
xs_train = complex_to_real_imag(addzero(xs))
ys_train = ys_confused
xs_retain = complex_to_real_imag(addzero(xs_retain))
xs_forget = complex_to_real_imag(addzero(xs_forget))
xs_val = complex_to_real_imag(addzero(xs_val))
print(xs_train.shape, xs_retain.shape, xs_forget.shape, xs_val.shape)

# mnist
classical_models, classical_metrics = train_keras_mlp_model(
            xs_train,
            ys_train,
            xs_val,
            ys_val,
            config={
                "hidden_layers_list": [[16, 4]],   ### 64,16
                "dropout_rates": [0.0],
                "epochs_list": [100],
                "batch_sizes": [128],
                "learning_rates": [0.01],
                "num_trials": 1,
                "early_stop_threshold": 1.0,
                "metric": "val_acc",
            },
        )


start = KerasMLPUnlearning.evaluate_model(
classical_models,
xs_train,
ys_train,      
xs_retain,
ys_retain,
xs_forget,
ys_forget,
xs_val,
ys_val,
)
print("start:",start)

# mnist
model = classical_models
epo = 50
bat = xs_retain.shape[0]
lr = 0.001

ave = 5
mode = 'retrain'
resultre = mlp_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,
                       xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.)
# ave = 1
mode = 'cf'
resultcf = mlp_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,
                       xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.)
mode = 'scrub'
resultsc = mlp_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,
                       xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.2)
mode = 'ga'
resultga = mlp_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,
                       xs_forget,ys_forget_pollu,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.2)

np.save("mnist_mlp_y.npy",[resultre,resultcf,resultsc,resultga])

mode = 'scrub'
final_re = []
for kl_w in np.arange(0., 1.0, 0.1):
    for fo_w in np.arange(0., 1.0, 0.1):
        results = mlp_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=kl_w,ce_w=1.,fo_w=fo_w)
        val_accs=[results[i]["metrilist"] for i in range(len(results))]
        acc=[]
        for i in range(ave):
            for j in range(len(val_accs[i])):
                acc.append(val_accs[i][j]["val_acc"])
        val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
        means_every_three = np.mean(val_accs_reshaped, axis=0)
        final = means_every_three[-1]
        final_re.append(final)
np.save("mnist_scrub_mlpy.npy",final_re)

mode = 'ga'
final_re = []
for fo_w in np.arange(0., 1.0, 0.1):
    results = mlp_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget_pollu,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=fo_w)
    val_accs=[results[i]["metrilist"] for i in range(len(results))]
    acc=[]
    for i in range(ave):
        for j in range(len(val_accs[i])):
            acc.append(val_accs[i][j]["val_acc"])
    val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
    means_every_three = np.mean(val_accs_reshaped, axis=0)
    final = means_every_three[-1]
    final_re.append(final)
np.save("mnist_ga_mlpy.npy",final_re)


qmodels, qmetrics = train_qnn_model(
            xs_train,
            ys_train,
            xs_val,
            ys_val,
            config={
                "depths": [6],
                "epochs_list": [100],
                "batch_sizes": [256],
                "learning_rates": [0.005],
                "num_trials": 1,
                "early_stop_threshold": 1.0,
                "metric": "val_acc",
            },
            )

start=QNNUnlearning.evaluate_model(
    qmodels,
    xs_train,
    ys_train,    
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    addzero(xs_val),
    ys_val,
)
print("start:",start)

model = qmodels
epo = 50
bat = 256
lr = 0.005

ave = 5
mode = 'retrain'
resultre = qnn_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.)
# ave = 1
mode = 'cf'
resultcf = qnn_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.)
mode = 'scrub'
resultsc = qnn_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.2)
mode = 'ga'
resultga = qnn_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget_pollu,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.2)

np.save("mnist_qnn_y.npy",[resultre,resultcf,resultsc,resultga])

mode = 'scrub'
final_re = []
for kl_w in np.arange(0., 1.0, 0.1):
    for fo_w in np.arange(0., 1.0, 0.1):
        results = qnn_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=kl_w,ce_w=1.,fo_w=fo_w)
        val_accs=[results[i]["metrilist"] for i in range(len(results))]
        acc=[]
        for i in range(ave):
            for j in range(len(val_accs[i])):
                acc.append(val_accs[i][j]["val_acc"])
        val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
        means_every_three = np.mean(val_accs_reshaped, axis=0)
        final = means_every_three[-1]
        final_re.append(final)
np.save("mnist_scrub_qnny.npy",final_re)

mode = 'ga'
final_re = []
for fo_w in np.arange(0., 1.0, 0.1):
    results = qnn_unlearn(model,mode,xs_train,ys_train,xs_retain, ys_retain,xs_forget,ys_forget_pollu,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=fo_w)
    val_accs=[results[i]["metrilist"] for i in range(len(results))]
    acc=[]
    for i in range(ave):
        for j in range(len(val_accs[i])):
            acc.append(val_accs[i][j]["val_acc"])
    val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
    means_every_three = np.mean(val_accs_reshaped, axis=0)
    final = means_every_three[-1]
    final_re.append(final)
np.save("mnist_ga_qnny.npy",final_re)