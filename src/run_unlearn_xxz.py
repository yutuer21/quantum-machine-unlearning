import numpy as np
import tensorcircuit as tc
from models import *
from datar import *
from poison_unlearn import *


K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")

xs0, ys0, xs_val, ys_val = xxz_data()

# xxz phase
resultsqnny = qnn_y_flip(
    xs0, ys0, xs_val, ys_val, ave=5, dep=4, epo=200, bat=32, lr=0.03
)
resultsqnnx = qnn_x_poison(
    xs0, ys0, xs_val, ys_val, ave=5, dep=4, epo=200, bat=32, lr=0.03
)
resultsmlpy = mlp_y_flip(
    xs0, ys0, xs_val, ys_val, ave=5, hid=[64, 16], epo=400, bat=32, lr=0.01
)
resultsmlpx = mlp_x_poison(
    xs0, ys0, xs_val, ys_val, ave=5, hid=[64, 16], epo=400, bat=32, lr=0.01
)

resultsmlpy2 = mlp_y_flip(
    xs0, ys0, xs_val, ys_val, ave=5, hid=[32, 8], epo=400, bat=32, lr=0.01
)
resultsmlpy3 = mlp_y_flip(
    xs0, ys0, xs_val, ys_val, ave=5, hid=[16, 4], epo=400, bat=32, lr=0.01
)
resultsmlpy4 = mlp_y_flip(
    xs0, ys0, xs_val, ys_val, ave=5, hid=[8, 2], epo=400, bat=32, lr=0.01
)
resultsmlpy5 = mlp_y_flip(
    xs0, ys0, xs_val, ys_val, ave=5, hid=[8, 4], epo=400, bat=32, lr=0.01
)
np.save(
    "xxz_phase.npy",
    [
        resultsqnny,
        resultsmlpy,
        resultsqnnx,
        resultsmlpx,
        resultsmlpy2,
        resultsmlpy3,
        resultsmlpy4,
        resultsmlpy5,
    ],
)

# unlearn_y
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

# xxz
classical_models, classical_metrics = train_mlp_model(
    xs_train,
    ys_train,
    xs_val,
    ys_val,
    config={
        "hidden_layers_list": [[64, 16]],  ### 64,16
        "dropout_rates": [0.0],
        "epochs_list": [400],
        "batch_sizes": [32],
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
print("start:", start)

# xxz
model = classical_models
epo = 50
bat = xs_retain.shape[0]
lr = 0.01

ave = 5
mode = "retrain"
resultre = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.0,
)
# ave = 1
mode = "cf"
resultcf = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.0,
)
mode = "scrub"
resultsc = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.2,
)
mode = "ga"
resultga = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget_pollu,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.2,
)

np.save("xxz_mlp_y.npy", [resultre, resultcf, resultsc, resultga])

mode = "scrub"
final_re = []
for kl_w in np.arange(0.0, 1.0, 0.1):
    for fo_w in np.arange(0.0, 1.0, 0.1):
        results = mlp_unlearn(
            model,
            mode,
            xs_train,
            ys_train,
            xs_retain,
            ys_retain,
            xs_forget,
            ys_forget,
            xs_val,
            ys_val,
            ave,
            epo,
            bat,
            lr,
            kl_w=kl_w,
            ce_w=1.0,
            fo_w=fo_w,
        )
        val_accs = [results[i]["metrilist"] for i in range(len(results))]
        acc = []
        for i in range(ave):
            for j in range(len(val_accs[i])):
                acc.append(val_accs[i][j]["val_acc"])
        val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
        means_every_three = np.mean(val_accs_reshaped, axis=0)
        final = means_every_three[-1]
        final_re.append(final)
np.save("xxz_scrub_mlpy.npy", final_re)

mode = "ga"
final_re = []
for fo_w in np.arange(0.0, 1.0, 0.1):
    results = mlp_unlearn(
        model,
        mode,
        xs_train,
        ys_train,
        xs_retain,
        ys_retain,
        xs_forget,
        ys_forget_pollu,
        xs_val,
        ys_val,
        ave,
        epo,
        bat,
        lr,
        kl_w=0.0,
        ce_w=1.0,
        fo_w=fo_w,
    )
    val_accs = [results[i]["metrilist"] for i in range(len(results))]
    acc = []
    for i in range(ave):
        for j in range(len(val_accs[i])):
            acc.append(val_accs[i][j]["val_acc"])
    val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
    means_every_three = np.mean(val_accs_reshaped, axis=0)
    final = means_every_three[-1]
    final_re.append(final)
np.save("xxz_ga_mlpy.npy", final_re)


qmodels, qmetrics = train_qnn_model(
    xs_train,
    ys_train,
    xs_val,
    ys_val,
    config={
        "depths": [4],
        "epochs_list": [200],
        "batch_sizes": [32],
        "learning_rates": [0.03],
        "num_trials": 1,
        "early_stop_threshold": 1.0,
        "metric": "val_acc",
    },
)

start = QNNUnlearning.evaluate_model(
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
print("start:", start)

model = qmodels
epo = 50
bat = xs_retain.shape[0]
lr = 0.005

ave = 5
mode = "retrain"
resultre = qnn_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.0,
)
# ave = 1
mode = "cf"
resultcf = qnn_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.0,
)
mode = "scrub"
resultsc = qnn_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.2,
)
mode = "ga"
resultga = qnn_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget_pollu,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.2,
)

np.save("xxz_qnn_y.npy", [resultre, resultcf, resultsc, resultga])

##


# unlearn_x
alpha = 0.8
# ys_confused, confused_indices = flip_labels(ys0, alpha=alpha)
xs_confused, confused_indices = data_poison(xs0, alpha=alpha, xes=1, ordered=False)
xs_forget, xs_retain = select_complement_indices(xs0, confused_indices)
ys_forget, ys_retain = select_complement_indices(ys0, confused_indices)
# ys_forget_pollu, ys_retain = select_complement_indices(ys_confused, confused_indices)
xs_forget_pollu, xs_retain = select_complement_indices(xs_confused, confused_indices)

# xs, ys_confused = shuffle_zip(xs0, ys_confused)
xs_confused, ys = shuffle_zip(xs_confused, ys0)
xs_train = complex_to_real_imag(addzero(xs_confused))
ys_train = ys
xs_retain = complex_to_real_imag(addzero(xs_retain))
xs_forget = complex_to_real_imag(addzero(xs_forget))
xs_val = complex_to_real_imag(addzero(xs_val))
xs_forget_pollu = complex_to_real_imag(addzero(xs_forget_pollu))
print(xs_train.shape, xs_retain.shape, xs_forget.shape, xs_val.shape)


classical_models, classical_metrics = train_mlp_model(
    xs_train,
    ys_train,
    xs_val,
    ys_val,
    config={
        "hidden_layers_list": [[64, 16]],
        "dropout_rates": [0.0],
        "epochs_list": [400],
        "batch_sizes": [32],
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
print("start:", start)

model = classical_models
epo = 50
bat = xs_retain.shape[0]
lr = 0.01

ave = 5
mode = "retrain"
resultre = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.0,
)
mode = "cf"
resultcf = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.0,
)
mode = "scrub"
resultsc = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.2,
)
mode = "ga"
resultga = mlp_unlearn(
    model,
    mode,
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget_pollu,
    ys_forget,
    xs_val,
    ys_val,
    ave,
    epo,
    bat,
    lr,
    kl_w=0.0,
    ce_w=1.0,
    fo_w=0.2,
)

np.save("xxz_mlp_x.npy", [resultre, resultcf, resultsc, resultga])

mode = "scrub"
final_re = []
for kl_w in np.arange(0.0, 1.0, 0.1):
    for fo_w in np.arange(0.0, 1.0, 0.1):
        results = mlp_unlearn(
            model,
            mode,
            xs_train,
            ys_train,
            xs_retain,
            ys_retain,
            xs_forget,
            ys_forget,
            xs_val,
            ys_val,
            ave,
            epo,
            bat,
            lr,
            kl_w=kl_w,
            ce_w=1.0,
            fo_w=fo_w,
        )
        val_accs = [results[i]["metrilist"] for i in range(len(results))]
        acc = []
        for i in range(ave):
            for j in range(len(val_accs[i])):
                acc.append(val_accs[i][j]["val_acc"])
        val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
        means_every_three = np.mean(val_accs_reshaped, axis=0)
        final = means_every_three[-1]
        final_re.append(final)
np.save("xxz_scrub_mlpx.npy", final_re)

mode = "ga"
final_re = []
for fo_w in np.arange(0.0, 1.0, 0.1):
    results = mlp_unlearn(
        model,
        mode,
        xs_train,
        ys_train,
        xs_retain,
        ys_retain,
        xs_forget_pollu,
        ys_forget,
        xs_val,
        ys_val,
        ave,
        epo,
        bat,
        lr,
        kl_w=0.0,
        ce_w=1.0,
        fo_w=fo_w,
    )
    val_accs = [results[i]["metrilist"] for i in range(len(results))]
    acc = []
    for i in range(ave):
        for j in range(len(val_accs[i])):
            acc.append(val_accs[i][j]["val_acc"])
    val_accs_reshaped = np.array(acc).reshape(-1, len(val_accs[i]))
    means_every_three = np.mean(val_accs_reshaped, axis=0)
    final = means_every_three[-1]
    final_re.append(final)
np.save("xxz_ga_mlpx.npy", final_re)
