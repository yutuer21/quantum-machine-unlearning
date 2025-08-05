import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import tensorcircuit as tc
from models import *
from datar import *

# poison
def qnn_y_flip(xs0,ys0,xs_val,ys_val,ave=5,dep=6,epo=100,bat=256,lr=0.005):
    results = []
    for alpha in np.arange(0., 1.0, 0.1):
        for _ in range(ave):
            print(alpha)
            ys_confused, confused_indices = flip_labels(ys0, alpha=alpha)
            xs, ys_confused = shuffle_zip(xs0, ys_confused)
            qmodels, qmetrics = train_qnn_model(
            addzero(xs),
            ys_confused,
            addzero(xs_val),
            ys_val,
            config={
                "depths": [dep],
                "epochs_list": [epo],
                "batch_sizes": [bat],
                "learning_rates": [lr],
                "num_trials": 1,
                "early_stop_threshold": 1.0,
                "metric": "val_acc",
            },
        )
            results.append(qmetrics)
    return results

def mlp_y_flip(xs0,ys0,xs_val,ys_val,ave=5,hid=[64, 16],epo=100,bat=256,lr=0.005):
    results = []
    for alpha in np.arange(0., 1., 0.1):
        for _ in range(ave):
            print(alpha)
            ys_confused, confused_indices = flip_labels(ys0, alpha=alpha)
            xs, ys_confused = shuffle_zip(xs0, ys_confused)
            classical_models, classical_metrics = train_keras_mlp_model(
            complex_to_real_imag(addzero(xs)),
            ys_confused,
            complex_to_real_imag(addzero(xs_val)),
            ys_val,
            config={
                "hidden_layers_list": [hid],
                "dropout_rates": [0.0],
                "epochs_list": [epo],
                "batch_sizes": [bat],
                "learning_rates": [lr],
                "num_trials": 1,
                "early_stop_threshold": 1.0,
                "metric": "val_acc",
            },
        )
            results.append(classical_metrics)
    return results


def mlp_x_poison(xs0,ys0,xs_val,ys_val,ave=5,hid=[64, 16],epo=100,bat=256,lr=0.005):
    results = []
    for alpha in np.arange(0., 1.0, 0.1):
        print(alpha)
        for _ in range(ave):
            xs_confused, confused_indices = data_poison(xs0, alpha=alpha, xes=1, ordered=False)
            xs_confused, ys = shuffle_zip(xs_confused, ys0)  
            classical_models, classical_metrics = train_keras_mlp_model(
            complex_to_real_imag(addzero(xs_confused)),
            ys,
            complex_to_real_imag(addzero(xs_val)),
            ys_val,
            config={
                "hidden_layers_list": [hid],
                "dropout_rates": [0.0],
                "epochs_list": [epo],
                "batch_sizes": [bat],
                "learning_rates": [lr],
                "num_trials": 1,
                "early_stop_threshold": 1.0,
                "metric": "val_acc",
            },
        )
            results.append(classical_metrics)
    return results
    

def qnn_x_poison(xs0,ys0,xs_val,ys_val,ave=5,dep=6,epo=100,bat=256,lr=0.005):
    results = []
    for alpha in np.arange(0., 1., 0.1):
        print(alpha)
        for _ in range(ave):
            xs_confused, confused_indices = data_poison(xs0, alpha=alpha, xes=1, ordered=False)
            xs_confused, ys = shuffle_zip(xs_confused, ys0)
            qmodels, qmetrics = train_qnn_model(
            addzero(xs_confused),
            ys,
            addzero(xs_val),
            ys_val,
            config={
                "depths": [dep],
                "epochs_list": [epo],
                "batch_sizes": [bat],
                "learning_rates": [lr],
                "num_trials": 1,
                "early_stop_threshold": 1.0,
                "metric": "val_acc",
            },
        )
            results.append(qmetrics)
    return results
   

# unlearn

# mnist
def mlp_unlearn(model,mode,xs,ys,xs_retain, ys_retain,xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.):
    results = []
    for _ in range(ave):
        model1, metrics = KerasMLPUnlearning.unlearn(
            model,
            xs,
            ys,
            xs_retain,
            ys_retain,
            xs_forget,
            ys_forget,
            xs_val,
            ys_val,
            mode = mode,
            config={
                "epochs_list": [epo],
                "batch_sizes": [bat],
                "learning_rates": [lr],
                "num_trials": 1,
                "metric": "val_acc",
                "kl_weight": kl_w,
                "ce_weight": ce_w,  # CE is usually the primary objective
                "fo_weight": fo_w,

            },
        )
        results.append(metrics)
    return results

def qnn_unlearn(model,mode,xs,ys,xs_retain, ys_retain,xs_forget,ys_forget,xs_val,ys_val, ave,epo,bat,lr,kl_w=0.,ce_w=1.,fo_w=0.):
    results = []
    for _ in range(ave):
        model1, metrics = QNNUnlearning.unlearn(
            model,
            xs,
            ys,
            xs_retain,
            ys_retain,
            xs_forget,
            ys_forget,
            xs_val,
            ys_val,
            mode = mode,
            config={
                "epochs_list": [epo],
                "batch_sizes": [bat], #xs_retain.shape[0]
                "learning_rates": [lr],
                "num_trials": 1,
                "metric": "val_acc",
                "kl_weight": kl_w,
                "ce_weight": ce_w,
                "fo_weight": fo_w,

            },
        )
        results.append(metrics)
    return results