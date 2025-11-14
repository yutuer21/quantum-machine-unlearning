import numpy as np
import tensorcircuit as tc
from models import *
from datar import *
from poison_unlearn import *

# --- 1. 基本设置 (无变化) ---
K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")

# --- 2. 大幅减少数据量 ---
print("Loading a small dataset for testing...")
xs0, ys0, xs_val, ys_val = mnist_data(40)

# --- 3. 简化 "Phase" 部分的测试 ---
# 参数统一调整为: ave=1 (不重复), dep=1 (最浅QNN), epo=2 (最短训练), hid=[16, 4] (最小MLP), bat=16 (小批量)
print("Running simplified 'Phase' tests...")
TEST_AVE = 1
TEST_EPO = 2
TEST_BAT = 16
TEST_LR = 0.005
TEST_DEP = 1
TEST_HID = [16, 4]

resultsqnny = qnn_y_flip(
    xs0,
    ys0,
    xs_val,
    ys_val,
    ave=TEST_AVE,
    dep=TEST_DEP,
    epo=TEST_EPO,
    bat=TEST_BAT,
    lr=TEST_LR,
)
resultsmlpy = mlp_y_flip(
    xs0,
    ys0,
    xs_val,
    ys_val,
    ave=TEST_AVE,
    hid=TEST_HID,
    epo=TEST_EPO,
    bat=TEST_BAT,
    lr=TEST_LR,
)
resultsqnnx = qnn_x_poison(
    xs0,
    ys0,
    xs_val,
    ys_val,
    ave=TEST_AVE,
    dep=TEST_DEP,
    epo=TEST_EPO,
    bat=TEST_BAT,
    lr=TEST_LR,
)
resultsmlpx = mlp_x_poison(
    xs0,
    ys0,
    xs_val,
    ys_val,
    ave=TEST_AVE,
    hid=TEST_HID,
    epo=TEST_EPO,
    bat=TEST_BAT,
    lr=TEST_LR,
)
# 移除了多余的 mlp*2 测试，因为函数是一样的
np.save(
    "test_mnist_phase.npy",
    [resultsqnny, resultsmlpy, resultsqnnx, resultsmlpx],
)
print("'Phase' tests completed and saved.")

# --- 4. 数据遗忘准备 (逻辑保留，但数据量小，运行快) ---
print("Preparing data for unlearning tests...")
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
print("Data shapes:", xs_train.shape, xs_retain.shape, xs_forget.shape, xs_val.shape)


# --- 5. 简化 MLP 训练和遗忘测试 ---
print("Testing initial MLP training...")
classical_models, classical_metrics = train_mlp_model(
    xs_train,
    ys_train,
    xs_val,
    ys_val,
    config={
        "hidden_layers_list": [TEST_HID],  # 使用最小模型
        "dropout_rates": [0.0],
        "epochs_list": [TEST_EPO],  # 最短训练
        "batch_sizes": [TEST_BAT],  # 小批量
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
print("Initial MLP evaluation:", start)

print("Testing MLP unlearning methods (retrain, cf, scrub, ga)...")
# 使用统一的快速测试参数
model = classical_models
resultre = mlp_unlearn(
    model,
    "retrain",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
)
resultcf = mlp_unlearn(
    model,
    "cf",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
)
resultsc = mlp_unlearn(
    model,
    "scrub",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
    fo_w=0.2,
)
resultga = mlp_unlearn(
    model,
    "ga",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget_pollu,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
    fo_w=0.2,
)
np.save("test_mnist_mlp_y.npy", [resultre, resultcf, resultsc, resultga])
print("MLP unlearning tests completed.")

# --- 6. 移除昂贵的超参数搜索循环 ---
# 原脚本中有两个大的嵌套 for 循环用于搜索 kl_w 和 fo_w，这是最耗时的部分。
# 对于功能测试，我们完全移除它们。
print("Skipping expensive hyperparameter search loops for MLP.")


# --- 7. 简化 QNN 训练和遗忘测试 ---
print("Testing initial QNN training...")
qmodels, qmetrics = train_qnn_model(
    xs_train,
    ys_train,
    xs_val,
    ys_val,
    config={
        "depths": [TEST_DEP],  # 最浅模型
        "epochs_list": [TEST_EPO],  # 最短训练
        "batch_sizes": [TEST_BAT],  # 小批量
        "learning_rates": [TEST_LR],
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
print("Initial QNN evaluation:", start)

print("Testing QNN unlearning methods (retrain, cf, scrub, ga)...")
model = qmodels
resultre = qnn_unlearn(
    model,
    "retrain",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
)
resultcf = qnn_unlearn(
    model,
    "cf",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
)
resultsc = qnn_unlearn(
    model,
    "scrub",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
    fo_w=0.2,
)
resultga = qnn_unlearn(
    model,
    "ga",
    xs_train,
    ys_train,
    xs_retain,
    ys_retain,
    xs_forget,
    ys_forget_pollu,
    xs_val,
    ys_val,
    TEST_AVE,
    TEST_EPO,
    TEST_BAT,
    TEST_LR,
    fo_w=0.2,
)
np.save("test_mnist_qnn_y.npy", [resultre, resultcf, resultsc, resultga])
print("QNN unlearning tests completed.")

# --- 8. 同样，移除QNN的超参数搜索循环 ---
print("Skipping expensive hyperparameter search loops for QNN.")

print("\nAll library functions tested successfully with minimal parameters.")
