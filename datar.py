import numpy as np
import tensorflow as tf
import math

def shuffle_zip(a, b):
    assert len(a) == len(b)
    combined = list(zip(a, b))
    np.random.shuffle(combined)
    a_shuffled, b_shuffled = zip(*combined)
    return np.array(a_shuffled), np.array(b_shuffled)


def complex_to_real_imag(complex_array):
    if complex_array.dtype == 'complex128':
        real_part = np.real(complex_array)
        imag_part = np.imag(complex_array)
        return np.concatenate([real_part, imag_part], axis=-1)
    else:
        return complex_array


def interclass_confusion(ys, alpha):
    # Create copy of original data
    new_ys = np.copy(ys)

    # Calculate number of samples to confuse
    n_samples = len(ys)
    n_confused = int(n_samples * alpha)

    # Randomly select indices to confuse
    confused_indices = np.random.choice(n_samples, n_confused, replace=False)

    # Randomly assign new labels for confused samples
    new_ys[confused_indices] = np.random.choice([0, 1], n_confused)

    return new_ys, confused_indices

def flip_labels(ys, alpha):
    # Create copy of original data
    new_ys = np.copy(ys)
    
    # Calculate number of samples to flip
    n_samples = len(ys)
    n_flip = int(n_samples * alpha)
    
    # Ensure n_flip is even for balanced allocation
    if n_flip % 2 == 1:
        n_flip -= 1
    
    # Split dataset into first and second halves
    first_half = np.arange(n_samples // 2)
    second_half = np.arange(n_samples // 2, n_samples)
    
    # Randomly select half of flip samples from each half
    flip_indices_first = np.random.choice(first_half, n_flip // 2, replace=False)
    flip_indices_second = np.random.choice(second_half, n_flip // 2, replace=False)

    # Combine flip indices from both halves
    flip_indices = np.concatenate([flip_indices_first, flip_indices_second])
    
    # Flip selected labels (0->1, 1->0)
    new_ys[flip_indices] = 1 - new_ys[flip_indices]
    
    return new_ys, flip_indices


def data_poison(xs, alpha, xes=1, ordered=True):
    """Poison the dataset by selecting half of the samples from each half of the dataset.
    """
    # Create copy of original data
    poisoned_xs = np.copy(xs)

    # Calculate number of samples to poison
    n_samples = len(xs)
    n_poison = int(n_samples * alpha)
    
    # Ensure n_poison is even for balanced allocation
    if n_poison % 2 != 0:
        n_poison += 1
    
    # Split dataset into first and second halves
    mid_point = n_samples // 2
    first_half_indices = np.arange(mid_point)
    second_half_indices = np.arange(mid_point, n_samples)
    
    # Select half of poison samples from each half
    n_poison_per_half = n_poison // 2
    first_half_poison = np.random.choice(first_half_indices, n_poison_per_half, replace=False)
    second_half_poison = np.random.choice(second_half_indices, n_poison_per_half, replace=False)
    
    # Combine selected indices
    poison_indices = np.concatenate([first_half_poison, second_half_poison])
    
    # Replace selected samples
    if ordered:
        for idx in poison_indices:
            poisoned_xs[idx] = xes[idx]
    else:
        # Replace with random complex values
        thetas = np.random.normal(0, 1e-1, size=[len(poison_indices), 2, len(xs[0])])
        for i, idx in enumerate(poison_indices):
            if xs.dtype == 'complex128':
                poisoned_xs[idx] = thetas[i, 0] + 1.0j * thetas[i, 1]
            else:
                poisoned_xs[idx] = thetas[i, 0]
            poisoned_xs[idx] = poisoned_xs[idx]/np.linalg.norm(poisoned_xs[idx])

    return poisoned_xs, poison_indices


def select_complement_indices(array, indices_to_exclude):
    # Create a boolean mask (True for indices not in indices_to_exclude)
    mask = np.ones(len(array), dtype=bool)
    mask[indices_to_exclude] = False
    mask1 = np.zeros(len(array), dtype=bool)
    mask1[indices_to_exclude] = True
    return array[mask1], array[mask]

def addzero(x):
    original_length = x.shape[1]

    if original_length > 0 and (original_length & (original_length - 1) == 0):
        # Original length is already a power of 2, no padding needed
        return x

    if original_length == 0:
        target_length = 1 # Edge case: if input is empty vector, pad to length 1
    else:
        n = math.ceil(math.log2(original_length))
        target_length = 2**n
    
    padding_needed = target_length - original_length
    pad_width = ((0, 0), (0, padding_needed))
    x_padded = np.pad(x, pad_width, mode='constant', constant_values=0)
    return x_padded

def normalize_to_wavefunction(image_vectors):
    norms = np.linalg.norm(image_vectors, axis=1, keepdims=True)
    # Add a small epsilon to denominator to prevent division by zero for zero vectors (all black images)
    epsilon = 1e-12
    normalized_vectors = image_vectors / (norms + epsilon)
    return normalized_vectors


def xxz_data():
    data = np.load("1D_xxz_dataset.npz")   # x: ground state, y: hzz coefficient 12qubit
    datae = np.load("1D_xxz_5_excited_dataset.npz")

    xs = []
    xes = []
    ys = []
    for x, xe, y in zip(
        data["xs"][107:189:2], datae["xs"][107:189:2], data["ys"][107:189:2]
    ):
        xs.append(x)
        xes.append(xe)
        ys.append(0)
    for x, xe, y in zip(
        data["xs"][200:-12:2], datae["xs"][200:-12:2], data["ys"][200:-12:2]
    ):
        xs.append(x)
        xes.append(xe)
        ys.append(1)
    xs0 = np.array(xs)
    ys0 = np.array(ys)
    xes0 = np.array(xes)

    xs, ys = shuffle_zip(xs0, ys0)


    xs_val = []
    ys_val = []
    xse_val = []
    for x, xe, y in zip(
        data["xs"][106:190:2], datae["xs"][106:190:2], data["ys"][106:190:2]
    ):
        xs_val.append(x)
        xse_val.append(xe)
        ys_val.append(0)
    for x, xe, y in zip(
        data["xs"][199:-11:2], datae["xs"][199:-11:2], data["ys"][199:-11:2]
    ):
        xs_val.append(x)
        xse_val.append(xe)
        ys_val.append(1)
    xs_val0 = np.array(xs_val)
    ys_val0 = np.array(ys_val)
    xes_val0 = np.array(xse_val)
    xs_val, ys_val = xs_val0, ys_val0

    print(ys0.tolist().count(0),ys0.tolist().count(1))
    print("len_train:",xs0.shape,"len_val:", xs_val.shape)

    return  xs0,ys0,xs_val, ys_val


def mnist_data(num_train):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_mask_1_or_9 = np.logical_or(y_train == 1, y_train == 9)
    x_train_1_9 = x_train[train_mask_1_or_9]
    y_train_1_9 = y_train[train_mask_1_or_9]

    test_mask_1_or_9 = np.logical_or(y_test == 1, y_test == 9)
    x_test_1_9 = x_test[test_mask_1_or_9]
    y_test_1_9 = y_test[test_mask_1_or_9]

    x_train_1_9_normalized = x_train_1_9.astype('float32') / 255.0
    x_test_1_9_normalized = x_test_1_9.astype('float32') / 255.0

    x_train_1_9_normalized = normalize_to_wavefunction(x_train_1_9_normalized)
    x_test_1_9_normalized = normalize_to_wavefunction(x_test_1_9_normalized)

    y_train_1_9_binary = np.where(y_train_1_9 == 1, 0, 1) # 0 if 1, 1 if 9 (i.e. if 9 then 1)
    y_test_1_9_binary = np.where(y_test_1_9 == 1, 0, 1)

    if x_train_1_9_normalized.ndim == 3:
        x_train_1_9_reshaped = np.expand_dims(x_train_1_9_normalized, axis=-1)
        x_test_1_9_reshaped = np.expand_dims(x_test_1_9_normalized, axis=-1)
    else:
        x_train_1_9_reshaped = x_train_1_9_normalized
        x_test_1_9_reshaped = x_test_1_9_normalized

    # Get indices of samples with label 0
    indices_train_label_0 = np.where(y_train_1_9_binary == 0)[0]
    # Get indices of samples with label 1
    indices_train_label_1 = np.where(y_train_1_9_binary == 1)[0]

    # Extract data based on indices
    x_train_label_0 = x_train_1_9_reshaped[indices_train_label_0[:int(num_train/2)]]
    y_train_label_0 = y_train_1_9_binary[indices_train_label_0[:int(num_train/2)]]

    x_train_label_1 = x_train_1_9_reshaped[indices_train_label_1[:int(num_train/2)]]
    y_train_label_1 = y_train_1_9_binary[indices_train_label_1[:int(num_train/2)]]

    # Merge sorted data
    x_train_sorted = np.concatenate((x_train_label_0, x_train_label_1), axis=0)
    y_train_sorted = np.concatenate((y_train_label_0, y_train_label_1), axis=0)

    xs0 = x_train_sorted
    ys0 = y_train_sorted
    xs0 = xs0.reshape(xs0.shape[0], -1)

    print("Training:",xs0.shape)
    print(np.count_nonzero(ys0 == 0),np.count_nonzero(ys0 == 1))

    xs_val = x_test_1_9_reshaped[:1000]
    ys_val = y_test_1_9_binary[:1000]
    xs_val = xs_val.reshape(xs_val.shape[0], -1)

    print("Validation:",xs_val.shape)

    return xs0,ys0,xs_val,ys_val