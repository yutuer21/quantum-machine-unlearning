import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorcircuit as tc
import jax
import optax

K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("cotengra")

# Small constant to avoid numerical issues
eps = 1e-14

# ====================== QNN Model Implementation ======================


def build_ansatz(params, inputs):
    """Build hardware efficient ansatz using tensorcircuit"""
    L = int(np.log(len(inputs)) / np.log(2) + 1e-6)    # 2^L = data dimension, data directly as basis coefficients
    depth = int(params.shape[0])
    c = tc.Circuit(L, inputs=inputs)
    for j in range(depth):
        for i in range(L):
            c.rx(i, theta=params[j, i, 0])
            c.ry(i, theta=params[j, i, 1])
            c.rx(i, theta=params[j, i, 2])
        for i in range(L):
            c.rzz(i, (i + 1) % L, theta=params[j, i, 3])
    return c


# def y_pred(params, xt):
#     L = int(np.log(xt.shape[-1]) / np.log(2) + 1e-6)
#     c = build_ansatz(params, xt)
#     return K.real(c.expectation_ps(z=[L - 1])) / 2 + 1 / 2    # Measure IIIZ

def y_pred(params, xt):
    L = int(np.log(xt.shape[-1]) / np.log(2) + 1e-6)
    c = build_ansatz(params, xt)
    me = K.real(c.expectation_ps(z=[L - 1]))  
    yout = K.sigmoid(me*5)  # Newly added
    return  yout    

y_pred_vmap = K.jit(K.vmap(y_pred, vectorized_argnums=1))    # Automate batch processing for a dimension of xt


def cross_entropy(yp, yt):
    return -(yt * K.log(yp + eps) + (1 - yt) * K.log(1 - yp + eps))

# Helper functions for SCRUB method
def kl_divergence(p, q):
    return K.mean(p * K.log((p + eps) / (q + eps)))  # From the perspective of P, how inaccurate Q is


def loss(params, xt, yt):
    yp = y_pred(params, xt)
    return cross_entropy(yp, yt)


@K.jit
def compute_accuracy(params, xs, ys):
    predictions = y_pred_vmap(params, xs)
    # correct = jax.numpy.logical_or(
    #     jax.numpy.logical_and(ys >= 0.5, predictions > 0.5 - 1e-6), # When true label ys >= 0.5 (category 1), prediction must be > 0.5 to be correct
    #     jax.numpy.logical_and(ys < 0.5, predictions < 0.5 + 1e-6),
    # )
    predicted_labels = (predictions > 0.5)
    true_labels = (ys >= 0.5)
    correct = jax.numpy.equal(predicted_labels, true_labels)
    return K.mean(correct)    # Proportion of correct samples


vgf = K.jit(K.vvag(loss, vectorized_argnums=(1, 2), argnums=0))  # vmap and value_and_grad. Vectorized batch input for xt,yt. Compute gradient for params.


class QNN:
    def __init__(self, L, d):
        self.L = L
        self.depth = d
        self.params = K.implicit_randn(stddev=0.01, shape=[self.depth, self.L, 4])

    def load(self, params):
        self.params = params

    def fit(self, xs_train, ys_train, epochs=100, batch=16, lr=0.05):
        dataset = tf.data.Dataset.from_tensor_slices((xs_train, ys_train))  # Output structure: dataset with each element as (x_sample, y_sample)
        dataset = dataset.repeat(epochs)  # Number of iterations of the model over the entire dataset
        dataset = dataset.shuffle(128) # Random shuffle with buffer size 128
        dataset = dataset.batch(batch)
        self.fit_dataset(dataset, lr)

    def fit_dataset(self, dataset, lr=0.05):
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(self.params)
        for i, (x_batch, y_batch) in enumerate(dataset):
            ce, gs = vgf(self.params, x_batch.numpy(), y_batch.numpy())
            updates, opt_state = optimizer.update(gs, opt_state)  # updates is the parameter update amount
            self.params = optax.apply_updates(self.params, updates)

    def evaluate(self, xs_val, ys_val):
        return compute_accuracy(self.params, xs_val, ys_val)

    def predict(self, xs_val):
        return K.vmap(y_pred, vectorized_argnums=1)(self.params, xs_val)

    def get_params(self):
        return {"d": self.depth, "params": self.params}


# ====================== Keras MLP Model Implementation ======================


class KerasMLP:
    def __init__(self, hidden_layers=[64, 16], dropout_rate=0.0, learning_rate=0.01):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self, input_shape):
        model = Sequential()
        # Input layer
        model.add(tf.keras.layers.Input(shape=(input_shape,)))
        model.add(Dense(self.hidden_layers[0], activation="relu"))
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation="relu"))
            model.add(Dropout(self.dropout_rate))
        # Output layer
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return model

    def fit(
        self,
        xs_train,
        ys_train,
        epochs=100,
        batch_size=32,
        validation_data=None,
        verbose=0,  # Whether to output training status for each epoch, output=1
    ):
        if self.model is None:
            self.build_model(xs_train.shape[1])

        self.model.fit(
            xs_train,
            ys_train,
            epochs=epochs,   # Number of iterations of the model over the entire dataset
            batch_size=batch_size,
            verbose=verbose,
            validation_data=validation_data,
        )

    def evaluate(self, xs_val, ys_val):
        preds = (self.predict(xs_val) > 0.5).astype(int).flatten()  # Predict samples with probability > 0.5 as category 1, otherwise 0
        return accuracy_score(ys_val, preds)   # Proportion of correctly predicted samples

    def predict(self, xs_val):
        return self.model.predict(xs_val)

    def get_params(self):
        return {
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "model": self.model,
        }

    def load_model(self, model):
        self.model = model


# ====================== Model Training Functions ======================


def train_qnn_model(xs_train, ys_train, xs_val, ys_val, config=None):
    """
    Train a QNN model with configurable hyperparameters

    Args:
        xs_train: Training features
        ys_train: Training labels
        xs_val: Validation features
        ys_val: Validation labels
        config: Dictionary with hyperparameters
            - depths: List of depth values to try
            - epochs_list: List of epoch values to try
            - learning_rates: List of learning rate values to try
            - batch_sizes: List of batch size values to try

    Returns:
        best_model: Best trained model
        best_params: Parameters of the best model
        best_score: Best validation accuracy
    """
    if config is None:
        config = {
            "depths": [4, 5, 6],
            "epochs_list": [30, 50, 100],
            "learning_rates": [0.01, 0.05, 0.1],
            "batch_sizes": [16, 32, 64],
        }

    best_score = 0
    best_model = None
    best_params = None

    # Grid search over hyperparameters
    for depth in config["depths"]:
        for epochs in config["epochs_list"]:
            for lr in config["learning_rates"]:
                for batch_size in config["batch_sizes"]:
                    # Create and train model
                    L = int(np.log2(xs_train.shape[1] + 1e-6))
                    model = QNN(L, depth)
                    model.fit(xs_train, ys_train, epochs=epochs, batch=batch_size, lr=lr)

                    # Evaluate model
                    score = model.evaluate(xs_val, ys_val)

                    # Update best model if current is better
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = {
                            "depth": depth,
                            "epochs": epochs,
                            "learning_rate": lr,
                            "batch_size": batch_size,
                        }

    return best_model, best_params, best_score


def train_mlp_model(xs_train, ys_train, xs_val, ys_val, config=None):
    """
    Train an MLP model with configurable hyperparameters

    Args:
        xs_train: Training features
        ys_train: Training labels
        xs_val: Validation features
        ys_val: Validation labels
        config: Dictionary with hyperparameters
            - hidden_layers_list: List of hidden layer configurations
            - dropout_rates: List of dropout rates to try
            - learning_rates: List of learning rate values to try
            - epochs_list: List of epoch values to try
            - batch_sizes: List of batch size values to try

    Returns:
        best_model: Best trained model
        best_params: Parameters of the best model
        best_score: Best validation accuracy
    """
    if config is None:
        config = {
            "hidden_layers_list": [[32, 16], [64, 32], [64, 16]],
            "dropout_rates": [0.0, 0.1, 0.2],
            "learning_rates": [0.001, 0.01, 0.1],
            "epochs_list": [50, 100],
            "batch_sizes": [32, 64],
        }

    best_score = 0
    best_model = None
    best_params = None

    # Grid search over hyperparameters
    for hidden_layers in config["hidden_layers_list"]:
        for dropout_rate in config["dropout_rates"]:
            for lr in config["learning_rates"]:
                for epochs in config["epochs_list"]:
                    for batch_size in config["batch_sizes"]:
                        # Create and train model
                        model = KerasMLP(
                            hidden_layers=hidden_layers,
                            dropout_rate=dropout_rate,
                            learning_rate=lr,
                        )
                        model.fit(
                            xs_train,
                            ys_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(xs_val, ys_val),
                            verbose=0,
                        )

                        # Evaluate model
                        score = model.evaluate(xs_val, ys_val)

                        # Update best model if current is better
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_params = {
                                "hidden_layers": hidden_layers,
                                "dropout_rate": dropout_rate,
                                "learning_rate": lr,
                                "epochs": epochs,
                                "batch_size": batch_size,
                            }

    return best_model, best_params, best_score