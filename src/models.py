import numpy as np
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
    L = int(
        np.log(len(inputs)) / np.log(2) + 1e-6
    )  # 2^L = data dimension, data directly as basis coefficients
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
    yout = K.sigmoid(me * 5)  # Newly added
    return yout


y_pred_vmap = K.jit(
    K.vmap(y_pred, vectorized_argnums=1)
)  # Automate batch processing for a dimension of xt


def cross_entropy(yp, yt):
    return -(yt * K.log(yp + eps) + (1 - yt) * K.log(1 - yp + eps))


# Helper functions for SCRUB method
def kl_divergence(p, q):
    return K.mean(
        p * K.log((p + eps) / (q + eps))
    )  # From the perspective of P, how inaccurate Q is


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
    predicted_labels = predictions > 0.5
    true_labels = ys >= 0.5
    correct = jax.numpy.equal(predicted_labels, true_labels)
    return K.mean(correct)  # Proportion of correct samples


vgf = K.jit(
    K.vvag(loss, vectorized_argnums=(1, 2), argnums=0)
)  # vmap and value_and_grad. Vectorized batch input for xt,yt. Compute gradient for params.


class QNN:
    def __init__(self, L, d):
        self.L = L
        self.depth = d
        self.params = K.implicit_randn(stddev=0.01, shape=[self.depth, self.L, 4])

    def load(self, params):
        self.params = params

    def fit(self, xs_train, ys_train, epochs=100, batch=16, lr=0.05):
        dataset = tf.data.Dataset.from_tensor_slices(
            (xs_train, ys_train)
        )  # Output structure: dataset with each element as (x_sample, y_sample)
        dataset = dataset.repeat(
            epochs
        )  # Number of iterations of the model over the entire dataset
        dataset = dataset.shuffle(128)  # Random shuffle with buffer size 128
        dataset = dataset.batch(batch)
        self.fit_dataset(dataset, lr)

    def fit_dataset(self, dataset, lr=0.05):
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(self.params)
        for i, (x_batch, y_batch) in enumerate(dataset):
            ce, gs = vgf(self.params, x_batch.numpy(), y_batch.numpy())
            updates, opt_state = optimizer.update(
                gs, opt_state
            )  # updates is the parameter update amount
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
            epochs=epochs,  # Number of iterations of the model over the entire dataset
            batch_size=batch_size,
            verbose=verbose,
            validation_data=validation_data,
        )

    def evaluate(self, xs_val, ys_val):
        preds = (
            (self.predict(xs_val) > 0.5).astype(int).flatten()
        )  # Predict samples with probability > 0.5 as category 1, otherwise 0
        return accuracy_score(
            ys_val, preds
        )  # Proportion of correctly predicted samples

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
        best_params: Parameters and performance metrics of the best model
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
                    model.fit(
                        xs_train, ys_train, epochs=epochs, batch=batch_size, lr=lr
                    )

                    # Evaluate model on both validation and training sets (MODIFIED)
                    val_score = model.evaluate(xs_val, ys_val)
                    train_score = model.evaluate(xs_train, ys_train)  # ADDED

                    # Update best model if current is better
                    if val_score > best_score:
                        best_score = val_score
                        best_model = model
                        # Include acc and val_acc in best_params (MODIFIED)
                        best_params = {
                            "depth": depth,
                            "epochs": epochs,
                            "learning_rate": lr,
                            "batch_size": batch_size,
                            "acc": train_score,
                            "val_acc": val_score,
                        }

    return best_model, best_params


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
        best_params: Parameters and performance metrics of the best model
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

                        # Evaluate model on both validation and training sets (MODIFIED)
                        # Assuming model.evaluate returns accuracy directly
                        val_score = model.evaluate(xs_val, ys_val)
                        train_score = model.evaluate(xs_train, ys_train)  # ADDED

                        # Update best model if current is better
                        if val_score > best_score:
                            best_score = val_score
                            best_model = model
                            # Include acc and val_acc in best_params (MODIFIED)
                            best_params = {
                                "hidden_layers": hidden_layers,
                                "dropout_rate": dropout_rate,
                                "learning_rate": lr,
                                "epochs": epochs,
                                "batch_size": batch_size,
                                "acc": train_score,
                                "val_acc": val_score,
                            }

    return best_model, best_params


# ====================== Unlearning Methods ======================


# QNN Unlearning Methods
class QNNUnlearning:
    @staticmethod
    def evaluate_model(
        model,
        xs_train,
        ys_train,
        xs_retain,  # 其余的好的数据
        ys_retain,
        xs_forget,
        ys_forget,
        xs_val,
        ys_val,
    ):
        """Evaluate a model on all datasets"""
        metrics = {
            "train_acc": model.evaluate(xs_train, ys_train),
            "val_acc": model.evaluate(xs_val, ys_val),
            "retain_acc": model.evaluate(xs_retain, ys_retain),
            "forget_acc": model.evaluate(xs_forget, ys_forget),
        }
        return metrics

    @staticmethod
    def unlearn(
        model,
        xs_train,
        ys_train,
        xs_retain,
        ys_retain,
        xs_forget,
        ys_forget,
        xs_val,
        ys_val,
        mode="scrub",  # 'retrain', 'cf', 'ga'
        config=None,
    ):
        # ==================== 1. Configure based on Mode ====================
        is_retrain_mode = mode == "retrain"

        default_config = {
            "epochs_list": [30, 50],
            "batch_sizes": [16],
            "learning_rates": [0.01],
            "num_trials": 2,
            "metric": "val_acc",
            # Default weights for 'scrub' and 'ga' modes
            "kl_weight": 0.5,
            "ce_weight": 1.0,
            "fo_weight": 0.5,  # Used for neg-KL in scrub, and neg-CE in ga
        }

        if mode == "cf" or mode == "retrain" or mode == "ga":
            default_config.update(
                {"kl_weight": 0.0, "fo_weight": 0.0, "ce_weight": 1.0}
            )

        if config is None:
            config = default_config
        else:
            default_config.update(config)
            config = default_config

        best_metrics = {
            "val_acc": 0,
            "acc": 0,
            "retain_acc": 0,
            "forget_acc": 0,
            "model": None,
        }

        # ==================== 2. Define Unified JAX Loss and Grad Functions ====================

        def unified_loss_fn(
            params,
            x_retain_b,
            y_retain_b,
            x_forget_b,
            y_forget_b,
            teacher_model,
            mode,
            kl_w,
            ce_w,
            fo_w,
        ):
            total_loss = 0.0

            # --- Retain Loss (Cross-Entropy) ---
            if ce_w > 0:
                student_retain_pred = y_pred_vmap(params, x_retain_b)
                ce_retain = K.mean(cross_entropy(student_retain_pred, y_retain_b))
                total_loss += ce_w * ce_retain

            # --- SCRUB-specific losses ---
            if mode == "scrub":
                if kl_w > 0:
                    student_retain_pred = y_pred_vmap(params, x_retain_b)
                    teacher_retain_pred = teacher_model.predict(x_retain_b)
                    kl_retain = K.mean(
                        kl_divergence(teacher_retain_pred, student_retain_pred)
                    )
                    total_loss += kl_w * kl_retain

                if fo_w > 0:
                    student_forget_pred = y_pred_vmap(params, x_forget_b)
                    teacher_forget_pred = teacher_model.predict(x_forget_b)
                    kl_forget = -K.mean(
                        kl_divergence(teacher_forget_pred, student_forget_pred)
                    )
                    total_loss += fo_w * kl_forget

            # --- GRADIENT ASCENT-specific loss ---
            elif mode == "ga":
                if fo_w > 0:
                    student_forget_pred = y_pred_vmap(params, x_forget_b)
                    # Maximize loss on forget set == Minimize NEGATIVE loss
                    ascent_loss = -K.mean(
                        cross_entropy(student_forget_pred, y_forget_b)
                    )  # forget 为污染后的数据
                    total_loss += fo_w * ascent_loss

            return total_loss

        def retrain_loss_fn(params, x_retain_b, y_retain_b):
            student_retain_pred = y_pred_vmap(params, x_retain_b)
            return K.mean(cross_entropy(student_retain_pred, y_retain_b))

        # Create JIT-compiled value_and_grad functions
        # Mark all non-array arguments as static
        static_args = (5, 6, 7, 8, 9)  # teacher_model, mode, kl_w, ce_w, fo_w
        vgf_unified = K.jit(
            K.value_and_grad(unified_loss_fn, argnums=0), static_argnums=static_args
        )
        vgf_retrain = K.jit(K.value_and_grad(retrain_loss_fn, argnums=0))

        # ==================== 3. Loop through Hyperparameters ====================
        for epochs in config["epochs_list"]:
            for batch_size in config["batch_sizes"]:
                for lr in config["learning_rates"]:
                    for _ in range(config["num_trials"]):

                        # ==================== 4. Initialize Student Model based on Mode ====================
                        teacher_qnn = model
                        student_qnn = QNN(teacher_qnn.L, teacher_qnn.depth)
                        if not is_retrain_mode:
                            student_qnn.load(teacher_qnn.params)

                        optimizer = optax.adam(learning_rate=lr)
                        opt_state = optimizer.init(student_qnn.params)

                        # ==================== 5. Unified Training Loop ====================
                        retain_dataset = (
                            tf.data.Dataset.from_tensor_slices((xs_retain, ys_retain))
                            .shuffle(len(xs_retain))
                            .batch(batch_size)
                        )
                        train_dataset = retain_dataset.repeat(epochs)

                        metrilist = []
                        for step, data_batch in enumerate(train_dataset):
                            steps_per_epoch = int(np.ceil(len(xs_retain) / batch_size))
                            if step % steps_per_epoch == 0:
                                metrics = QNNUnlearning.evaluate_model(
                                    student_qnn,
                                    xs_train,
                                    ys_train,
                                    xs_retain,
                                    ys_retain,
                                    xs_forget,
                                    ys_forget,
                                    xs_val,
                                    ys_val,
                                )
                                metrilist.append(metrics)
                                print(
                                    step,
                                    f"{mode.upper()}: epochs={epochs}, batch={batch_size}, lr={lr}, "
                                    f"acc={metrics['train_acc']:.4f}, val_acc={metrics['val_acc']:.4f}, "
                                    f"retain_acc={metrics['retain_acc']:.4f}, forget_acc={metrics['forget_acc']:.4f}",
                                )

                            # --- A. Unpack data and calculate gradients based on mode ---
                            if is_retrain_mode:
                                x_retain_batch, y_retain_batch = data_batch
                                _, grads = vgf_retrain(
                                    student_qnn.params,
                                    x_retain_batch.numpy(),
                                    y_retain_batch.numpy(),
                                )
                            else:
                                x_retain_batch, y_retain_batch = data_batch
                                _, grads = vgf_unified(
                                    student_qnn.params,
                                    x_retain_batch.numpy(),
                                    y_retain_batch.numpy(),
                                    xs_forget,
                                    ys_forget,
                                    teacher_qnn,
                                    mode,
                                    config["kl_weight"],
                                    config["ce_weight"],
                                    config["fo_weight"],
                                )

                            # --- B. Apply Gradients ---
                            updates, opt_state = optimizer.update(grads, opt_state)
                            student_qnn.params = optax.apply_updates(
                                student_qnn.params, updates
                            )

                        # ==================== 6. Final Evaluation & Update Best ====================
                        metrics = QNNUnlearning.evaluate_model(
                            student_qnn,
                            xs_train,
                            ys_train,
                            xs_retain,
                            ys_retain,
                            xs_forget,
                            ys_forget,
                            xs_val,
                            ys_val,
                        )
                        print(
                            f"Mode: {mode.upper()}, epochs={epochs}, lr={lr}, val_acc={metrics['val_acc']:.4f}, forget_acc={metrics['forget_acc']:.4f}"
                        )

                        if metrics[config["metric"]] > best_metrics[config["metric"]]:
                            best_metrics.update(metrics)
                            best_metrics.update(
                                {
                                    "model": student_qnn,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "metrilist": metrilist,
                                }
                            )

        return best_metrics["model"], best_metrics


# Keras MLP Unlearning Methods
class KerasMLPUnlearning:
    @staticmethod
    def evaluate_model(
        model,
        xs_train,
        ys_train,
        xs_retain,
        ys_retain,
        xs_forget,
        ys_forget,
        xs_val,
        ys_val,
    ):
        """Evaluate a model on all datasets"""
        metrics = {
            "train_acc": model.evaluate(xs_train, ys_train),
            "val_acc": model.evaluate(xs_val, ys_val),
            "retain_acc": model.evaluate(xs_retain, ys_retain),
            "forget_acc": model.evaluate(xs_forget, ys_forget),
        }
        return metrics

    @staticmethod
    def unlearn(
        model,
        xs_train,
        ys_train,
        xs_retain,
        ys_retain,
        xs_forget,
        ys_forget,
        xs_val,
        ys_val,
        mode="scrub",  # 'retrain', 'cf', 'ga'
        config=None,
    ):
        # ==================== 1. Configure based on Mode ====================
        is_retrain_mode = mode == "retrain"

        default_config = {
            "epochs_list": [30, 50],
            "batch_sizes": [32],
            "learning_rates": [0.001],
            "num_trials": 2,
            "metric": "val_acc",
            # Default weights for 'scrub' and 'ga' modes
            "kl_weight": 0.5,  # for scrub old/new
            "ce_weight": 1.0,
            "fo_weight": 0.5,  # for scrub (negative KL) or ga (negative CE)
        }

        if mode == "cf" or mode == "retrain" or mode == "ga":
            default_config.update(
                {"kl_weight": 0.0, "fo_weight": 0.0, "ce_weight": 1.0}
            )

        if config is None:
            config = default_config
        else:
            default_config.update(config)
            config = default_config

        best_metrics = {
            "val_acc": 0,
            "acc": 0,
            "retain_acc": 0,
            "forget_acc": 0,
            "model": None,
        }

        # ==================== 2. Loop through Hyperparameters ====================
        for epochs in config["epochs_list"]:
            for batch_size in config["batch_sizes"]:
                for lr in config["learning_rates"]:
                    for _ in range(config["num_trials"]):

                        # ==================== 3. Initialize Student Model based on Mode ====================
                        teacher_model = model.get_params()["model"]
                        student_model = tf.keras.models.clone_model(teacher_model)

                        if not is_retrain_mode:
                            student_model.set_weights(teacher_model.get_weights())

                        student_model.compile(
                            optimizer=Adam(learning_rate=lr),
                            loss="binary_crossentropy",  # Placeholder
                            metrics=["accuracy"],
                        )
                        student_mlp = KerasMLP(
                            hidden_layers=model.get_params()["hidden_layers"],
                            dropout_rate=model.get_params()["dropout_rate"],
                            learning_rate=lr,
                        )
                        student_mlp.load_model(student_model)

                        # ==================== 4. Unified Training Loop ====================
                        retain_dataset = (
                            tf.data.Dataset.from_tensor_slices((xs_retain, ys_retain))
                            .shuffle(len(xs_retain))
                            .batch(batch_size)
                        )
                        train_dataset = retain_dataset.repeat(epochs)

                        metrilist = []
                        for step, data_batch in enumerate(train_dataset):
                            steps_per_epoch = int(np.ceil(len(xs_retain) / batch_size))
                            if step % steps_per_epoch == 0:
                                metrics = KerasMLPUnlearning.evaluate_model(
                                    student_mlp,
                                    xs_train,
                                    ys_train,
                                    xs_retain,
                                    ys_retain,
                                    xs_forget,
                                    ys_forget,
                                    xs_val,
                                    ys_val,
                                )
                                metrilist.append(metrics)
                                print(
                                    step,
                                    f"CF: epochs={epochs}, batch={batch_size}, lr={lr}, "
                                    f"acc={metrics['train_acc']:.4f}, val_acc={metrics['val_acc']:.4f}, "
                                    f"retain_acc={metrics['retain_acc']:.4f}, forget_acc={metrics['forget_acc']:.4f}",
                                )

                            # --- A. Unpack data and perform forward passes ---
                            with tf.GradientTape() as tape:
                                total_loss = 0.0

                                if is_retrain_mode:
                                    x_retain_batch, y_retain_batch = data_batch
                                    retain_preds = student_model(
                                        x_retain_batch, training=True
                                    )
                                    total_loss = tf.reduce_mean(
                                        tf.keras.losses.BinaryCrossentropy()(
                                            y_retain_batch, retain_preds
                                        )
                                    )
                                else:
                                    x_retain_batch, y_retain_batch = data_batch

                                    # --- B. Calculate Loss Components based on Mode and Config ---
                                    # Retain Loss (Cross-Entropy for all modes except retrain)
                                    if config["ce_weight"] > 0:
                                        retain_preds = student_model(
                                            x_retain_batch, training=True
                                        )
                                        ce_retain = tf.reduce_mean(
                                            tf.keras.losses.BinaryCrossentropy()(
                                                y_retain_batch, retain_preds
                                            )
                                        )
                                        total_loss += config["ce_weight"] * ce_retain

                                    # --- SCRUB-specific losses ---
                                    if mode == "scrub":
                                        if config["kl_weight"] > 0:
                                            retain_preds = student_model(
                                                x_retain_batch, training=True
                                            )
                                            kl_retain = tf.reduce_mean(
                                                tf.keras.losses.KLDivergence()(
                                                    teacher_model.predict(
                                                        x_retain_batch
                                                    ),
                                                    retain_preds,
                                                )
                                            )
                                            total_loss += (
                                                config["kl_weight"] * kl_retain
                                            )

                                        if config["fo_weight"] > 0:
                                            forget_preds = student_model(
                                                xs_forget, training=True
                                            )
                                            kl_forget = -tf.reduce_mean(
                                                tf.keras.losses.KLDivergence()(
                                                    teacher_model.predict(xs_forget),
                                                    forget_preds,
                                                )
                                            )
                                            total_loss += (
                                                config["fo_weight"] * kl_forget
                                            )

                                    # --- GRADIENT ASCENT-specific loss ---
                                    elif mode == "ga":
                                        if config["fo_weight"] > 0:
                                            forget_preds = student_model(
                                                xs_forget, training=True
                                            )  # forget 为污染后的数据
                                            # Maximize loss on forget set == Minimize NEGATIVE loss
                                            # We use the polluted labels `ys_forget` for ascent
                                            ascent_loss = -tf.reduce_mean(
                                                tf.keras.losses.BinaryCrossentropy()(
                                                    ys_forget, forget_preds
                                                )
                                            )
                                            total_loss += (
                                                config["fo_weight"] * ascent_loss
                                            )

                            # --- C. Apply Gradients ---
                            grads = tape.gradient(
                                total_loss, student_model.trainable_variables
                            )
                            student_model.optimizer.apply_gradients(
                                zip(grads, student_model.trainable_variables)
                            )

                        # ==================== 5. Final Evaluation & Update Best ====================
                        metrics = KerasMLPUnlearning.evaluate_model(
                            student_mlp,
                            xs_train,
                            ys_train,
                            xs_retain,
                            ys_retain,
                            xs_forget,
                            ys_forget,
                            xs_val,
                            ys_val,
                        )

                        print(
                            f"Mode: {mode.upper()}, epochs={epochs}, lr={lr}, val_acc={metrics['val_acc']:.4f}, forget_acc={metrics['forget_acc']:.4f}"
                        )

                        if metrics[config["metric"]] > best_metrics[config["metric"]]:
                            best_metrics.update(metrics)
                            best_metrics.update(
                                {
                                    "model": student_mlp,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "metrilist": metrilist,
                                }
                            )

        return best_metrics["model"], best_metrics
