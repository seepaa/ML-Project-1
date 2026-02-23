"""
Feedforward Neural Network with Backpropagation for Binary Classification
==========================================================================
CS 4033/5033 Machine Learning Fundamentals - Project 1, Spring 2026

This module implements a feedforward neural network (FFNN) trained via
error backpropagation from scratch using only NumPy. It supports:
  - Arbitrary number of hidden layers and neurons per layer
  - Sigmoid activation for hidden layers, sigmoid output for binary classification
  - Mean Squared Error (MSE) loss
  - Configurable learning rate, epochs, and train/validation/test splits
  - Multiple repetitions with random weight re-initialization
  - Decision boundary visualization (2D datasets)
  - Per-dataset performance logging (accuracy, loss curves)

Authors: Kiara Nelson, Laura Nguyen]
"""

import numpy as np
import csv
import os


# =============================================================================
# 1. Activation Functions
# =============================================================================

def sigmoid(z):
    """Sigmoid activation, clipped for numerical stability."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid given the activation output a = sigmoid(z)."""
    return a * (1.0 - a)


# =============================================================================
# 2. Neural Network Class
# =============================================================================

class FeedforwardNeuralNetwork:
    """
    A fully-connected feedforward neural network for binary classification.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer, including input and output.
        Example: [2, 8, 1] means 2 inputs, 8 hidden neurons, 1 output.
    learning_rate : float
        Step size for gradient descent (default 0.1).
    momentum : float
        Momentum coefficient for weight updates (default 0.0).
    random_seed : int or None
        Seed for reproducible weight initialization.
    """

    def __init__(self, layer_sizes, learning_rate=0.1, momentum=0.0, random_seed=None):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_layers = len(layer_sizes)
        self._init_weights(random_seed)

    def _init_weights(self, seed=None):
        """Initialize weights using Xavier/Glorot initialization and biases to zero."""
        if seed is not None:
            np.random.seed(seed)
        self.weights = []   # weights[l] has shape (layer_sizes[l], layer_sizes[l+1])
        self.biases = []    # biases[l] has shape (1, layer_sizes[l+1])
        self.vel_w = []     # momentum velocity for weights
        self.vel_b = []     # momentum velocity for biases
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)
            self.vel_w.append(np.zeros_like(w))
            self.vel_b.append(np.zeros_like(b))

    # ----- Forward Pass -----
    def forward(self, X):
        """
        Compute forward pass through the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        activations : list of ndarrays
            Activation at each layer (including input).
        """
        activations = [X]
        a = X
        for i in range(self.num_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = sigmoid(z)
            activations.append(a)
        return activations

    # ----- Backward Pass -----
    def backward(self, activations, y):
        """
        Compute gradients via backpropagation and update weights.

        Parameters
        ----------
        activations : list of ndarrays from forward pass.
        y : ndarray of shape (n_samples, 1), target labels.
        """
        m = y.shape[0]
        # Output layer error (MSE derivative * sigmoid derivative)
        delta = (activations[-1] - y) * sigmoid_derivative(activations[-1])

        for i in reversed(range(self.num_layers - 1)):
            grad_w = (activations[i].T @ delta) / m
            grad_b = np.mean(delta, axis=0, keepdims=True)

            # Momentum update
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.learning_rate * grad_w
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.learning_rate * grad_b
            self.weights[i] += self.vel_w[i]
            self.biases[i] += self.vel_b[i]

            if i > 0:
                delta = (delta @ self.weights[i].T) * sigmoid_derivative(activations[i])

    # ----- Prediction & Loss -----
    def predict(self, X):
        """Return predicted class labels (0 or 1)."""
        activations = self.forward(X)
        return (activations[-1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Return raw output probabilities."""
        return self.forward(X)[-1]

    @staticmethod
    def mse_loss(y_pred, y_true):
        """Mean Squared Error."""
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def accuracy(y_pred_labels, y_true):
        """Classification accuracy."""
        return np.mean(y_pred_labels == y_true)

    # ----- Training Loop -----
    def train(self, X_train, y_train, X_val, y_val, epochs=500, verbose=False):
        """
        Train the network for a fixed number of epochs.

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            Lists of per-epoch metrics.
        """
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            # Forward
            activations = self.forward(X_train)
            # Backward (updates weights in-place)
            self.backward(activations, y_train)

            # Record metrics
            train_pred = (activations[-1] >= 0.5).astype(int)
            t_loss = self.mse_loss(activations[-1], y_train)
            t_acc = self.accuracy(train_pred, y_train)

            val_acts = self.forward(X_val)
            val_pred = (val_acts[-1] >= 0.5).astype(int)
            v_loss = self.mse_loss(val_acts[-1], y_val)
            v_acc = self.accuracy(val_pred, y_val)

            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:4d} | Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f} | "
                      f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}")

        return history


# =============================================================================
# 3. Data Loading Utilities
# =============================================================================

def load_dataset(filepath):
    """
    Load a CSV dataset and return X (features) and y (labels).

    The CSV has no header. For 2D data (4 columns): cols 0-1 are Class 0,
    cols 2-3 are Class 1. For 3D data (6 columns): cols 0-2 are Class 0,
    cols 3-5 are Class 1.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples, 1)
    """
    data = np.genfromtxt(filepath, delimiter=',')
    n_cols = data.shape[1]
    n_features = n_cols // 2

    class0 = data[:, :n_features]          # shape (N, d)
    class1 = data[:, n_features:]          # shape (N, d)

    X = np.vstack([class0, class1])
    y = np.vstack([np.zeros((class0.shape[0], 1)),
                   np.ones((class1.shape[0], 1))])
    return X, y


def train_val_test_split(X, y, train_frac=0.6, val_frac=0.2, seed=None):
    """
    Randomly shuffle and split data into train / validation / test sets.

    Parameters
    ----------
    train_frac : float  – fraction for training (default 60%)
    val_frac   : float  – fraction for validation (default 20%)
    (Remaining fraction goes to test, default 20%.)

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def normalize(X_train, X_val, X_test):
    """Z-score normalization using training set statistics."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma, mu, sigma


# =============================================================================
# 4. Experiment Runner
# =============================================================================

def run_experiment(filepath, dataset_name, hidden_layers, epochs=1000,
                   learning_rate=0.1, momentum=0.9, n_reps=10, output_dir='results'):
    """
    Run the full experiment pipeline on one dataset.

    Parameters
    ----------
    filepath : str       – path to the CSV file.
    dataset_name : str   – human-readable name for logging/plots.
    hidden_layers : list – list of hidden layer sizes, e.g. [8] or [16, 8].
    epochs : int         – number of training epochs per repetition.
    learning_rate : float
    momentum : float
    n_reps : int         – number of repetitions (different random seeds).
    output_dir : str     – folder to save results.

    Returns
    -------
    summary : dict with aggregated results across repetitions.
    """
    os.makedirs(output_dir, exist_ok=True)
    X, y = load_dataset(filepath)
    n_features = X.shape[1]

    # Build layer_sizes list
    layer_sizes = [n_features] + hidden_layers + [1]

    all_train_acc = []
    all_val_acc = []
    all_test_acc = []
    all_histories = []

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Architecture: {layer_sizes}  |  LR={learning_rate}  Mom={momentum}  Epochs={epochs}")
    print(f"{'='*70}")

    for rep in range(n_reps):
        # Different random split each repetition
        split_seed = rep * 42
        X_train, y_train, X_val, y_val, X_test, y_test = \
            train_val_test_split(X, y, seed=split_seed)

        # Normalize
        X_train_n, X_val_n, X_test_n, mu, sigma = normalize(X_train, X_val, X_test)

        # Build and train network
        net = FeedforwardNeuralNetwork(layer_sizes, learning_rate=learning_rate,
                                       momentum=momentum, random_seed=rep)
        history = net.train(X_train_n, y_train, X_val_n, y_val,
                            epochs=epochs, verbose=(rep == 0))

        # Test evaluation
        test_pred = net.predict(X_test_n)
        test_acc = FeedforwardNeuralNetwork.accuracy(test_pred, y_test)

        all_train_acc.append(history['train_acc'][-1])
        all_val_acc.append(history['val_acc'][-1])
        all_test_acc.append(test_acc)
        all_histories.append(history)

        print(f"  Rep {rep+1:2d} | Train Acc: {history['train_acc'][-1]:.4f} | "
              f"Val Acc: {history['val_acc'][-1]:.4f} | Test Acc: {test_acc:.4f}")

    # Summary statistics
    summary = {
        'dataset': dataset_name,
        'architecture': layer_sizes,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'n_reps': n_reps,
        'train_acc_mean': np.mean(all_train_acc),
        'train_acc_std': np.std(all_train_acc),
        'val_acc_mean': np.mean(all_val_acc),
        'val_acc_std': np.std(all_val_acc),
        'test_acc_mean': np.mean(all_test_acc),
        'test_acc_std': np.std(all_test_acc),
        'all_train_acc': all_train_acc,
        'all_val_acc': all_val_acc,
        'all_test_acc': all_test_acc,
        'all_histories': all_histories,
    }

    print(f"\n  SUMMARY: Test Acc = {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")

    # ---- Save raw data for this dataset ----
    raw_path = os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_raw.csv")
    with open(raw_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Repetition', 'Train_Acc', 'Val_Acc', 'Test_Acc'])
        for i in range(n_reps):
            writer.writerow([i + 1, all_train_acc[i], all_val_acc[i], all_test_acc[i]])
        writer.writerow([])
        writer.writerow(['Mean', np.mean(all_train_acc), np.mean(all_val_acc), np.mean(all_test_acc)])
        writer.writerow(['Std', np.std(all_train_acc), np.std(all_val_acc), np.std(all_test_acc)])

    return summary, net, mu, sigma


# =============================================================================
# 5. Plotting Utilities
# =============================================================================

def plot_learning_curves(histories, dataset_name, output_dir='results'):
    """Plot average training/validation loss and accuracy curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_epochs = len(histories[0]['train_loss'])
    avg_train_loss = np.mean([h['train_loss'] for h in histories], axis=0)
    avg_val_loss = np.mean([h['val_loss'] for h in histories], axis=0)
    avg_train_acc = np.mean([h['train_acc'] for h in histories], axis=0)
    avg_val_acc = np.mean([h['val_acc'] for h in histories], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(avg_train_loss, label='Train Loss')
    axes[0].plot(avg_val_loss, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title(f'{dataset_name} – Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(avg_train_acc, label='Train Accuracy')
    axes[1].plot(avg_val_acc, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{dataset_name} – Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_decision_boundary(net, X, y, mu, sigma, dataset_name, output_dir='results'):
    """
    Plot the decision boundary for 2D datasets. Shows data points colored
    by class and the learned decision region as a filled contour.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if X.shape[1] != 2:
        print(f"  Skipping decision boundary plot for {dataset_name} (not 2D).")
        return

    # Create mesh
    margin = 0.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_norm = (grid - mu) / (sigma + 1e-8)

    Z = net.predict_proba(grid_norm).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 51),
                          cmap='RdBu_r', alpha=0.7)
    plt.colorbar(contour, ax=ax, label='P(Class 1)')

    # Draw decision boundary line
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # Scatter data points
    class0 = y.ravel() == 0
    class1 = y.ravel() == 1
    ax.scatter(X[class0, 0], X[class0, 1], c='blue', edgecolors='k',
               s=30, label='Class 0', alpha=0.8)
    ax.scatter(X[class1, 0], X[class1, 1], c='red', edgecolors='k',
               s=30, label='Class 1', alpha=0.8)

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(f'{dataset_name} – Decision Boundary')
    ax.legend()

    plt.tight_layout()
    fname = os.path.join(output_dir, f"{dataset_name.replace(' ', '_')}_boundary.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_summary_table(summaries, output_dir='results'):
    """Create and save a summary bar chart of test accuracies across all datasets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [s['dataset'] for s in summaries]
    means = [s['test_acc_mean'] for s in summaries]
    stds = [s['test_acc_std'] for s in summaries]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Across All Datasets (Mean ± Std over 10 Repetitions)')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f'{m:.1%}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'summary_accuracy.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved summary chart: {fname}")


# =============================================================================
# 6. Main – Run All Experiments
# =============================================================================

def main():
    """Run experiments on all 9 datasets and generate all plots and data files."""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(data_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # ----- Configuration -----
    # Design choices:
    #   - Gaussian 2D: 1 hidden layer with 8 neurons (simple problem, 2 inputs)
    #   - Gaussian 3D: 1 hidden layer with 16 neurons (3 inputs, slightly more capacity)
    #   - Moons 2D:    2 hidden layers [16, 8] (nonlinear boundary requires more capacity)
    #
    # Common settings:
    #   - Learning rate: 0.1 (standard starting point for sigmoid networks)
    #   - Momentum: 0.9 (speeds convergence, helps escape shallow local minima)
    #   - Epochs: 1000 (enough for convergence on these small datasets)
    #   - Repetitions: 10 (balance between statistical reliability and compute time)
    #   - Train/Val/Test split: 60/20/20 (standard split)
    #   - Activation: Sigmoid (common for binary classification)
    #   - Weight init: Xavier/Glorot (prevents vanishing/exploding gradients)
    #   - Normalization: Z-score on training set (ensures equal feature contribution)

    datasets = [
        ('Gaussian 2D Wide',    'Gaussian 2D Wide.csv',    [8],      1000, 0.1, 0.9),
        ('Gaussian 2D Narrow',  'Gaussian 2D Narrow.csv',  [8],      1000, 0.1, 0.9),
        ('Gaussian 2D Overlap', 'Gaussian 2D Overlap.csv', [8],      1000, 0.1, 0.9),
        ('Gaussian 3D Wide',    'Gaussian 3D Wide.csv',    [16],     1000, 0.1, 0.9),
        ('Gaussian 3D Narrow',  'Gaussian 3D Narrow.csv',  [16],     1000, 0.1, 0.9),
        ('Gaussian 3D Overlap', 'Gaussian 3D Overlap.csv', [16],     1000, 0.1, 0.9),
        ('Moons 2D Wide',       'Moons 2D Wide.csv',       [16, 8],  1000, 0.1, 0.9),
        ('Moons 2D Narrow',     'Moons 2D Narrow.csv',     [16, 8],  1000, 0.1, 0.9),
        ('Moons 2D Overlap',    'Moons 2D Overlap.csv',    [16, 8],  1000, 0.1, 0.9),
    ]

    all_summaries = []

    for name, fname, hidden, epochs, lr, mom in datasets:
        filepath = os.path.join(data_dir, fname)
        summary, net, mu, sigma = run_experiment(
            filepath, name, hidden_layers=hidden,
            epochs=epochs, learning_rate=lr, momentum=mom,
            n_reps=10, output_dir=output_dir
        )
        all_summaries.append(summary)

        # Plots
        plot_learning_curves(summary['all_histories'], name, output_dir)

        # Decision boundary (reload full data for plotting)
        X, y = load_dataset(filepath)
        plot_decision_boundary(net, X, y, mu, sigma, name, output_dir)

    # Summary chart
    plot_summary_table(all_summaries, output_dir)

    # Save grand summary CSV
    summary_path = os.path.join(output_dir, 'grand_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Architecture', 'Epochs', 'LR', 'Momentum',
                         'Train_Acc_Mean', 'Train_Acc_Std',
                         'Val_Acc_Mean', 'Val_Acc_Std',
                         'Test_Acc_Mean', 'Test_Acc_Std'])
        for s in all_summaries:
            writer.writerow([
                s['dataset'], str(s['architecture']), s['epochs'],
                s['learning_rate'], s['momentum'],
                f"{s['train_acc_mean']:.4f}", f"{s['train_acc_std']:.4f}",
                f"{s['val_acc_mean']:.4f}", f"{s['val_acc_std']:.4f}",
                f"{s['test_acc_mean']:.4f}", f"{s['test_acc_std']:.4f}",
            ])
    print(f"\nSaved grand summary: {summary_path}")
    print("\nAll experiments complete!")


if __name__ == '__main__':
    main()
