"""Quantum-inspired Classifiers for Machine Learning Workflows.

Features Quantum Kernel Ridge Classification and Variational Quantum Classifiers (VQC)
using analytical parameter-shift gradients.
"""

from __future__ import annotations
from typing import Optional, Literal
import numpy as np

from quantrix.ml.kernels import QuantumKernel, QuantumFeatureMap
from quantrix.ml.circuits import ParameterizedQuantumCircuit


class QuantumKernelClassifier:
    """Quantum Kernel Ridge Classifier.

    Maps inputs into high-dimensional quantum superposition space and fits
    a dual linear model with Tikhonov regularization.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding: Literal["angle", "amplitude", "zz"] = "angle",
        alpha: float = 1e-3,
        backend: Optional[str] = None
    ) -> None:
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.alpha = alpha
        self.kernel = QuantumKernel(n_qubits=n_qubits, encoding=encoding, backend=backend)
        self.X_train: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantumKernelClassifier:
        """Trains classifier by computing train Gram matrix and solving dual system."""
        self.X_train = np.asarray(X)
        self.classes = np.unique(y)
        
        # Binary target representation (-1, +1)
        if len(self.classes) == 2:
            y_target = np.where(y == self.classes[0], -1.0, 1.0)
        else:
            # One-hot encoding for multi-class
            y_target = np.eye(len(self.classes))[y]

        # Compute full training Gram matrix using vectorized einsum
        K = self.kernel(self.X_train, self.X_train)
        reg = self.alpha * np.eye(len(self.X_train))
        
        # Solve dual coefficients (K + alpha * I) w = y
        self.weights = np.linalg.solve(K + reg, y_target)
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Computes raw model decision outputs: K_test @ weights."""
        if self.X_train is None or self.weights is None:
            raise RuntimeError("Classifier is not fitted.")
        K_test = self.kernel(X, self.X_train)
        return K_test @ self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for test samples."""
        raw = self.predict_raw(X)
        if len(self.classes) == 2:  # type: ignore
            binary_pred = np.where(raw >= 0.0, self.classes[1], self.classes[0])  # type: ignore
            return binary_pred
        # Multiclass argmax
        pred_indices = np.argmax(raw, axis=1)
        return self.classes[pred_indices]  # type: ignore

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculates classification accuracy on test dataset."""
        preds = self.predict(X)
        return float(np.mean(preds == y))


class VariationalQuantumClassifier:
    """Variational Quantum Classifier using Parameterized Quantum Circuits.

    Optimizes rotation angles via gradient descent using exact Parameter Shift Rule gradients.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        learning_rate: float = 0.05,
        max_epochs: int = 20,
        backend: Optional[str] = None
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.backend = backend
        self.feature_map = QuantumFeatureMap(n_qubits=n_qubits, encoding="angle", backend=backend)
        self.circuit = ParameterizedQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers, backend=backend)

    def fit(self, X: np.ndarray, y: np.ndarray) -> VariationalQuantumClassifier:
        """Optimizes circuit parameters using Parameter-Shift gradient updates."""
        X_arr = np.asarray(X)
        y_target = np.where(y == 0, -1.0, 1.0)
        n_samples = len(X_arr)

        for _ in range(self.max_epochs):
            total_grad = np.zeros_like(self.circuit.params)
            for i in range(n_samples):
                # Encode input sample into initial state
                init_state = self.feature_map.encode(X_arr[i:i+1])
                # Compute gradient for this sample
                pred = self.circuit.expectation(initial_state=init_state)
                err = pred - y_target[i]
                sample_grad = self.circuit.compute_gradients_parameter_shift(initial_state=init_state)
                total_grad += err * sample_grad

            # Gradient descent step
            self.circuit.params -= self.learning_rate * (total_grad / n_samples)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels based on sign of circuit expectation value."""
        X_arr = np.asarray(X)
        preds = []
        for i in range(len(X_arr)):
            init_state = self.feature_map.encode(X_arr[i:i+1])
            exp_val = self.circuit.expectation(initial_state=init_state)
            preds.append(1 if exp_val >= 0.0 else 0)
        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))
