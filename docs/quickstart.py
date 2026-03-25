"""
DomainML Quickstart Tutorial
----------------------------
This script demonstrates how to build an MVP model using strict monotonic
constraints with CVXPY via `MonotonicLinearRegression`.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from domainml.core.metadata import FeatureMetadata
from domainml.constraints.engine import MonotonicityEngine

def main():
    print("DomainML Quickstart: Monotonicity Constraint Demonstration\n")
    
    # 1. Generate Synthetic Data
    np.random.seed(42)
    X_train = np.random.rand(50, 1) * 10
    y_train = X_train[:, 0] * 2 + np.random.randn(50) * 5

    # Add a conflicting trend intentionally (e.g., data points trending down at the end)
    X_conflict = np.array([[8], [9], [10]])
    y_conflict = np.array([5, -2, -10])
    X_train = np.vstack((X_train, X_conflict))
    y_train = np.concatenate((y_train, y_conflict))

    print("[*] Generated synthetic data with an intentional conflicting downward trend.")

    # 2. Defining Constraints
    # We know that the target should always strictly increase with respect to this feature.
    print("[*] Defining FeatureMetadata with strict 'inc' (increasing) constraint.")
    meta = FeatureMetadata(
        feature_names=["f1"],
        monotonicities=["inc"],
        constraint_types=["strict"]
    )

    # 3. Model Training
    # Standard Linear Regression
    print("[*] Fitting Standard Linear Regression (will overfit the conflict).")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # DomainML Engine wrapper for Linear Regression
    print("[*] Fitting DomainML MonotonicityEngine (will enforce domain knowledge).")
    engine = MonotonicityEngine(LinearRegression())
    engine.fit(X_train, y_train, metadata=meta)

    # 4. Evaluation & Visualization
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred_standard = lr.predict(X_test)
    y_pred_constrained = engine.predict(X_test)

    print("[*] Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, label="Training data", alpha=0.5)
    plt.plot(X_test, y_pred_standard, color="red", linestyle="--", label="Standard LR (Overfitting Conflict)")
    plt.plot(X_test, y_pred_constrained, color="green", linewidth=2, label="DomainML (Strict Increasing)")
    
    plt.title("DomainML: Strict Monotonicity vs Standard Regression")
    plt.xlabel("Feature (f1)")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
