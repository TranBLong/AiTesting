"""
ISTQB CT-AI - Chương 3
Hands-on: Demonstrate Overfitting and Underfitting (H0)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_function(X):
    """Hàm thực tế: y = cos(1.5 * pi * x)"""
    return np.cos(1.5 * np.pi * X)


def plot_overfitting_underfitting():
    np.random.seed(42)
    n_samples = 30
    degrees = [1, 4, 15]  # degree 1: underfit, 4: good, 15: overfit

    X = np.sort(np.random.rand(n_samples))
    y = true_function(X) + np.random.randn(n_samples) * 0.1

    plt.figure(figsize=(14, 5))

    for i, degree in enumerate(degrees):
        ax = plt.subplot(1, 3, i + 1)

        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([
            ("poly", polynomial_features),
            ("linear", linear_regression)
        ])
        pipeline.fit(X[:, np.newaxis], y)

        # Đánh giá bằng cross-validation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                 scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        plt.plot(X_test, true_function(X_test), label="True function")
        plt.scatter(X, y, edgecolor="b", s=20, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title(f"Degree {degree}\nMSE = {-scores.mean():.2e} (+/- {scores.std():.2e})")

    plt.suptitle("Overfitting & Underfitting Demo - CT-AI Chapter 3", fontsize=14)
    plt.tight_layout()
    plt.savefig("03_overfitting_underfitting.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Đã lưu ảnh: 03_overfitting_underfitting.png")
    print("\nGiải thích:")
    print("- Degree 1  : Underfitting (model quá đơn giản)")
    print("- Degree 4  : Fit tốt")
    print("- Degree 15 : Overfitting (model quá phức tạp, học cả nhiễu)")


if __name__ == "__main__":
    plot_overfitting_underfitting()
