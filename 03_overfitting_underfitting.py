"""
ISTQB CT-AI - Chương 3 (mục 3.5.3) Hands-on: Demonstrate Overfitting and Underfitting (H0)

Ý tưởng:
- Underfitting : model quá đơn giản (đa thức bậc 1) -> sai cả trên dữ liệu train lẫn dữ liệu mới.
- Good fit     : model vừa đủ (bậc 4).
- Overfitting  : model quá phức tạp (bậc 15) trên ÍT dữ liệu -> train rất tốt, dữ liệu mới rất tệ.
=> So sánh sai số TRAIN và sai số CROSS-VALIDATION (dữ liệu "chưa từng thấy").
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def true_function(X):
    """Hàm thực tế: y = cos(1.5 * pi * x)"""
    return np.cos(1.5 * np.pi * X)


def main():
    np.random.seed(42)
    n_samples = 30  # ít dữ liệu -> dễ overfit (đúng như mô tả syllabus 3.5.1)
    degrees = [1, 4, 15]
    labels = {1: "Underfitting", 4: "Good fit", 15: "Overfitting"}

    X = np.sort(np.random.rand(n_samples))
    y = true_function(X) + np.random.randn(n_samples) * 0.1

    plt.figure(figsize=(14, 5))
    results = []
    for i, degree in enumerate(degrees):
        ax = plt.subplot(1, 3, i + 1)
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ])
        pipeline.fit(X[:, np.newaxis], y)

        train_mse = mean_squared_error(y, pipeline.predict(X[:, np.newaxis]))
        cv_scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                    scoring="neg_mean_squared_error", cv=10)
        cv_mse = -cv_scores.mean()
        results.append((degree, labels[degree], train_mse, cv_mse))

        X_test = np.linspace(0, 1, 100)
        ax.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
        ax.plot(X_test, true_function(X_test), label="True function")
        ax.scatter(X, y, edgecolor="b", s=20, label="Samples")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim((0, 1))
        ax.set_ylim((-2, 2))
        ax.legend(loc="best")
        ax.set_title(f"Degree {degree} - {labels[degree]}\n"
                     f"Train MSE={train_mse:.3f} | CV MSE={cv_mse:.3g}")

    plt.suptitle("Overfitting & Underfitting Demo - CT-AI 3.5.3", fontsize=14)
    plt.tight_layout()
    plt.savefig("03_overfitting_underfitting.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("=" * 66)
    print(f"{'Bậc':<5}{'Trạng thái':<14}{'Train MSE':>12}{'CV MSE (dữ liệu mới)':>26}")
    print("=" * 66)
    for degree, label, tr, cv in results:
        print(f"{degree:<5}{label:<14}{tr:>12.4f}{cv:>26.4g}")
    print("=" * 66)
    print("Giải thích:")
    print("- Bậc 1 : Underfitting -> sai số cao CẢ trên train lẫn CV (model quá đơn giản).")
    print("- Bậc 4 : Fit tốt      -> sai số thấp và gần nhau giữa train và CV.")
    print("- Bậc 15: Overfitting  -> train MSE rất thấp nhưng CV MSE rất cao (học cả nhiễu).")
    print("=> Với tester: đừng chỉ đo trên dữ liệu train; khoảng cách train-vs-new là dấu hiệu overfitting.")
    print("Đã lưu ảnh: 03_overfitting_underfitting.png")


if __name__ == "__main__":
    main()
