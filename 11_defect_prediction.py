"""
ISTQB CT-AI - Chương 11
Hands-on: Build a simple AI-based Defect Prediction System (H2)
Sử dụng dữ liệu giả lập (software metrics → predict buggy/non-buggy)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def generate_synthetic_defect_data(n_samples=500, random_state=42):
    """
    Tạo dữ liệu giả lập các metric phần mềm:
    - loc (lines of code)
    - complexity (cyclomatic complexity)
    - num_methods
    - code_churn
    - num_developers
    → label: 1 = buggy, 0 = clean
    """
    rng = np.random.RandomState(random_state)

    loc = rng.randint(50, 2000, n_samples)
    complexity = rng.randint(1, 50, n_samples)
    num_methods = rng.randint(1, 100, n_samples)
    code_churn = rng.randint(0, 500, n_samples)
    num_developers = rng.randint(1, 15, n_samples)

    # Quy tắc đơn giản tạo nhãn (có nhiễu)
    # Module dễ lỗi nếu: loc cao + complexity cao + churn cao
    score = (
        0.001 * loc +
        0.05 * complexity +
        0.01 * code_churn +
        0.1 * num_developers -
        0.02 * num_methods
    )
    prob = 1 / (1 + np.exp(-score + 3))  # sigmoid
    y = (rng.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "loc": loc,
        "complexity": complexity,
        "num_methods": num_methods,
        "code_churn": code_churn,
        "num_developers": num_developers,
        "buggy": y
    })
    return df


def main():
    print("=" * 60)
    print("DEFECT PREDICTION SYSTEM - CT-AI Chapter 11")
    print("=" * 60)

    # 1. Tạo / load dữ liệu
    df = generate_synthetic_defect_data(n_samples=800)
    print("\nMẫu dữ liệu (5 dòng đầu):")
    print(df.head())
    print(f"\nTỷ lệ buggy : {df['buggy'].mean()*100:.1f}%")
    print(f"Tổng số mẫu: {len(df)}")

    # 2. Chuẩn bị dữ liệu
    feature_cols = ["loc", "complexity", "num_methods", "code_churn", "num_developers"]
    X = df[feature_cols]
    y = df["buggy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # 4. Đánh giá
    y_pred = model.predict(X_test)
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Buggy"]))

    # 5. Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nĐộ quan trọng của các metric:")
    print(importances)

    # 6. Vẽ
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Clean", "Buggy"], yticklabels=["Clean", "Buggy"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Feature importance
    importances.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Feature Importance")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("11_defect_prediction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nĐã lưu ảnh: 11_defect_prediction.png")


if __name__ == "__main__":
    main()
