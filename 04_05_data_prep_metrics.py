"""
ISTQB CT-AI - Chương 4 + 5
Hands-on:
- Data Preparation & Train/Validation/Test split (H2)
- Evaluate ML model with Accuracy, Precision, Recall, F1, Confusion Matrix (H2)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # 1. Load dữ liệu
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print("=" * 60)
    print("1. THÔNG TIN DATASET (Iris)")
    print("=" * 60)
    print(f"Số mẫu        : {X.shape[0]}")
    print(f"Số đặc trưng  : {X.shape[1]}")
    print(f"Tên lớp       : {list(target_names)}")
    print(f"Phân bố lớp   : {dict(zip(target_names, pd.Series(y).value_counts().sort_index()))}")

    # 2. Chia dữ liệu: 60% train - 20% validation - 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("\n" + "=" * 60)
    print("2. CHIA DỮ LIỆU (Train / Validation / Test)")
    print("=" * 60)
    print(f"Train      : {X_train.shape[0]} mẫu ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation : {X_val.shape[0]} mẫu ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test       : {X_test.shape[0]} mẫu ({X_test.shape[0]/len(X)*100:.1f}%)")

    # 3. Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Đánh giá trên Validation và Test
    print("\n" + "=" * 60)
    print("3. ĐÁNH GIÁ MÔ HÌNH (ML Functional Performance Metrics)")
    print("=" * 60)

    for name, (Xs, ys) in {"Validation": (X_val, y_val), "Test": (X_test, y_test)}.items():
        pred = model.predict(Xs)
        print(f"\n--- {name} set ---")
        print(f"Accuracy  : {accuracy_score(ys, pred):.4f}")
        print(f"Precision : {precision_score(ys, pred, average='macro'):.4f}")
        print(f"Recall    : {recall_score(ys, pred, average='macro'):.4f}")
        print(f"F1-score  : {f1_score(ys, pred, average='macro'):.4f}")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(ys, pred)
        print(cm)
        print("\nClassification Report:")
        print(classification_report(ys, pred, target_names=target_names))

    # 5. Vẽ Confusion Matrix cho Test set
    pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, pred_test)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set (CT-AI Ch.5)")
    plt.tight_layout()
    plt.savefig("04_05_confusion_matrix.png", dpi=150)
    plt.show()
    print("\nĐã lưu ảnh: 04_05_confusion_matrix.png")


if __name__ == "__main__":
    main()
