"""
ISTQB CT-AI - Chương 8
Hands-on: Model Explainability với LIME (H2)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


def main():
    # 1. Load & chia dữ liệu
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Độ chính xác trên tập test: {model.score(X_test, y_test):.3f}")

    # 3. Tạo LIME Explainer
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    # 4. Giải thích một mẫu cụ thể (ví dụ mẫu đầu tiên của test set)
    sample_idx = 0
    sample = X_test[sample_idx]
    true_label = y_test[sample_idx]
    pred_label = model.predict([sample])[0]

    print("\n" + "=" * 60)
    print("GIẢI THÍCH MỘT DỰ ĐOÁN BẰNG LIME")
    print("=" * 60)
    print(f"Mẫu số        : {sample_idx}")
    print(f"Giá trị đặc trưng: {dict(zip(feature_names, sample))}")
    print(f"Nhãn thực tế  : {class_names[true_label]}")
    print(f"Nhãn dự đoán  : {class_names[pred_label]}")

    # Giải thích
    explanation = explainer.explain_instance(
        data_row=sample,
        predict_fn=model.predict_proba,
        num_features=4
    )

    print("\nCác đặc trưng quan trọng nhất (theo LIME):")
    for feature, weight in explanation.as_list():
        print(f"  {feature:40s} → trọng số: {weight:+.4f}")

    # Lưu ảnh giải thích
    fig = explanation.as_pyplot_figure()
    plt.title(f"LIME Explanation - Sample {sample_idx}\n"
              f"True: {class_names[true_label]} | Pred: {class_names[pred_label]}")
    plt.tight_layout()
    plt.savefig("08_lime_explanation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nĐã lưu ảnh: 08_lime_explanation.png")
    print("\nGiải thích: Thanh màu càng dài = đặc trưng càng ảnh hưởng mạnh đến quyết định của mô hình.")


if __name__ == "__main__":
    main()
