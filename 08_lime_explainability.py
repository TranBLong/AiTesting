"""
ISTQB CT-AI - Chương 8 (mục 8.6.1) Hands-on: Model Explainability với LIME (H2)

LƯU Ý quan trọng: LIME mặc định chỉ giải thích MỘT lớp (lớp số 1). Phải chỉ định đúng
lớp mà model DỰ ĐOÁN (pred_label) thì giải thích mới có ý nghĩa.
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Độ chính xác trên tập test: {model.score(X_test, y_test):.3f}")

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=42,
    )

    print("\n" + "=" * 64)
    print("GIẢI THÍCH DỰ ĐOÁN BẰNG LIME")
    print("=" * 64)

    # Giải thích 1 mẫu đúng + (nếu có) 1 mẫu model dự đoán sai để thấy giá trị với tester
    preds = model.predict(X_test)
    wrong = [i for i in range(len(y_test)) if preds[i] != y_test[i]]
    sample_indices = [0] + wrong[:1]

    for k, idx in enumerate(sample_indices):
        sample = X_test[idx]
        true_label = y_test[idx]
        pred_label = preds[idx]
        proba = model.predict_proba([sample])[0]
        tag = "DỰ ĐOÁN SAI" if pred_label != true_label else "dự đoán đúng"

        print(f"\n--- Mẫu test số {idx} ({tag}) ---")
        print(f"Đặc trưng  : { {n: round(float(v), 2) for n, v in zip(feature_names, sample)} }")
        print(f"Nhãn thực  : {class_names[true_label]}")
        print(f"Nhãn dự đoán: {class_names[pred_label]} (xác suất {proba[pred_label]:.2f})")

        # top_labels=1 -> LIME giải thích lớp có xác suất cao nhất; as_list(label=...) lấy đúng lớp đó
        explanation = explainer.explain_instance(
            data_row=sample, predict_fn=model.predict_proba,
            num_features=4, top_labels=1)

        print(f"Đặc trưng ảnh hưởng đến việc chọn '{class_names[pred_label]}' (theo LIME):")
        for feature, weight in explanation.as_list(label=pred_label):
            direction = "ủng hộ" if weight > 0 else "phản đối"
            print(f"  {feature:36s} → {weight:+.4f} ({direction})")

        explanation.as_pyplot_figure(label=pred_label)
        plt.title(f"LIME - Mẫu {idx} | True: {class_names[true_label]} | "
                  f"Pred: {class_names[pred_label]}")
        plt.tight_layout()
        fname = f"08_lime_explanation_{k + 1}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Đã lưu ảnh: {fname}")

    print("\nGiải thích:")
    print("- Trọng số DƯƠNG: đặc trưng đẩy dự đoán VỀ PHÍA lớp được giải thích.")
    print("- Trọng số ÂM   : đặc trưng đẩy dự đoán RA XA lớp đó.")
    print("- Với tester: kiểm tra model có dựa vào đặc trưng hợp lý không "
          "(ví dụ petal length/width cho hoa Iris) thay vì 'lý do lạ'.")


if __name__ == "__main__":
    main()
