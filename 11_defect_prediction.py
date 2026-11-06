"""
ISTQB CT-AI - Chương 11 (mục 11.5.1) Hands-on: Build a simple AI-based Defect Prediction System (H2)

Yêu cầu syllabus: dùng dataset (số đo mã nguồn + nhãn lỗi), xây model dự đoán khả năng lỗi,
dùng >= 4 features, và THỬ NHIỀU TẬP FEATURE KHÁC NHAU để thấy kết quả thay đổi thế nào.

Lưu ý: dữ liệu ở đây là GIẢ LẬP (không phải dữ liệu dự án thật). Nếu có dữ liệu thật
(ví dụ NASA MDP, PROMISE), thay hàm generate_synthetic_defect_data() bằng pd.read_csv(...).
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

ALL_FEATURES = ["loc", "complexity", "num_methods", "code_churn", "num_developers"]


def generate_synthetic_defect_data(n_samples=800, random_state=42):
    """Metric phần mềm giả lập -> nhãn buggy (1) / clean (0)."""
    rng = np.random.RandomState(random_state)
    loc = rng.randint(50, 2000, n_samples)
    complexity = rng.randint(1, 50, n_samples)
    num_methods = rng.randint(1, 100, n_samples)
    code_churn = rng.randint(0, 500, n_samples)
    num_developers = rng.randint(1, 15, n_samples)

    score = (0.001 * loc + 0.05 * complexity + 0.01 * code_churn
             + 0.1 * num_developers - 0.02 * num_methods)
    prob = 1 / (1 + np.exp(-score + 3))
    y = (rng.rand(n_samples) < prob).astype(int)
    return pd.DataFrame({"loc": loc, "complexity": complexity, "num_methods": num_methods,
                         "code_churn": code_churn, "num_developers": num_developers, "buggy": y})


def train_eval(X_train, X_test, y_train, y_test, features):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train[features], y_train)
    pred = model.predict(X_test[features])
    return model, pred, {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def main():
    print("=" * 64)
    print("DEFECT PREDICTION SYSTEM - CT-AI 11.5.1")
    print("=" * 64)

    df = generate_synthetic_defect_data()
    print("Mẫu dữ liệu (5 dòng đầu):")
    print(df.head())
    print(f"\nTỷ lệ buggy: {df['buggy'].mean() * 100:.1f}% | Tổng số mẫu: {len(df)}")

    X, y = df[ALL_FEATURES], df["buggy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # ---- Model chính: dùng đủ 5 features ----
    model, y_pred, _ = train_eval(X_train, X_test, y_train, y_test, ALL_FEATURES)
    print("\n" + "=" * 64)
    print("A. MODEL CHÍNH (5 features)")
    print("=" * 64)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Buggy"]))

    importances = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    print("Độ quan trọng của các metric:")
    print(importances.round(4).to_string())

    # ---- Thử nhiều tập feature khác nhau (yêu cầu syllabus 11.5.1) ----
    print("\n" + "=" * 64)
    print("B. SO SÁNH CÁC TẬP FEATURE KHÁC NHAU")
    print("=" * 64)
    feature_sets = {
        "Chỉ kích thước (loc, num_methods)": ["loc", "num_methods"],
        "Kích thước + độ phức tạp (loc, complexity, num_methods)": ["loc", "complexity", "num_methods"],
        "Code metrics (4: loc, complexity, num_methods, code_churn)": ["loc", "complexity", "num_methods", "code_churn"],
        "Process metrics (code_churn, num_developers)": ["code_churn", "num_developers"],
        "Toàn bộ 5 features": ALL_FEATURES,
        "5 features trừ code_churn": [f for f in ALL_FEATURES if f != "code_churn"],
    }
    rows = []
    for name, feats in feature_sets.items():
        _, _, m = train_eval(X_train, X_test, y_train, y_test, feats)
        rows.append({"Tập feature": name, "So feature": len(feats), **{k: round(v, 3) for k, v in m.items()}})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    best = result.sort_values("f1", ascending=False).iloc[0]
    worst = result.sort_values("f1").iloc[0]
    print(f"\nF1 cao nhất : {best['f1']} ({best['Tập feature']})")
    print(f"F1 thấp nhất: {worst['f1']} ({worst['Tập feature']})")
    print("Nhận xét: kết quả thay đổi rõ rệt theo feature được chọn. Chọn feature (feature selection)")
    print("ảnh hưởng trực tiếp đến chất lượng dự đoán, và tester phải kiểm tra điều đó.")

    # ---- Dự đoán cho một module mới (sử dụng thực tế: ưu tiên test) ----
    print("\n" + "=" * 64)
    print("C. DỰ ĐOÁN KHẢ NĂNG LỖI CHO MODULE MỚI (để ưu tiên kiểm thử)")
    print("=" * 64)
    new_modules = pd.DataFrame({
        "loc": [1800, 300], "complexity": [45, 5], "num_methods": [20, 60],
        "code_churn": [400, 10], "num_developers": [12, 2]},
        index=["Module_A (lớn, phức tạp)", "Module_B (nhỏ, đơn giản)"])
    p = model.predict_proba(new_modules[ALL_FEATURES])[:, 1]
    for name, prob in zip(new_modules.index, p):
        print(f"{name:<28}: xác suất có lỗi = {prob * 100:.1f}%")
    print("=> Dồn nhiều test hơn cho module có xác suất lỗi cao.")

    # ---- Vẽ ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Clean", "Buggy"], yticklabels=["Clean", "Buggy"])
    axes[0].set_title("Confusion Matrix (5 features)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    importances.sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Feature Importance")

    short = ["size", "size+cx", "code(4)", "process", "all(5)", "no churn"]
    axes[2].bar(short, result["f1"], color="seagreen")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("F1-score theo tập feature")
    axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("11_defect_prediction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nĐã lưu ảnh: 11_defect_prediction.png")


if __name__ == "__main__":
    main()
