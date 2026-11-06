"""
ISTQB CT-AI - Chương 4 + 5 Hands-on:
  4.1.2  Data Preparation for ML (H2)
  4.2.1  Identify Training and Test Data and Create an ML Model (H2)
  5.4.1  Evaluate the Created ML Model (H2)

Luồng:
  Dữ liệu thô (Iris + lỗi được cố ý chèn vào) -> làm sạch -> chuẩn hoá
  -> chia Train/Validation/Test -> tinh chỉnh bằng Validation -> chấm bằng Test (1 lần)
  -> Accuracy / Precision / Recall / F1 / Confusion Matrix.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def title(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def make_raw_data():
    """Tạo dữ liệu 'thô' từ Iris, cố ý chèn lỗi để minh hoạ các bước Data Preparation."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    rng = np.random.RandomState(0)

    # (a) Thiếu dữ liệu: 8 ô bị trống
    for _ in range(8):
        df.iloc[rng.randint(0, len(df)), rng.randint(0, 4)] = np.nan
    # (b) Dữ liệu trùng lặp: lặp lại 5 dòng
    df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)
    # (c) Ngoại lệ (outlier / nhập sai đơn vị): 3 giá trị lớn gấp 10 lần
    for idx in rng.choice(len(df), 3, replace=False):
        df.loc[idx, "sepal_length"] = df.loc[idx, "sepal_length"] * 10
    return df, list(iris.target_names)


def prepare_data(df):
    """4.1 Data preparation: pre-processing (cleaning, transformation)."""
    title("1. CHUẨN BỊ DỮ LIỆU (Data Preparation - mục 4.1)")
    print(f"Dữ liệu thô: {df.shape[0]} dòng")
    print(f"- Giá trị thiếu : {int(df.isna().sum().sum())}")
    print(f"- Dòng trùng lặp: {int(df.duplicated().sum())}")

    # Cleaning 1: bỏ trùng lặp
    df = df.drop_duplicates().copy()

    # Cleaning 2: xử lý ngoại lệ (sepal_length > 15 cm là phi thực tế với hoa Iris)
    n_out = int((df["sepal_length"] > 15).sum())
    df.loc[df["sepal_length"] > 15, "sepal_length"] = np.nan
    print(f"- Ngoại lệ (sepal_length > 15cm) coi là thiếu dữ liệu: {n_out}")

    # Cleaning 3: Imputation - điền giá trị thiếu bằng MEDIAN của từng lớp
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in feature_cols:
        df[col] = df.groupby("species")[col].transform(lambda s: s.fillna(s.median()))
    print(f"Sau cleaning: {df.shape[0]} dòng, thiếu = {int(df.isna().sum().sum())}")
    return df, feature_cols


def main():
    raw_df, target_names = make_raw_data()
    df, feature_cols = prepare_data(raw_df)
    X = df[feature_cols].values
    y = df["species"].values

    # ---- 4.2 Chia dữ liệu: 60% train - 20% validation - 20% test ----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Transformation: chuẩn hoá. CHỈ fit trên train để tránh rò rỉ dữ liệu (data leakage)
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))

    title("2. CHIA DỮ LIỆU (Train / Validation / Test - mục 4.2)")
    for name, arr in [("Train", X_train), ("Validation", X_val), ("Test", X_test)]:
        print(f"{name:<11}: {arr.shape[0]:>3} mẫu ({arr.shape[0] / len(X) * 100:.1f}%)")

    # ---- Tinh chỉnh (tuning) bằng VALIDATION set ----
    title("3. TINH CHỈNH MÔ HÌNH BẰNG VALIDATION SET")
    best = None
    for depth in [1, 2, 3, 5, None]:
        m = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_val, m.predict(X_val))
        print(f"max_depth={str(depth):<5} -> Validation accuracy = {acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, depth, m)
    val_acc, best_depth, model = best
    print(f"=> Chọn max_depth = {best_depth} (validation accuracy = {val_acc:.4f})")

    # ---- 5.x Đánh giá cuối cùng trên TEST (dữ liệu chưa từng dùng để chọn model) ----
    title("4. ĐÁNH GIÁ (ML Functional Performance Metrics - mục 5)")
    for name, (Xs, ys) in {"Validation": (X_val, y_val), "Test": (X_test, y_test)}.items():
        pred = model.predict(Xs)
        print(f"\n--- {name} set ---")
        print(f"Accuracy  : {accuracy_score(ys, pred):.4f}")
        print(f"Precision : {precision_score(ys, pred, average='macro', zero_division=0):.4f} (macro)")
        print(f"Recall    : {recall_score(ys, pred, average='macro'):.4f} (macro)")
        print(f"F1-score  : {f1_score(ys, pred, average='macro'):.4f} (macro)")
        print("Confusion Matrix:")
        print(confusion_matrix(ys, pred))
        if name == "Test":
            print("\nClassification Report:")
            print(classification_report(ys, pred, zero_division=0))

    test_acc = accuracy_score(y_test, model.predict(X_test))
    title("5. EVALUATING/TUNING vs TESTING")
    print(f"Validation accuracy = {val_acc:.4f} | Test accuracy = {test_acc:.4f}")
    print("- Validation set được dùng để CHỌN/tinh chỉnh model -> con số này hơi 'lạc quan'")
    print("  vì model đã được chọn dựa trên chính nó.")
    print("- Test set chỉ dùng MỘT LẦN ở cuối -> ước lượng khách quan hơn về chất lượng thực tế.")
    print("- Chênh lệch giữa hai con số cho thấy mức độ model được 'chọn theo' validation.")

    # ---- Vẽ Confusion Matrix ----
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set (CT-AI 5.1)")
    plt.tight_layout()
    plt.savefig("04_05_confusion_matrix.png", dpi=150)
    plt.show()
    print("\nĐã lưu ảnh: 04_05_confusion_matrix.png")


if __name__ == "__main__":
    main()
