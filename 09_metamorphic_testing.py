"""
ISTQB CT-AI - Chương 9 (mục 9.5.1) Hands-on: Metamorphic Testing (H2)

Phần A: Khởi động với hàm average() (ví dụ trong syllabus).
Phần B: Áp dụng cho HỆ THỐNG AI THẬT:
  - Model phân loại hoa Iris      (MR giữ nguyên kết quả)
  - Model dự đoán lỗi (defect)    (MR kết quả thay đổi theo hướng xác định)

Khái niệm: Source test case -> biến đổi theo Metamorphic Relation (MR) -> Follow-up test case.
Không cần biết "đáp án đúng" tuyệt đối, chỉ kiểm tra QUAN HỆ giữa các đầu ra.
"""
import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------- Phần A
def average(nums):
    if len(nums) == 0:
        raise ValueError("Danh sách rỗng")
    return sum(nums) / len(nums)


def part_a_average():
    print("=" * 64)
    print("PHẦN A - METAMORPHIC TESTING cho hàm average()")
    print("=" * 64)
    src = [3, 5, 8, 10, 12]
    expected = average(src)
    print(f"Source test case: {src} -> {expected:.4f}")

    shuffled = src[:]
    random.shuffle(shuffled)
    assert abs(average(shuffled) - expected) < 1e-9
    print(f"[MR1 Hoán vị     | kết quả GIỮ NGUYÊN] {shuffled} -> PASSED")

    k = 2.5
    assert abs(average([x * k for x in src]) - expected * k) < 1e-9
    print(f"[MR2 Nhân {k}     | kết quả NHÂN {k}]    -> PASSED")

    assert abs(average(src + [expected]) - expected) < 1e-9
    print("[MR3 Thêm giá trị trung bình | GIỮ NGUYÊN] -> PASSED")

    c = 7
    assert abs(average([x + c for x in src]) - (expected + c)) < 1e-9
    print(f"[MR4 Cộng {c}      | kết quả CỘNG {c}]    -> PASSED")


# ----------------------------------------------------------------- Phần B1
def part_b1_iris_classifier():
    print("\n" + "=" * 64)
    print("PHẦN B1 - METAMORPHIC TESTING cho model phân loại Iris")
    print("=" * 64)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Source test cases: dự đoán ban đầu trên tập test
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    source_pred = model.predict(X_te)
    n = len(X_te)

    # MR-A: thêm nhiễu rất nhỏ (±0.01 cm) -> nhãn phải giữ nguyên  (robustness)
    rng = np.random.RandomState(0)
    noisy = X_te + rng.normal(0, 0.01, X_te.shape)
    same = int((model.predict(noisy) == source_pred).sum())
    print(f"[MR-A Nhiễu nhỏ (σ=0.01cm)   | nhãn GIỮ NGUYÊN] {same}/{n} mẫu thoả MR "
          f"-> {'PASSED' if same == n else 'có vi phạm'}")

    # MR-B: đổi đơn vị cm -> mm cho TOÀN BỘ dữ liệu (train + test); model có chuẩn hoá
    #       -> dự đoán phải giữ nguyên.
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    pred_cm = pipe.predict(X_te)
    pipe_mm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr * 10, y_tr)
    pred_mm = pipe_mm.predict(X_te * 10)
    same = int((pred_cm == pred_mm).sum())
    print(f"[MR-B Đổi đơn vị cm->mm      | nhãn GIỮ NGUYÊN] {same}/{n} mẫu thoả MR "
          f"-> {'PASSED' if same == n else 'có vi phạm'}")

    # MR-C: đảo thứ tự dữ liệu huấn luyện -> dự đoán gần như không đổi
    perm = np.random.RandomState(1).permutation(len(X_tr))
    pipe2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr[perm], y_tr[perm])
    same = int((pipe2.predict(X_te) == pred_cm).sum())
    print(f"[MR-C Đảo thứ tự dữ liệu train| nhãn GIỮ NGUYÊN] {same}/{n} mẫu thoả MR "
          f"-> {'PASSED' if same == n else 'có vi phạm'}")

    # MR-D (kết quả KHÁC): cho mẫu setosa, tăng mạnh petal length/width lên mức của virginica
    #       -> nhãn KHÔNG được còn là setosa.
    setosa = X_te[source_pred == 0]
    moved = setosa.copy()
    moved[:, 2] = 6.0  # petal length
    moved[:, 3] = 2.0  # petal width
    still = int((model.predict(moved) == 0).sum())
    print(f"[MR-D Tăng petal -> cỡ virginica | nhãn PHẢI ĐỔI khỏi setosa] "
          f"{len(setosa) - still}/{len(setosa)} mẫu đổi nhãn "
          f"-> {'PASSED' if still == 0 else 'có vi phạm'}")


# ----------------------------------------------------------------- Phần B2
def make_defect_data(n=800, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "loc": rng.randint(50, 2000, n),
        "complexity": rng.randint(1, 50, n),
        "num_methods": rng.randint(1, 100, n),
        "code_churn": rng.randint(0, 500, n),
        "num_developers": rng.randint(1, 15, n),
    })
    score = (0.001 * df["loc"] + 0.05 * df.complexity + 0.01 * df.code_churn
             + 0.1 * df.num_developers - 0.02 * df.num_methods)
    prob = 1 / (1 + np.exp(-score + 3))
    df["buggy"] = (rng.rand(n) < prob).astype(int)
    return df


def part_b2_defect_model():
    print("\n" + "=" * 64)
    print("PHẦN B2 - METAMORPHIC TESTING cho model dự đoán lỗi (không có oracle)")
    print("=" * 64)
    df = make_defect_data()
    cols = ["loc", "complexity", "num_methods", "code_churn", "num_developers"]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[cols], df["buggy"])

    # Source test cases: 200 module bất kỳ (không cần biết đáp án đúng)
    src = df[cols].sample(200, random_state=7).reset_index(drop=True)
    p_src = model.predict_proba(src)[:, 1]

    # MR-E: tăng code_churn thêm 100 -> xác suất lỗi KHÔNG ĐƯỢC giảm đáng kể (kết quả đổi theo hướng xác định)
    follow = src.copy()
    follow["code_churn"] = follow["code_churn"] + 100
    p_fol = model.predict_proba(follow)[:, 1]
    violated = int((p_fol < p_src - 0.05).sum())
    print(f"[MR-E code_churn +100 | xác suất lỗi KHÔNG GIẢM > 0.05] "
          f"vi phạm {violated}/200 module ({violated / 2:.1f}%)")

    # MR-F: tăng loc thêm 500 -> xác suất lỗi không giảm đáng kể
    follow = src.copy()
    follow["loc"] = follow["loc"] + 500
    p_fol = model.predict_proba(follow)[:, 1]
    violated_f = int((p_fol < p_src - 0.05).sum())
    print(f"[MR-F loc +500        | xác suất lỗi KHÔNG GIẢM > 0.05] "
          f"vi phạm {violated_f}/200 module ({violated_f / 2:.1f}%)")

    print("\nNhận xét: các vi phạm (nếu có) là 'ứng viên lỗi' để tester điều tra: model có thể")
    print("chưa học đúng quan hệ đơn điệu (monotonic) giữa chỉ số code và khả năng lỗi.")
    print("(Lưu ý theo syllabus: MT không đảm bảo source test case đúng, chỉ kiểm tra quan hệ.)")


def main():
    part_a_average()
    part_b1_iris_classifier()
    part_b2_defect_model()
    print("\n" + "=" * 64)
    print("Kết luận: MT giúp kiểm thử AI khi KHÔNG có oracle (đáp án đúng 100%).")
    print("=" * 64)


if __name__ == "__main__":
    main()
