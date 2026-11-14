"""
ISTQB CT-AI - Chương 9 (mục 9.2.1) Hands-on: Pairwise Testing (H2)

Yêu cầu syllabus: hệ thống AI có >= 5 tham số và >= 500 tổ hợp. Dùng công cụ pairwise để
rút gọn tập tổ hợp, chạy test, rồi SO SÁNH số ca pairwise với số ca nếu test mọi tổ hợp.

Hệ thống AI được test: model dự đoán module "dễ lỗi" (defect prediction).
5 tham số, mỗi tham số 5 giá trị -> 5^5 = 3125 tổ hợp (>> 500).
Công cụ: thư viện allpairspy.
"""
import itertools
import numpy as np
import pandas as pd
from allpairspy import AllPairs
from sklearn.ensemble import RandomForestClassifier


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


def main():
    # 1. Hệ thống AI cần test
    df = make_defect_data()
    cols = ["loc", "complexity", "num_methods", "code_churn", "num_developers"]
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(df[cols], df["buggy"])

    # 2. Các tham số và giá trị (phân vùng tương đương + biên)
    parameters = {
        "loc":            [50, 500, 1000, 1500, 1999],
        "complexity":     [1, 10, 25, 40, 49],
        "num_methods":    [1, 25, 50, 75, 99],
        "code_churn":     [0, 100, 250, 400, 499],
        "num_developers": [1, 4, 8, 12, 14],
    }
    names = list(parameters)
    values = list(parameters.values())
    all_count = int(np.prod([len(v) for v in values]))

    print("=" * 64)
    print("PAIRWISE TESTING - Model dự đoán lỗi (CT-AI 9.2.1)")
    print("=" * 64)
    print(f"Số tham số: {len(names)} | giá trị mỗi tham số: {[len(v) for v in values]}")
    print(f"Số tổ hợp đầy đủ (all combinations): {all_count}")

    # 3. Sinh tập test pairwise
    pairwise_cases = [list(case) for case in AllPairs(values)]
    n_pair = len(pairwise_cases)

    # Kiểm tra: mọi CẶP giá trị của mọi CẶP tham số đều được phủ
    needed = set()
    for (i, vi), (j, vj) in itertools.combinations(enumerate(values), 2):
        for a in vi:
            for b in vj:
                needed.add((i, a, j, b))
    covered = set()
    for case in pairwise_cases:
        for i, j in itertools.combinations(range(len(names)), 2):
            covered.add((i, case[i], j, case[j]))
    print(f"Số cặp giá trị cần phủ: {len(needed)} | đã phủ: {len(needed & covered)} "
          f"-> {'ĐỦ 100%' if needed <= covered else 'THIẾU'}")

    # 4. Thực thi các test và kiểm tra tiêu chí đơn giản (không cần oracle chính xác)
    test_df = pd.DataFrame(pairwise_cases, columns=names)
    proba = model.predict_proba(test_df)[:, 1]
    pred = model.predict(test_df)
    test_df["P(buggy)"] = proba.round(3)
    test_df["pred"] = pred

    # Tiêu chí pass: xác suất hợp lệ [0,1] và dự đoán nhất quán với xác suất (>0.5 <=> buggy)
    valid = (proba >= 0) & (proba <= 1) & (((proba > 0.5).astype(int)) == pred)
    test_df["result"] = np.where(valid, "PASS", "FAIL")

    print(f"\nSố ca test pairwise: {n_pair}")
    print("\n10 ca test đầu tiên:")
    print(test_df.head(10).to_string(index=False))
    print(f"\nKết quả: PASS = {int(valid.sum())} | FAIL = {int((~valid).sum())}")
    print(f"Dự đoán 'buggy': {int(pred.sum())}/{n_pair} ca")

    # 5. So sánh
    print("\n" + "=" * 64)
    print("SO SÁNH")
    print("=" * 64)
    print(f"Tất cả tổ hợp : {all_count} ca")
    print(f"Pairwise      : {n_pair} ca")
    print(f"Tiết kiệm     : {(1 - n_pair / all_count) * 100:.1f}% số ca test")
    print("Ý nghĩa: phần lớn lỗi do tương tác của 1–2 tham số, nên pairwise phát hiện được")
    print("nhiều lỗi với chi phí thấp hơn rất nhiều so với test toàn bộ tổ hợp.")

    test_df.to_csv("09_pairwise_testcases.csv", index=False)
    print("Đã lưu: 09_pairwise_testcases.csv")


if __name__ == "__main__":
    main()