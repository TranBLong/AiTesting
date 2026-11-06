"""
ISTQB CT-AI - Chương 9 (mục 9.6.1) Hands-on: Exploratory Testing & Exploratory Data Analysis (EDA) (H2)

Yêu cầu syllabus: với một model + dataset, thực hiện "data tour" (xem các loại dữ liệu và
phân bố của từng tham số), làm EDA để tìm DỮ LIỆU THIẾU và/hoặc BIAS tiềm ẩn.

Dataset: dữ liệu defect (mô phỏng) được CỐ Ý làm "bẩn" để EDA có thứ để tìm:
  - thiếu dữ liệu ở cột code_churn
  - dữ liệu lệch: gần như toàn bộ module từ nhóm ít developer (sampling bias)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def make_dataset(n=800, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "loc": rng.randint(50, 2000, n),
        "complexity": rng.randint(1, 50, n),
        "num_methods": rng.randint(1, 100, n),
        "code_churn": rng.randint(0, 500, n).astype(float),
        # sampling bias: 90% module đến từ team nhỏ (1-4 dev)
        "num_developers": np.where(rng.rand(n) < 0.9, rng.randint(1, 5, n), rng.randint(5, 15, n)),
    })
    score = (0.001 * df["loc"] + 0.05 * df.complexity + 0.01 * df.code_churn
             + 0.1 * df.num_developers - 0.02 * df.num_methods)
    prob = 1 / (1 + np.exp(-score + 3))
    df["buggy"] = (rng.rand(n) < prob).astype(int)
    # Làm bẩn: 12% code_churn bị thiếu, tập trung ở module có lỗi (thiếu không ngẫu nhiên)
    miss = (df["buggy"] == 1) & (rng.rand(n) < 0.18)
    df.loc[miss, "code_churn"] = np.nan
    return df


def section(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


def main():
    df = make_dataset()

    section("1. DATA TOUR - kiểu dữ liệu và thống kê từng cột")
    print(df.dtypes.to_string())
    print("\n", df.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())

    section("2. DỮ LIỆU THIẾU (missing data)")
    miss = df.isna().sum()
    print(miss[miss > 0].to_string() if miss.sum() else "Không có dữ liệu thiếu")
    rate = df.groupby("buggy")["code_churn"].apply(lambda s: s.isna().mean() * 100)
    print("\nTỷ lệ thiếu code_churn theo nhãn (%):")
    print(rate.round(1).rename({0: "clean", 1: "buggy"}).to_string())
    if rate.max() > 2 * max(rate.min(), 0.5):
        print("⚠ Dữ liệu thiếu KHÔNG ngẫu nhiên (tập trung ở một nhãn) -> điền bừa có thể sinh bias.")

    section("3. CÂN BẰNG NHÃN (class imbalance)")
    dist = df["buggy"].value_counts(normalize=True).sort_index() * 100
    print(dist.round(1).rename({0: "clean", 1: "buggy"}).to_string())
    if dist.max() > 70:
        print("⚠ Nhãn lệch (>70% một lớp): accuracy đơn thuần dễ gây hiểu nhầm; cần xem precision/recall.")

    section("4. BIAS TIỀM ẨN (sampling bias theo nhóm số developer)")
    df["team_size"] = pd.cut(df["num_developers"], bins=[0, 4, 14], labels=["nhóm nhỏ (1-4)", "nhóm lớn (5-14)"])
    grp = df.groupby("team_size", observed=True).agg(so_mau=("buggy", "size"), ti_le_loi=("buggy", "mean"))
    grp["ti_le_loi"] = (grp["ti_le_loi"] * 100).round(1)
    print(grp.to_string())
    share = grp["so_mau"].max() / grp["so_mau"].sum() * 100
    if share > 75:
        print(f"⚠ {share:.0f}% mẫu đến từ một nhóm -> model có thể hoạt động kém với nhóm còn lại.")

    section("5. NGOẠI LỆ (outliers - quy tắc IQR)")
    for c in ["loc", "complexity", "num_methods", "code_churn"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        print(f"{c:<15}: {n_out} ngoại lệ")

    # Vẽ biểu đồ
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.flat[:4], ["loc", "complexity", "num_methods", "code_churn"]):
        sns.histplot(df[c].dropna(), bins=30, ax=ax, color="steelblue")
        ax.set_title(f"Phân bố {c}")
    sns.countplot(x="num_developers", data=df, ax=axes.flat[4], color="orange")
    axes.flat[4].set_title("Số developer (lệch về nhóm nhỏ)")
    sns.heatmap(df[["loc", "complexity", "num_methods", "code_churn", "num_developers", "buggy"]].corr(),
                annot=True, fmt=".2f", cmap="coolwarm", ax=axes.flat[5])
    axes.flat[5].set_title("Tương quan")
    plt.suptitle("EDA - CT-AI 9.6.1", fontsize=14)
    plt.tight_layout()
    plt.savefig("09_eda.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\nĐã lưu ảnh: 09_eda.png")

    section("KẾT LUẬN (góc nhìn Exploratory Testing)")
    print("EDA là bước 'khám phá' dữ liệu: các phát hiện trên (thiếu không ngẫu nhiên, nhãn lệch,")
    print("mẫu lệch nhóm) trở thành giả thuyết để thiết kế test tiếp theo cho model.")


if __name__ == "__main__":
    main()
