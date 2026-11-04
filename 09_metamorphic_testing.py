"""
ISTQB CT-AI - Chương 9
Hands-on: Metamorphic Testing (H2)
Ví dụ với hàm tính trung bình + một số quan hệ metamorphic phổ biến
"""

import random
import numpy as np


def average(nums):
    """Hàm cần kiểm thử"""
    if len(nums) == 0:
        raise ValueError("Danh sách rỗng")
    return sum(nums) / len(nums)


def test_metamorphic_relations():
    print("=" * 60)
    print("METAMORPHIC TESTING - Hàm average()")
    print("=" * 60)

    # Source test case
    src = [3, 5, 8, 10, 12]
    expected = average(src)
    print(f"Source input : {src}")
    print(f"Source output: {expected:.4f}\n")

    # ---------- MR1: Hoán vị (Permutation) ----------
    # Đổi thứ tự phần tử → kết quả không đổi
    shuffled = src[:]
    random.shuffle(shuffled)
    result_mr1 = average(shuffled)
    assert abs(result_mr1 - expected) < 1e-9, "MR1 thất bại!"
    print(f"[MR1 - Permutation] Input: {shuffled}")
    print(f"  Output: {result_mr1:.4f} → PASSED ✓")

    # ---------- MR2: Nhân với hằng số ----------
    # Nhân mỗi phần tử với k → kết quả cũng nhân với k
    k = 2.5
    scaled = [x * k for x in src]
    result_mr2 = average(scaled)
    assert abs(result_mr2 - expected * k) < 1e-9, "MR2 thất bại!"
    print(f"\n[MR2 - Scale by {k}] Input: {scaled}")
    print(f"  Output: {result_mr2:.4f} (expected {expected * k:.4f}) → PASSED ✓")

    # ---------- MR3: Thêm phần tử trung bình ----------
    # Thêm giá trị = average hiện tại → average không đổi
    new_list = src + [expected]
    result_mr3 = average(new_list)
    assert abs(result_mr3 - expected) < 1e-9, "MR3 thất bại!"
    print(f"\n[MR3 - Add mean value] Input: {new_list}")
    print(f"  Output: {result_mr3:.4f} → PASSED ✓")

    # ---------- MR4: Cộng hằng số ----------
    # Cộng c vào mỗi phần tử → average cũng cộng c
    c = 7
    shifted = [x + c for x in src]
    result_mr4 = average(shifted)
    assert abs(result_mr4 - (expected + c)) < 1e-9, "MR4 thất bại!"
    print(f"\n[MR4 - Add constant {c}] Input: {shifted}")
    print(f"  Output: {result_mr4:.4f} (expected {expected + c:.4f}) → PASSED ✓")

    print("\n" + "=" * 60)
    print("Tất cả Metamorphic Relations đều PASSED!")
    print("=" * 60)


def additional_example_ml():
    """
    Ví dụ Metamorphic cho mô hình phân loại ảnh (ý tưởng):
    - MR: Lật ảnh ngang → nhãn không đổi (với nhiều lớp)
    - MR: Thêm nhiễu nhỏ → nhãn không đổi (robustness)
    Ở đây chỉ minh họa ý tưởng.
    """
    print("\n\n--- Ý tưởng Metamorphic Testing cho AI/ML ---")
    print("Ví dụ với mô hình phân loại:")
    print("  MR1: Xoay ảnh nhỏ (≤ 5°) → nhãn giữ nguyên")
    print("  MR2: Thêm Gaussian noise nhẹ → nhãn giữ nguyên")
    print("  MR3: Lật ngang ảnh (với đối tượng đối xứng) → nhãn giữ nguyên")
    print("  → Đây là cách kiểm thử khi không có oracle chính xác 100%.")


if __name__ == "__main__":
    test_metamorphic_relations()
    additional_example_ml()
