# ISTQB CT-AI Hands-on Demos

Bộ demo các bài Hands-on theo syllabus **Certified Tester AI Testing (CT-AI) v1.0** (mục 0.4).
Ngày bảo vệ: **10/11**.

## Đối chiếu với syllabus

| File | Mục syllabus | Mức | Nội dung |
|------|--------------|-----|----------|
| `03_overfitting_underfitting.py` | 3.5.3 | H0 | Đa thức bậc 1/4/15, so sánh sai số train và cross-validation |
| `04_05_data_prep_metrics.py` | 4.1.2, 4.2.1, 5.4.1 | H2 | Làm sạch dữ liệu bẩn, chuẩn hoá, chia 60/20/20, tinh chỉnh bằng validation, chấm bằng test, Accuracy/Precision/Recall/F1/Confusion Matrix |
| `06_perceptron_and.py` | 6.1.1 | H1 | Perceptron học hàm AND qua các epoch đến khi error = 0 |
| `08_lime_explainability.py` | 8.6.1 | H2 | Giải thích dự đoán bằng LIME (một mẫu đúng, một mẫu sai) |
| `09_pairwise_testing.py` | 9.2.1 | H2 | 5 tham số, 3125 tổ hợp, rút gọn còn ~31 ca bằng `allpairspy` |
| `09_metamorphic_testing.py` | 9.5.1 | H2 | MR cho `average()`, model Iris và model dự đoán lỗi |
| `09_exploratory_eda.py` | 9.6.1 | H2 | Data tour, dữ liệu thiếu, mất cân bằng nhãn, bias, ngoại lệ |
| `11_defect_prediction.py` | 11.5.1 | H2 | Dự đoán lỗi, so sánh nhiều tập feature |

Ghi chú: dữ liệu defect là **giả lập**; dữ liệu "thô" trong `04_05` là Iris được cố ý chèn lỗi
(thiếu, trùng, ngoại lệ). Bài 11.1.1 (Use of AI in Testing) và phần "Discuss activities where AI
is less likely to be used" chưa có code (là bài thảo luận).

## Cài đặt

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Chạy

```bash
python 03_overfitting_underfitting.py
python 04_05_data_prep_metrics.py
python 06_perceptron_and.py
python 08_lime_explainability.py
python 09_pairwise_testing.py
python 09_metamorphic_testing.py
python 09_exploratory_eda.py
python 11_defect_prediction.py
```

Mỗi script in kết quả ra terminal và lưu ảnh `.png` (hoặc `.csv`) để đưa vào slide.

## Yêu cầu

Python 3.9+, không cần GPU.
