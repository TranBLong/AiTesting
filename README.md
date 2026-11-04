# ISTQB CT-AI Hands-on Demos

Bộ demo đầy đủ các bài Hands-on theo syllabus **Certified Tester AI Testing (CT-AI) v1.0**.

Ngày bảo vệ: **10/11**.

## Cấu trúc

| File | Chương | Nội dung Hands-on |
|------|--------|-------------------|
| `03_overfitting_underfitting.py` | 3 | Demonstrate Overfitting & Underfitting |
| `04_05_data_prep_metrics.py` | 4 + 5 | Data Preparation + Train/Val/Test + Accuracy/Precision/Recall/F1 + Confusion Matrix |
| `06_perceptron_and.py` | 6 | Implement a Simple Perceptron (học hàm AND) |
| `08_lime_explainability.py` | 8 | Model Explainability với LIME |
| `09_metamorphic_testing.py` | 9 | Metamorphic Testing |
| `11_defect_prediction.py` | 11 | Build a simple Defect Prediction System |

## Cài đặt

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv

# Kích hoạt
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Cài thư viện
pip install -r requirements.txt
```

## Cách chạy từng demo

```bash
python 03_overfitting_underfitting.py
python 04_05_data_prep_metrics.py
python 06_perceptron_and.py
python 08_lime_explainability.py
python 09_metamorphic_testing.py
python 11_defect_prediction.py
```

Mỗi script sẽ in kết quả ra terminal và lưu ảnh `.png` (nếu có) để đưa vào slide.

## Gợi ý trình bày

1. Chạy live trong VS Code / terminal khi bảo vệ.
2. Hoặc chạy trước, chụp ảnh kết quả + ảnh đồ thị đưa vào slide.
3. Giải thích ngắn gọn:
   - Mục tiêu bài hands-on là gì
   - Code làm gì
   - Kết quả ý nghĩa gì đối với kiểm thử AI

## Yêu cầu hệ thống

- Python 3.9+
- Không cần GPU
- Chạy được trên Windows / macOS / Linux

---