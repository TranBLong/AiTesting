"""
ISTQB CT-AI - Chương 6
Hands-on: Implement a Simple Perceptron (H1)
Học hàm logic AND
"""

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=10):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        """Step function"""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        print("Bắt đầu huấn luyện Perceptron cho hàm AND...\n")
        for epoch in range(self.n_epochs):
            total_error = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                error = y[idx] - y_pred
                # Cập nhật trọng số
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error
                total_error += abs(error)

            print(f"Epoch {epoch+1:2d} | Weights: {self.weights} | Bias: {self.bias:.2f} | Error: {total_error}")
            if total_error == 0:
                print("→ Hội tụ sớm!")
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)


def main():
    # Dữ liệu hàm AND
    # Input: (0,0), (0,1), (1,0), (1,1)
    # Output: 0, 0, 0, 1
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    print("=" * 50)
    print("PERCEPTRON HỌC HÀM LOGIC AND")
    print("=" * 50)
    print("Dữ liệu huấn luyện:")
    for i in range(len(X)):
        print(f"  {X[i]} → {y[i]}")

    model = Perceptron(learning_rate=0.1, n_epochs=20)
    model.fit(X, y)

    print("\n" + "=" * 50)
    print("KẾT QUẢ DỰ ĐOÁN SAU HUẤN LUYỆN")
    print("=" * 50)
    predictions = model.predict(X)
    for i in range(len(X)):
        status = "✓" if predictions[i] == y[i] else "✗"
        print(f"  {X[i]} → Dự đoán: {predictions[i]} | Thực tế: {y[i]} {status}")

    print(f"\nTrọng số cuối cùng: {model.weights}")
    print(f"Bias cuối cùng   : {model.bias:.2f}")


if __name__ == "__main__":
    main()
