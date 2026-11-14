import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
diabetes = load_diabetes(as_frame=True)
X = diabetes.data.copy()
y = diabetes.target.copy()
print("Kich thuoc du lieu:", X.shape)
print("So mau:", len(X))
print("So feature:", X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nSo mau train:", len(X_train))
print("So mau test:", len(X_test))
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    ),
    "SVR": make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=100, epsilon=0.1)
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}
results = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Mo hinh": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })
    print("\nMo hinh:", name)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
results_df = pd.DataFrame(results)
print("\n========== BANG SO SANH ==========")
print(results_df.round(4).to_string(index=False))
metrics = ["MAE", "RMSE", "R2"]
for metric in metrics:
    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        results_df["Mo hinh"],
        results_df[metric],
        edgecolor="black"
    )
    plt.title(f"So sanh cac mo hinh theo {metric}")
    plt.xlabel("Mo hinh")
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )
    plt.tight_layout()
    plt.show()
best_model_name = results_df.loc[
    results_df["R2"].idxmax(), "Mo hinh"
]
best_pred = predictions[best_model_name]
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.7)
min_value = min(y_test.min(), best_pred.min())
max_value = max(y_test.max(), best_pred.max())
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--",
    label="Du doan ly tuong"
)
plt.xlabel("Gia tri thuc te")
plt.ylabel("Gia tri du doan")
plt.title(f"Thuc te va du doan - {best_model_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
best_mae = results_df.loc[results_df["MAE"].idxmin()]
best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
best_r2 = results_df.loc[results_df["R2"].idxmax()]
print("\n" + "=" * 55)
print("NHAN XET KET QUA BAI 3")
print("=" * 55)
print("\n1. Theo MAE:")
print(
    f"Mo hinh tot nhat: {best_mae['Mo hinh']}, "
    f"MAE = {best_mae['MAE']:.4f}"
)
print("MAE cang nho, sai so tuyet doi trung binh cang thap.")
print("\n2. Theo RMSE:")
print(
    f"Mo hinh tot nhat: {best_rmse['Mo hinh']}, "
    f"RMSE = {best_rmse['RMSE']:.4f}"
)
print("RMSE cang nho, sai so du doan cang thap.")
print("\n3. Theo R2:")
print(
    f"Mo hinh tot nhat: {best_r2['Mo hinh']}, "
    f"R2 = {best_r2['R2']:.4f}"
)
print("R2 cang cao thi kha nang giai thich bien dong cua target cang tot.")
print("\n4. Ket qua tung mo hinh:")
for _, row in results_df.iterrows():
    print(
        f"- {row['Mo hinh']}: "
        f"MAE = {row['MAE']:.4f}, "
        f"RMSE = {row['RMSE']:.4f}, "
        f"R2 = {row['R2']:.4f}"
    )
print("\n5. Ket luan:")
print(
    f"Tren tap test nay, {best_r2['Mo hinh']} "
    "co R2 cao nhat trong cac mo hinh duoc thu nghiem."
)
print(
    "Can xem dong thoi MAE, RMSE va R2 de danh gia. "
    "Ket qua co the thay doi khi thay doi cach chia du lieu "
    "hoac tham so mo hinh."
)
