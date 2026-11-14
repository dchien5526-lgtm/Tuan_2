import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
diabetes = load_diabetes(as_frame=True)
X = diabetes.data.copy()
y = diabetes.target.copy()
df = X.copy()
df["target"] = y
print("Kich thuoc du lieu:", df.shape)
print("\n10 feature trong bo du lieu:")
print(X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nSo mau train:", len(X_train))
print("So mau test:", len(X_test))
train_corr = X_train.copy()
train_corr["target"] = y_train
correlations = train_corr.corr()["target"].drop("target")
top5_features = correlations.abs().sort_values(
    ascending=False
).head(5).index.tolist()
print("\nTop 5 feature theo tuong quan:")
print(top5_features)
feature_sets = {
    "bmi": ["bmi"],
    "bmi + bp": ["bmi", "bp"],
    "Top 5 features": top5_features,
    "All 10 features": X.columns.tolist()
}
results = []
for name, selected_features in feature_sets.items():
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Bo feature": name,
        "So feature": len(selected_features),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })
    print(f"\nMo hinh: {name}")
    print("Feature su dung:", selected_features)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
results_df = pd.DataFrame(results)
print("\n===== BANG SO SANH KET QUA =====")
print(results_df.round(4).to_string(index=False))
metrics = ["MAE", "RMSE", "R2"]
for metric in metrics:
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        results_df["Bo feature"],
        results_df[metric],
        edgecolor="black"
    )
    plt.title(f"So sanh cac mo hinh theo {metric}")
    plt.xlabel("Bo feature")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )
    plt.tight_layout()
    plt.show()
best_model = results_df.loc[results_df["R2"].idxmax()]
print("\n===== NHAN XET =====")
print("Bo feature co R2 cao nhat:", best_model["Bo feature"])
print(f"MAE:  {best_model['MAE']:.4f}")
print(f"RMSE: {best_model['RMSE']:.4f}")
print(f"R2:   {best_model['R2']:.4f}")
print("\nLuu y:")
print("- MAE cang nho thi sai so tuyet doi trung binh cang thap.")
print("- RMSE cang nho thi sai so du doan cang thap.")
print("- R2 cang gan 1 thi mo hinh giai thich du lieu cang tot.")
