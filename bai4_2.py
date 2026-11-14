import os
import warnings
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
warnings.filterwarnings("ignore")
print("=" * 65)
print("BAI 4 - HOI QUY DU DOAN GIA NHA")
print("=" * 65)
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
file_path = filedialog.askopenfilename(
    title="Chon file kc_house_data.csv",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
root.destroy()
if not file_path:
    file_path = input(
        "Nhap duong dan file kc_house_data.csv: "
    ).strip().strip('"')
if not os.path.isfile(file_path):
    raise FileNotFoundError(
        "Khong tim thay file CSV. "
        "Hay chay lai va chon dung file."
    )
print("\nFile duoc chon:", file_path)
df = pd.read_csv(file_path)
print("\n5 dong dau tien:")
print(df.head())
print("\nKich thuoc ban dau:", df.shape)
print("\nDanh sach cot:")
print(df.columns.tolist())
if "price" not in df.columns:
    raise ValueError("Du lieu khong co cot price.")
df["price"] = pd.to_numeric(
    df["price"], errors="coerce"
)
df = df.dropna(subset=["price"]).copy()
df = df[df["price"] > 0].copy()
df = df.drop_duplicates().copy()
print("\nKich thuoc sau khi lam sach:", df.shape)
if "date" in df.columns:
    date = pd.to_datetime(
        df["date"].astype(str),
        errors="coerce"
    )
    df["sale_year"] = date.dt.year
    df["sale_month"] = date.dt.month
    df.drop(columns=["date"], inplace=True)
if "yr_built" in df.columns:
    if "sale_year" in df.columns:
        df["house_age"] = (
            df["sale_year"] - df["yr_built"]
        ).clip(lower=0)
    else:
        df["house_age"] = (
            2026 - df["yr_built"]
        ).clip(lower=0)
if "yr_renovated" in df.columns:
    renovation_year = pd.to_numeric(
        df["yr_renovated"], errors="coerce"
    )
    df["renovated"] = (
        renovation_year.fillna(0) > 0
    ).astype(int)
    if "sale_year" in df.columns:
        years_since = (
            df["sale_year"] - renovation_year
        ).clip(lower=0)

        df["years_since_renovation"] = np.where(
            renovation_year.fillna(0) > 0,
            years_since,
            0
        )
if "sqft_living" in df.columns and "sqft_lot" in df.columns:
    df["living_lot_ratio"] = (
        df["sqft_living"] /
        df["sqft_lot"].replace(0, np.nan)
    )
if "bathrooms" in df.columns and "bedrooms" in df.columns:
    df["bath_bed_ratio"] = (
        df["bathrooms"] /
        df["bedrooms"].replace(0, np.nan)
    )
y = df["price"].copy()
X = df.drop(
    columns=["price", "id"],
    errors="ignore"
).copy()
X = X.select_dtypes(include=[np.number])
X = X.dropna(axis=1, how="all")
constant_cols = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]
X = X.drop(columns=constant_cols)
if X.shape[1] == 0:
    raise ValueError("Khong co dac trung hop le de huan luyen.")
print("\nSo dac trung:", X.shape[1])
print("Cac dac trung:")
print(X.columns.tolist())
print("\nThong ke gia nha:")
print(y.describe())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nSo mau train:", len(X_train))
print("So mau test:", len(X_test))
train_corr = X_train.copy()
train_corr["price"] = y_train
correlations = (
    train_corr.corr(numeric_only=True)["price"]
    .drop("price")
    .abs()
    .dropna()
    .sort_values(ascending=False)
)
top_features = correlations.head(
    min(10, len(correlations))
).index.tolist()
print("\nTop 10 dac trung theo tuong quan:")
print(correlations.head(10))
print("\nDanh sach Top features:")
print(top_features)
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
def create_models():
    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=5,
                random_state=42
            ))
        ]),

        "SVR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=100,
                epsilon=0.1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ))
        ])
    }
def evaluate_models(Xtr, Xte, ytr, yte, feature_label):
    results = []
    predictions = {}
    fitted_models = {}
    models = create_models()
    for name, model in models.items():
        print(
            "\nDang huan luyen:",
            name,
            "|",
            feature_label
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2 = r2_score(yte, pred)
        results.append({
            "Feature set": feature_label,
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        predictions[name] = pred
        fitted_models[name] = model
        print(f"MAE  = {mae:,.2f}")
        print(f"RMSE = {rmse:,.2f}")
        print(f"R2   = {r2:.4f}")

    return results, predictions, fitted_models
results_all, predictions_all, models_all = evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    "All features"
)
results_top, predictions_top, models_top = evaluate_models(
    X_train_top,
    X_test_top,
    y_train,
    y_test,
    "Top 10 features"
)
results_df = pd.DataFrame(results_all + results_top)
results_df = results_df.sort_values(
    by="R2",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 85)
print("BANG SO SANH CAC MO HINH")
print("=" * 85)

print(results_df.round(4).to_string(index=False))
output_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(
    output_dir,
    "ket_qua_bai4.csv"
)

results_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)
print("\nDa luu bang ket qua tai:")
print(result_path)
for metric in ["MAE", "RMSE", "R2"]:
    plt.figure(figsize=(13, 6))

    labels = (
        results_df["Model"] + "\n" +
        results_df["Feature set"]
    )

    bars = plt.bar(labels, results_df[metric])

    plt.title(f"So sanh cac mo hinh theo {metric}")
    plt.xlabel("Mo hinh va bo dac trung")
    plt.ylabel(metric)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()
best_row = results_df.loc[results_df["R2"].idxmax()]
best_name = best_row["Model"]
best_feature_set = best_row["Feature set"]
if best_feature_set == "All features":
    best_pred = predictions_all[best_name]
else:
    best_pred = predictions_top[best_name]
print("\n" + "=" * 65)
print("MO HINH TOT NHAT THEO R2")
print("=" * 65)
print("Mo hinh:", best_name)
print("Bo dac trung:", best_feature_set)
print(f"MAE:  {best_row['MAE']:,.2f}")
print(f"RMSE: {best_row['RMSE']:,.2f}")
print(f"R2:   {best_row['R2']:.4f}")
plt.figure(figsize=(8, 8))
plt.scatter(
    y_test,
    best_pred,
    alpha=0.5
)
min_value = min(y_test.min(), best_pred.min())
max_value = max(y_test.max(), best_pred.max())
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Gia nha thuc te")
plt.ylabel("Gia nha du doan")
plt.title(f"Gia thuc te va du doan - {best_name}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
gb_model = models_all["Gradient Boosting"]
gb_estimator = gb_model.named_steps["model"]
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": gb_estimator.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)
print("\nTop 10 dac trung quan trong:")
print(importance_df.head(10).to_string(index=False))

top_importance = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)
plt.figure(figsize=(10, 6))

plt.barh(
    top_importance["Feature"],
    top_importance["Importance"]
)
plt.xlabel("Feature importance")
plt.ylabel("Dac trung")
plt.title("Top 10 dac trung quan trong - Gradient Boosting")
plt.tight_layout()
plt.show()
print("\n" + "=" * 65)
print("NHAN XET KET QUA")
print("=" * 65)
print(
    f"1. Mo hinh tot nhat theo R2 la {best_name}, "
    f"voi bo dac trung {best_feature_set}."
)
print(
    f"2. MAE = {best_row['MAE']:,.2f}; "
    f"RMSE = {best_row['RMSE']:,.2f}; "
    f"R2 = {best_row['R2']:.4f}."
)
if best_row["R2"] >= 0.8:
    print(
        "3. Mo hinh giai thich duoc phan lon su bien thien "
        "gia nha tren tap test."
    )
elif best_row["R2"] >= 0.5:
    print(
        "3. Mo hinh co kha nang du bao kha, "
        "nhung van con sai so."
    )
else:
    print(
        "3. Kha nang du bao con han che; "
        "can xem xet them dac trung va tham so."
    )

best_all = results_df[
    results_df["Feature set"] == "All features"
].sort_values("R2", ascending=False).iloc[0]

best_top = results_df[
    results_df["Feature set"] == "Top 10 features"
].sort_values("R2", ascending=False).iloc[0]

print(
    f"4. R2 cao nhat voi toan bo dac trung: "
    f"{best_all['Model']} = {best_all['R2']:.4f}."
)

print(
    f"5. R2 cao nhat voi Top 10 dac trung: "
    f"{best_top['Model']} = {best_top['R2']:.4f}."
)

if best_all["R2"] > best_top["R2"]:
    print(
        "6. Trong lan chay nay, toan bo dac trung cho "
        "R2 cao hon Top 10 dac trung."
    )
elif best_all["R2"] < best_top["R2"]:
    print(
        "6. Trong lan chay nay, Top 10 dac trung cho "
        "R2 cao hon toan bo dac trung."
    )
else:
    print("6. Hai bo dac trung co R2 bang nhau.")

print("\nHOAN THANH BAI 4!")
