"""
BAI 4_2 - HUAN LUYEN MO HINH CHO BO DU LIEU KC_HOUSE_DATA THEO PIPELINE

Yeu cau de bai:
  - Huan luyen mo hinh theo pipeline
  - Day du cac truong hop Feature Engineering / Feature Selection
  - Nhieu mo hinh hoc may khac nhau
  - So sanh hieu nang tren cac chi so danh gia

Thiet ke thi nghiem (moi giai doan chi thay doi mot yeu to):
  Giai doan 1 - Feature Engineering: Selection co dinh = "All",
                so sanh 3 muc FE: Raw, FE, FE + log(price)        -> 18 cau hinh
  Giai doan 2 - Feature Selection: FE co dinh = muc tot nhat GD1,
                so sanh 4 cach chon: All, Pearson Top-10,
                Mutual Info Top-10, Lasso                           -> 24 cau hinh
  Giai doan 3 - 5-fold Cross Validation cho 3 cau hinh tot nhat

Moi ket qua (CSV + hinh) duoc luu trong thu muc "ket_qua_bai4" canh file code.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TOP_K = 10          # so dac trung giu lai o cac cach chon Top-K
RUN_SVR = True      # dat False neu may cham (SVR chay lau nhat)
RUN_CV = True       # dat False de bo qua giai doan 3
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ket_qua_bai4")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("BAI 4_2 - HOI QUY DU DOAN GIA NHA THEO PIPELINE")
print("=" * 65)

# =================================================================
# 1. CHON FILE VA DOC DU LIEU
# =================================================================
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    import tkinter as tk
    from tkinter import filedialog
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
if "price" not in df.columns:
    raise ValueError(
        "Du lieu khong co cot price. Co the ban da chon nham file ket qua "
        "(ket_qua_bai4...csv). Hay chon file du lieu goc kc_house_data.csv."
    )

# =================================================================
# 2. LAM SACH DU LIEU
# =================================================================
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"]).copy()
df = df[df["price"] > 0].copy()
df = df.drop_duplicates().reset_index(drop=True)
print("\nKich thuoc sau khi lam sach:", df.shape)
y = df["price"].copy()
print("\nThong ke gia nha:")
print(y.describe().round(0))

# =================================================================
# 3. FEATURE ENGINEERING - 3 TRUONG HOP
# =================================================================
def build_raw(data):
    """Muc Raw: chi dung cac cot so co san, bo id va date, khong tao dac trung moi."""
    X = data.drop(columns=["price", "id", "date"], errors="ignore")
    return X.select_dtypes(include=[np.number]).copy()


def build_fe(data):
    """Muc FE: Raw + cac dac trung tao tu hieu biet ve thi truong nha dat."""
    X = build_raw(data)
    if "date" in data.columns:
        date = pd.to_datetime(data["date"].astype(str), errors="coerce")
        X["sale_year"] = date.dt.year
        X["sale_month"] = date.dt.month
    if "yr_built" in X.columns:
        if "sale_year" in X.columns:
            X["house_age"] = (X["sale_year"] - X["yr_built"]).clip(lower=0)
        else:
            X["house_age"] = (2015 - X["yr_built"]).clip(lower=0)
    if "yr_renovated" in X.columns:
        renovation_year = pd.to_numeric(X["yr_renovated"], errors="coerce").fillna(0)
        X["renovated"] = (renovation_year > 0).astype(int)
        if "sale_year" in X.columns:
            X["years_since_renovation"] = np.where(
                renovation_year > 0,
                (X["sale_year"] - renovation_year).clip(lower=0),
                0
            )
    if "sqft_living" in X.columns and "sqft_lot" in X.columns:
        X["living_lot_ratio"] = X["sqft_living"] / X["sqft_lot"].replace(0, np.nan)
    if "bathrooms" in X.columns and "bedrooms" in X.columns:
        X["bath_bed_ratio"] = X["bathrooms"] / X["bedrooms"].replace(0, np.nan)
    if "sqft_basement" in X.columns:
        X["has_basement"] = (X["sqft_basement"] > 0).astype(int)
    return X


def drop_useless(X):
    """Bo cot rong hoan toan va cot hang so."""
    X = X.dropna(axis=1, how="all")
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    return X.drop(columns=constant_cols)


# ten muc FE -> (ma tran dac trung, co dung log(price) hay khong)
fe_levels = {
    "Raw": (drop_useless(build_raw(df)), False),
    "FE": (drop_useless(build_fe(df)), False),
    "FE + log(price)": (drop_useless(build_fe(df)), True),
}
for name, (X_fe, _) in fe_levels.items():
    print(f"So dac trung muc {name}: {X_fe.shape[1]}")

# =================================================================
# 4. CHIA TRAIN/TEST (DUNG CHUNG CHO MOI TRUONG HOP)
# =================================================================
idx_train, idx_test = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    random_state=RANDOM_STATE
)
y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
print("\nSo mau train:", len(idx_train))
print("So mau test:", len(idx_test))

# =================================================================
# 5. FEATURE SELECTION - 4 TRUONG HOP (NAM TRONG PIPELINE)
# =================================================================
class PearsonTopK(BaseEstimator, TransformerMixin):
    """Giu K dac trung co |tuong quan Pearson| voi target lon nhat (hoc tren train)."""
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
        corr = np.abs((Xc * yc[:, None]).sum(axis=0) / np.where(denom == 0, np.inf, denom))
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[np.argsort(-corr)[:min(self.k, X.shape[1])]] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


def mi_score(X, y):
    return mutual_info_regression(X, y, random_state=RANDOM_STATE)


def make_selector(selection):
    if selection == "All":
        return "passthrough"
    if selection == "Pearson Top-10":
        return PearsonTopK(k=TOP_K)
    if selection == "Mutual Info Top-10":
        return SelectKBest(mi_score, k=TOP_K)
    if selection == "Lasso":
        # Embedded: chuan hoa noi bo, LassoCV tu chon alpha, giu dac trung co he so khac 0
        return SelectFromModel(
            Pipeline([
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(cv=5, max_iter=20000, random_state=RANDOM_STATE))
            ]),
            importance_getter="named_steps.lasso.coef_",
            threshold=1e-5
        )
    raise ValueError(selection)


SELECTIONS = ["All", "Pearson Top-10", "Mutual Info Top-10", "Lasso"]

# =================================================================
# 6. CAC MO HINH VA PIPELINE
# =================================================================
def create_models():
    """ten -> (mo hinh, co can chuan hoa hay khong)"""
    models = {
        "Linear Regression": (LinearRegression(), False),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Decision Tree": (DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE), False),
        "SVR": (SVR(kernel="rbf", C=100, epsilon=0.1), True),
        "Gradient Boosting": (GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=4,
            random_state=RANDOM_STATE), False),
        "Random Forest": (RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE), False),
    }
    if not RUN_SVR:
        models.pop("SVR")
    return models


def build_pipeline(model, needs_scaling, selection, log_target):
    """Imputer -> Feature Selection -> (Scaler) -> Mo hinh.
    Neu dung log(price), boc ngoai bang TransformedTargetRegressor."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("select", make_selector(selection)),
    ]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(model)))
    pipe = Pipeline(steps)
    if log_target:
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def evaluate(pipe, Xtr, Xte, ytr, yte):
    start = time.time()
    pipe.fit(Xtr, ytr)
    fit_time = time.time() - start
    pred = pipe.predict(Xte)
    return {
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "R2": r2_score(yte, pred),
        "MAPE (%)": 100 * mean_absolute_percentage_error(yte, pred),
        "R2 train": r2_score(ytr, pipe.predict(Xtr)),
        "Time (s)": fit_time,
    }, pred


def selected_features(pipe, columns):
    reg = pipe.regressor_ if isinstance(pipe, TransformedTargetRegressor) else pipe
    selector = reg.named_steps["select"]
    if selector == "passthrough":
        return list(columns)
    return list(np.asarray(columns)[selector.get_support()])


def plot_compare(data, group_col, metric, title, file_name):
    table = data.pivot(index="Model", columns=group_col, values=metric)
    ax = table.plot(kind="bar", figsize=(12, 6), edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Mo hinh")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, file_name), dpi=150)
    plt.show()


models = create_models()
predictions = {}   # (FE, Selection, Model) -> (du doan, pipeline, ten cot)

# =================================================================
# 7. GIAI DOAN 1 - FEATURE ENGINEERING (Selection = All)
# =================================================================
print("\n" + "=" * 85)
print("GIAI DOAN 1 - FEATURE ENGINEERING (Selection = All)")
print("=" * 85)
rows1 = []
for fe_name, (X_fe, log_target) in fe_levels.items():
    X_train, X_test = X_fe.iloc[idx_train], X_fe.iloc[idx_test]
    for model_name, (model, needs_scaling) in models.items():
        print("Dang huan luyen:", fe_name, "|", model_name)
        pipe = build_pipeline(model, needs_scaling, "All", log_target)
        result, pred = evaluate(pipe, X_train, X_test, y_train, y_test)
        predictions[(fe_name, "All", model_name)] = (pred, pipe, list(X_fe.columns))
        rows1.append({"FE": fe_name, "Model": model_name,
                      "Features": X_fe.shape[1], **result})
        print(f"   R2 = {result['R2']:.4f} | MAE = {result['MAE']:,.0f} | "
              f"RMSE = {result['RMSE']:,.0f} | {result['Time (s)']:.1f}s")

stage1 = pd.DataFrame(rows1)
stage1.round(4).to_csv(os.path.join(OUT_DIR, "giai_doan_1_feature_engineering.csv"),
                       index=False, encoding="utf-8-sig")
print("\nKET QUA GIAI DOAN 1:")
print(stage1.round(4).to_string(index=False))
plot_compare(stage1, "FE", "R2", "Giai doan 1 - R2 theo muc Feature Engineering", "giai_doan_1_R2.png")
plot_compare(stage1, "FE", "MAE", "Giai doan 1 - MAE theo muc Feature Engineering", "giai_doan_1_MAE.png")

best_fe = stage1.groupby("FE")["R2"].mean().idxmax()
print("\nR2 trung binh theo muc FE:")
print(stage1.groupby("FE")["R2"].mean().round(4))
print("=> Muc FE tot nhat (dung cho giai doan 2):", best_fe)

# =================================================================
# 8. GIAI DOAN 2 - FEATURE SELECTION (FE = muc tot nhat)
# =================================================================
print("\n" + "=" * 85)
print(f"GIAI DOAN 2 - FEATURE SELECTION (FE = {best_fe})")
print("=" * 85)
X_fe, log_target = fe_levels[best_fe]
X_train, X_test = X_fe.iloc[idx_train], X_fe.iloc[idx_test]
rows2 = []
chosen_features = {}
for selection in SELECTIONS:
    for model_name, (model, needs_scaling) in models.items():
        if selection == "All":
            pred, pipe, _ = predictions[(best_fe, "All", model_name)]
            old = stage1[(stage1.FE == best_fe) & (stage1.Model == model_name)].iloc[0]
            result = {k: old[k] for k in ["MAE", "RMSE", "R2", "MAPE (%)", "R2 train", "Time (s)"]}
        else:
            print("Dang huan luyen:", selection, "|", model_name)
            pipe = build_pipeline(model, needs_scaling, selection, log_target)
            result, pred = evaluate(pipe, X_train, X_test, y_train, y_test)
            predictions[(best_fe, selection, model_name)] = (pred, pipe, list(X_fe.columns))
            print(f"   R2 = {result['R2']:.4f} | MAE = {result['MAE']:,.0f}")
        features = selected_features(pipe, X_fe.columns)
        chosen_features.setdefault(selection, features)
        rows2.append({"Selection": selection, "Model": model_name,
                      "Features": len(features), **result})

stage2 = pd.DataFrame(rows2)
stage2.round(4).to_csv(os.path.join(OUT_DIR, "giai_doan_2_feature_selection.csv"),
                       index=False, encoding="utf-8-sig")
pd.DataFrame({k: pd.Series(v) for k, v in chosen_features.items()}).to_csv(
    os.path.join(OUT_DIR, "dac_trung_duoc_chon.csv"), index=False, encoding="utf-8-sig")
print("\nKET QUA GIAI DOAN 2:")
print(stage2.round(4).to_string(index=False))
print("\nDac trung duoc chon:")
for selection, features in chosen_features.items():
    print(f"- {selection} ({len(features)}): {features}")
plot_compare(stage2, "Selection", "R2",
             f"Giai doan 2 - R2 theo cach chon dac trung (FE = {best_fe})", "giai_doan_2_R2.png")
plot_compare(stage2, "Selection", "MAE",
             f"Giai doan 2 - MAE theo cach chon dac trung (FE = {best_fe})", "giai_doan_2_MAE.png")

# =================================================================
# 9. TONG HOP, MO HINH TOT NHAT, FEATURE IMPORTANCE
# =================================================================
results_df = pd.concat([
    stage1.assign(Selection="All"),
    stage2[stage2.Selection != "All"].assign(FE=best_fe)
], ignore_index=True).sort_values("R2", ascending=False).reset_index(drop=True)
cols = ["FE", "Selection", "Model", "Features", "MAE", "RMSE", "R2",
        "MAPE (%)", "R2 train", "Time (s)"]
results_df = results_df[cols]
results_df.round(4).to_csv(os.path.join(OUT_DIR, "tong_hop_tat_ca_cau_hinh.csv"),
                           index=False, encoding="utf-8-sig")
print("\n" + "=" * 85)
print("BANG SO SANH - 10 CAU HINH TOT NHAT THEO R2")
print("=" * 85)
print(results_df.head(10).round(4).to_string(index=False))

best_row = results_df.iloc[0]
best_pred, best_pipe, _ = predictions[(best_row.FE, best_row.Selection, best_row.Model)]
print("\n" + "=" * 65)
print("CAU HINH TOT NHAT THEO R2")
print("=" * 65)
print("Feature Engineering:", best_row.FE)
print("Feature Selection:  ", best_row.Selection)
print("Mo hinh:            ", best_row.Model)
print(f"MAE:  {best_row['MAE']:,.2f}")
print(f"RMSE: {best_row['RMSE']:,.2f}")
print(f"R2:   {best_row['R2']:.4f}")
print(f"MAPE: {best_row['MAPE (%)']:.2f}%")

plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_pred, alpha=0.5)
min_value = min(y_test.min(), best_pred.min())
max_value = max(y_test.max(), best_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="red")
plt.xlabel("Gia nha thuc te")
plt.ylabel("Gia nha du doan")
plt.title(f"Gia thuc te va du doan - {best_row.Model} ({best_row.FE}, {best_row.Selection})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "thuc_te_vs_du_doan.png"), dpi=150)
plt.show()

tree_name = "Random Forest" if "Random Forest" in models else "Gradient Boosting"
_, tree_pipe, tree_cols = predictions[(best_fe, "All", tree_name)]
tree_reg = tree_pipe.regressor_ if isinstance(tree_pipe, TransformedTargetRegressor) else tree_pipe
importance_df = pd.DataFrame({
    "Feature": tree_cols,
    "Importance": tree_reg.named_steps["model"].feature_importances_
}).sort_values(by="Importance", ascending=False)
importance_df.round(4).to_csv(os.path.join(OUT_DIR, "feature_importance.csv"),
                              index=False, encoding="utf-8-sig")
print(f"\nTop 10 dac trung quan trong ({tree_name}, FE = {best_fe}):")
print(importance_df.head(10).to_string(index=False))
top_importance = importance_df.head(15).sort_values(by="Importance", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(top_importance["Feature"], top_importance["Importance"])
plt.xlabel("Feature importance")
plt.ylabel("Dac trung")
plt.title(f"Top 15 dac trung quan trong - {tree_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
plt.show()

# =================================================================
# 10. GIAI DOAN 3 - 5-FOLD CROSS VALIDATION CHO 3 CAU HINH TOT NHAT
# =================================================================
if RUN_CV:
    print("\n" + "=" * 85)
    print("GIAI DOAN 3 - 5-FOLD CROSS VALIDATION CHO 3 CAU HINH TOT NHAT")
    print("=" * 85)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows3 = []
    for _, row in results_df.head(3).iterrows():
        model, needs_scaling = models[row.Model]
        X_fe_cv, log_cv = fe_levels[row.FE]
        pipe = build_pipeline(model, needs_scaling, row.Selection, log_cv)
        scores = cross_validate(
            pipe, X_fe_cv.iloc[idx_train], y_train, cv=cv,
            scoring={"R2": "r2", "MAE": "neg_mean_absolute_error",
                     "RMSE": "neg_root_mean_squared_error"})
        rows3.append({
            "Cau hinh": f"{row.FE} / {row.Selection} / {row.Model}",
            "R2 CV mean": scores["test_R2"].mean(),
            "R2 CV std": scores["test_R2"].std(),
            "MAE CV": -scores["test_MAE"].mean(),
            "RMSE CV": -scores["test_RMSE"].mean(),
            "R2 test": row.R2,
        })
        print(f"{rows3[-1]['Cau hinh']}: R2 CV = {rows3[-1]['R2 CV mean']:.4f} "
              f"+/- {rows3[-1]['R2 CV std']:.4f} | R2 test = {row.R2:.4f}")
    pd.DataFrame(rows3).round(4).to_csv(
        os.path.join(OUT_DIR, "giai_doan_3_cross_validation.csv"),
        index=False, encoding="utf-8-sig")

# =================================================================
# 11. NHAN XET KET QUA
# =================================================================
print("\n" + "=" * 65)
print("NHAN XET KET QUA")
print("=" * 65)
fe_mean = stage1.groupby("FE")["R2"].mean()
print("1. Feature Engineering - R2 trung binh tren cac mo hinh:")
for name, value in fe_mean.items():
    print(f"   - {name}: {value:.4f}")
print(f"   => Muc FE tot nhat: {best_fe}.")
sel_mean = stage2.groupby("Selection")["R2"].mean()
print("2. Feature Selection - R2 trung binh tren cac mo hinh:")
for name, value in sel_mean.items():
    print(f"   - {name} ({len(chosen_features[name])} dac trung): {value:.4f}")
print(f"   => Cach chon tot nhat: {sel_mean.idxmax()}.")
model_best = results_df.groupby("Model")["R2"].max().sort_values(ascending=False)
print("3. Mo hinh - R2 cao nhat cua tung mo hinh:")
for name, value in model_best.items():
    print(f"   - {name}: {value:.4f}")
print(f"4. Cau hinh tot nhat: {best_row.FE} | {best_row.Selection} | {best_row.Model}, "
      f"R2 = {best_row['R2']:.4f}, MAE = {best_row['MAE']:,.0f}, MAPE = {best_row['MAPE (%)']:.2f}%.")
if best_row["R2"] >= 0.8:
    print("5. Mo hinh giai thich duoc phan lon su bien thien gia nha tren tap test.")
elif best_row["R2"] >= 0.5:
    print("5. Mo hinh co kha nang du bao kha, nhung van con sai so.")
else:
    print("5. Kha nang du bao con han che; can xem xet them dac trung va tham so.")
print("\nDa luu toan bo ket qua trong thu muc:", OUT_DIR)
print("\nHOAN THANH BAI 4_2!")
