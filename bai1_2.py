import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
X = diabetes.data.copy()
y = diabetes.target.copy()
df = X.copy()
df["target"] = y
print("Kích thước dữ liệu:", df.shape)
print("\n5 dòng dữ liệu đầu tiên:")
print(df.head())
features = ["bmi", "bp", "s5"]
print("\nTHỐNG KÊ MÔ TẢ:")
print(df[features].describe().T)
print("\nSỐ GIÁ TRỊ THIẾU:")
print(df[features + ["target"]].isnull().sum())
print("\nSố dòng trùng lặp:", df.duplicated().sum())
for col in features:
    print(f"Dang ve bieu do: {col}")
    plt.figure(figsize=(7, 4))
    plt.hist(df[col], bins=25, edgecolor="black", alpha=0.8)
    plt.title(f"Phan phoi cua {col}")
    plt.xlabel(col)
    plt.ylabel("Tan suat")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()
outlier_rows = []
for col in features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    outlier_rows.append({
        "Feature": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Ngưỡng dưới": lower,
        "Ngưỡng trên": upper,
        "Số outlier": int(mask.sum())
    })
print("\nKẾT QUẢ PHÁT HIỆN NGOẠI LỆ:")
print(pd.DataFrame(outlier_rows).round(4).to_string(index=False))
correlations = df[features + ["target"]].corr()["target"].drop("target")
print("\nHỆ SỐ TƯƠNG QUAN VỚI TARGET:")
print(correlations.to_frame("Correlation").round(4))
plt.figure(figsize=(7, 4))
plt.bar(correlations.index, correlations.values, edgecolor="black")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Tương quan giữa các feature và target")
plt.xlabel("Feature")
plt.ylabel("Hệ số tương quan Pearson")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
for col in features:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[col], df["target"], alpha=0.6)
    plt.title(f"Scatter plot: {col} và target")
    plt.xlabel(col)
    plt.ylabel("target")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
plt.figure(figsize=(8, 5))
df[features].boxplot()
plt.title("Boxplot của 3 feature")
plt.ylabel("Giá trị")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
print("\n" + "=" * 60)
print("PHAN NHAN XET KET QUA BAI 1")
print("=" * 60)
print("\n1. NHAN XET TONG QUAN:")
print(f"- Du lieu co {df.shape[0]} dong va {df.shape[1] - 1} dac trung.")
print(f"- So gia tri thieu: {df[features + ['target']].isnull().sum().sum()}")
print(f"- So dong trung lap: {df.duplicated().sum()}")
print("\n2. NHAN XET PHAN PHOI:")
for col in features:
    skewness = df[col].skew()
    print(f"- {col}:")
    if skewness > 0.5:
        print("  Phan phoi lech phai.")
    elif skewness < -0.5:
        print("  Phan phoi lech trai.")
    else:
        print("  Phan phoi tuong doi can doi.")
print("\n3. NHAN XET NGOAI LE:")
for col in features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper)).sum())
    print(f"- {col}: co {count} gia tri co kha nang la ngoai le.")
print("\n4. NHAN XET TUONG QUAN VOI TARGET:")
for col, value in correlations.items():
    if value > 0:
        direction = "tuong quan cung chieu"
    elif value < 0:
        direction = "tuong quan nguoc chieu"
    else:
        direction = "khong co tuong quan tuyen tinh"
    if abs(value) >= 0.5:
        strength = "manh"
    elif abs(value) >= 0.3:
        strength = "trung binh"
    else:
        strength = "yeu"
    print(f"- {col}: r = {value:.4f}, {direction}, muc do {strength}.")
strongest_feature = correlations.abs().idxmax()
print(
    f"- Dac trung co tuong quan manh nhat voi target la "
    f"{strongest_feature} (r = {correlations[strongest_feature]:.4f})."
)
print("\n5. NHAN XET SCATTER PLOT:")
for col in features:
    value = correlations[col]
    if value > 0:
        print(f"- {col}: target co xu huong tang khi {col} tang.")
    elif value < 0:
        print(f"- {col}: target co xu huong giam khi {col} tang.")
    else:
        print(f"- {col}: khong co moi quan he tuyen tinh ro rang voi target.")
print("\n6. KET LUAN:")
print("- Histogram giup quan sat phan phoi cua cac dac trung.")
print("- IQR va boxplot giup phat hien cac gia tri co kha nang la ngoai le.")
print("- He so Pearson va scatter plot giup danh gia moi quan he giua dac trung va target.")
print("- Cac dac trung co tuong quan manh hon co the duoc xem xet khi xay dung mo hinh hoi quy.")
print("- Tuong quan khong dong nghia voi quan he nhan qua.")
print("\nHOAN THANH BAI 1!")
