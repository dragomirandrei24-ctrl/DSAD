import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

fisier_local = "Wholesale customers data.csv"
url_uci = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"

if os.path.exists(fisier_local):
    df = pd.read_csv(fisier_local)
else:
    df = pd.read_csv(url_uci)

spending_cols = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[spending_cols].copy()

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
df_std = pd.DataFrame(X_std, columns=spending_cols, index=df.index)
df_std.to_csv("wholesale_standardizat.csv", index=False)

R = df_std.corr()
R.to_csv("R_wholesale.csv", index=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    R,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)
plt.title("Correlation Matrix - Customer Spending Categories")
plt.tight_layout()
plt.savefig("corelogram_heatmap.png")
plt.show()

pca = PCA()
C = pca.fit_transform(X_std)

explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

pca_summary = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(spending_cols))],
    "Eigenvalue (Alpha)": explained_variance,
    "Explained_Variance_Ratio": explained_variance_ratio,
    "Explained_Variance_Percent": explained_variance_ratio * 100,
    "Cumulative_Percent": cumulative_variance_ratio * 100
})

print(pca_summary.to_string(index=False))
pca_summary.to_csv("pca_varianta_explicata.csv", index=False)

num_components_80 = int(np.argmax(cumulative_variance_ratio >= 0.80) + 1)
print(f"\nNumarul de componente necesare pentru >= 80% din varianta: {num_components_80}")
print(f"Varianta cumulata explicata de primele {num_components_80} componente: {cumulative_variance_ratio[num_components_80 - 1] * 100:.2f}%")

loadings = pd.DataFrame(
    pca.components_.T,
    index=spending_cols,
    columns=[f"PC{i+1}" for i in range(len(spending_cols))]
)
loadings.to_csv("pca_loadings.csv", index=True)