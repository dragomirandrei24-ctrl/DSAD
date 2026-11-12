import numpy as np
import pandas as pd
from functii_online import nan_replace_df, standardizare_df

df = pd.read_csv("winequality-red.csv", sep=";")
print(df.shape)
print(df.head(5))
print(df.dtypes)

df = nan_replace_df(df)
print(df.isna().sum().sum())

vinuri_calitate_min7 = df[df["quality"] >= 7]
vinuri_calitate_min7.to_csv("data_out/vinuri_calitate_min7.csv", index=False)

top50_alcool = df.sort_values(by="alcohol", ascending=False).head(50)
top50_alcool.to_csv("data_out/top50_alcool.csv", index=False)

medii_pe_calitate = df.groupby("quality", as_index=False)[["alcohol", "pH"]].mean()
medii_pe_calitate.to_csv("data_out/medii_pe_calitate.csv", index=False)

df["aciditate_totala"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
coloane_cerinta6 = ["quality", "alcohol", "aciditate_totala", "pH"]
df[coloane_cerinta6].to_csv("data_out/vinuri_cu_aciditate.csv", index=False)

medii_alcool = df.groupby("quality")["alcohol"].mean()
nota_max_alcool = medii_alcool.idxmax()
df_max_alcool = pd.DataFrame({"quality": [nota_max_alcool]})
df_max_alcool.to_csv("data_out/calitate_max_alcool.csv", index=False)

coloane_corr = ["fixed acidity", "volatile acidity", "citric acid", "alcohol", "quality"]
R = df[coloane_corr].corr()
R.to_csv("data_out/R.csv", index=True)

df["alcohol_std"] = (df["alcohol"] - df["alcohol"].mean()) / df["alcohol"].std(ddof=1)
df[["alcohol", "alcohol_std"]].to_csv("data_out/alcohol_standardizat.csv", index=False)

coloane_dist = ["fixed acidity", "volatile acidity", "alcohol"]
date_std = standardizare_df(df[coloane_dist], ddof=1)
primele_20_std = date_std.iloc[:20]

X = primele_20_std.values
n = len(X)
matrice_dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        matrice_dist[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))

df_dist = pd.DataFrame(matrice_dist, index=primele_20_std.index, columns=primele_20_std.index)
df_dist.to_csv("data_out/d_euclid_20.csv", index=True)