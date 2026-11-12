import numpy as np
import pandas as pd

def calcul_medie_ponderata(t:pd.DataFrame):
    x = t.values
    mp = np.average(x[:,:-1],axis=0,weights=x[:,-1])
    return pd.Series(mp,t.columns[:-1])

industria_alim = pd.read_csv("IndustriaAlimentara.csv",index_col=0)
industrii = list(industria_alim)[1:]

angajati = industria_alim[industrii].apply(func=lambda x:x.sum(),axis=1)
# print(angajati)
cerinta1 = industria_alim[angajati>0]
cerinta1.to_csv("cerinta1.csv")

# Cerinta 2
cerinta2 = industria_alim[industrii].apply(func=lambda x: x*100/x.sum(),axis=1)
assert isinstance(cerinta2,pd.DataFrame)
cerinta2.insert(0,"Localitate",industria_alim["Localitate"])
# print(cerinta2)
cerinta2.to_csv("Cerinta2.csv")

# Cerinta 3
populatie = pd.read_csv("PopulatieLocalitati.csv",index_col=0)
industria_alim_loc = industria_alim.merge(populatie,left_index=True,right_index=True)
# print(industria_alim_loc)
industria_alim_judet = industria_alim_loc[industrii+["Populatie","Judet"]].groupby(by="Judet").sum()
# print(industria_alim_judet)
cerinta3 = industria_alim_judet.apply(func=lambda x:x[industrii].sum()/x["Populatie"],axis=1)
cerinta3.name="Pondere"
# print(cerinta3)
cerinta3.to_csv("Cerinta3.csv")

# Cerinta 4
cerinta4 = industria_alim_judet[industrii].apply(lambda x:x.index[x.argmax()],axis=1)
cerinta4.name = "Activitate"
cerinta4.to_csv("Cerinta4.csv")

# Cerinta 5
coduri_judete= pd.read_csv("Coduri_Judete.csv",index_col=0)
# print(industria_alim_loc)
industria_alim_loc_ = industria_alim_loc[industrii+["Judet","Populatie"]].merge(
    coduri_judete["Regiune"],
    left_on="Judet",
    right_index=True
)
# print(industria_alim_loc_)
cerinta5 = industria_alim_loc_[industrii+["Populatie","Regiune"]].groupby(by="Regiune").apply(
    func=calcul_medie_ponderata,
    include_groups=False
)
# print(cerinta5)
cerinta5.round(3).to_csv("Cerinta5.csv")

# Cerinta 6
# print(industria_alim_judet)
total_national = industria_alim_judet.sum(axis=0)
# print(total_national)
cerinta6 = industria_alim_judet.apply(
    lambda x: (x[industrii]/total_national[industrii]) /(x["Populatie"]/total_national["Populatie"]) ,
    axis=1
)
# print(cerinta6)
cerinta6.round(3).to_csv("Cerinta6.csv")
