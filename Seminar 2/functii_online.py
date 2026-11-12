import pandas as pd
from pandas.api.types import is_numeric_dtype


def nan_replace_df(tabel: pd.DataFrame) -> pd.DataFrame:
    t = tabel.copy()
    for col in t.columns:
        if t[col].isna().any():
            if is_numeric_dtype(t[col]):
                t[col] = t[col].fillna(t[col].mean())
            else:
                modul = t[col].mode()
                if not modul.empty:
                    t[col] = t[col].fillna(modul[0])
    return t


def standardizare_df(tabel: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    return (tabel - tabel.mean(axis=0)) / tabel.std(axis=0, ddof=ddof)