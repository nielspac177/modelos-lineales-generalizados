import numpy as np
import pandas as pd


def brier(y, p):
    """Error cuadrático medio de las probabilidades (Brier, 1950). 0 es perfecto."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return np.mean((p - y) ** 2)


def tabla_calibracion(y, p, n_grupos=10):
    """Agrupa por probabilidad predicha y compara con la frecuencia observada."""
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.clip(np.asarray(p, dtype=float), 0, 1 - 1e-12)})
    d["grupo"] = pd.cut(d["p"], np.linspace(0, 1, n_grupos + 1), include_lowest=True, right=False)
    return (d.groupby("grupo", observed=True)
             .agg(n=("y", "size"), p_media=("p", "mean"), frecuencia=("y", "mean"))
             .reset_index())
