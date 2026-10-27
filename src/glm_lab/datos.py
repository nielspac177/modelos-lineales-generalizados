import pandas as pd


def cargar_tequila(ruta):
    """Pedidos semanales del tequila por tienda; mes como categoría."""
    d = pd.read_csv(ruta, parse_dates=["semana"])
    d["mes"] = d["mes"].astype("category")
    return d


def cargar_vodka(ruta):
    """Ventas semanales de vodka por proveedor (la tabla del lab de regresión lineal)."""
    return pd.read_csv(ruta, parse_dates=["semana"])


def separar_por_fecha(d, corte):
    """Entrenamiento antes del corte, prueba desde el corte: nunca se entrena con el futuro."""
    prueba = d["semana"] >= pd.Timestamp(corte)
    return d[~prueba].copy(), d[prueba].copy()
