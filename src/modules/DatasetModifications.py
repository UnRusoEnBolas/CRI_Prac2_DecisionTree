import pandas as pd

def discretizeDataframe(dataframe, columns, n_bins):
    """
    Esta función sirve para separar los atribuos continuos que pueda contener el dataset de una manera sencilla
    (no óptima).
    Devuelve un pandas.DataFrame con las columnas que le llegan por el parámetro 'columns' discretizadas en tantos
    quartiles como inique el parámetro n_bins.
    """
    for column in columns:
        dataframe[column] = pd.qcut(dataframe[column], q=n_bins)
    return dataframe

def deleteRowsWithValues(dataframe, value):
    """
    Esta funcion sirve para eliminar las filas del dataset que incluyan en alguna de sus columnas 
    el valor que recibe como parámetro.
    Normalmente se usa para eliminar las fila que contienen valores desconocidos.
    Devuelve un pandas.DataFrame con  las filas que complen la coindición ya eliminadas.
    """
    for column in dataframe.columns:
        dataframe = dataframe[dataframe[column] != value]
        dataframe[column] = dataframe[column].astype("float64")
    return dataframe