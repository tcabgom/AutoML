import pandas as pd
import re

class CategoricalEncoder:

    def __init__(self, method: str ="auto", max_one_hot_unique: int =5, rare_category_threshold: float=0.01, verbose=True):
        if method not in {"auto", "ordinal", "onehot"}:
            raise ValueError("method must be 'auto', 'ordinal' or 'onehot'")
        if not (0.0 <= rare_category_threshold <= 1.0):
            raise ValueError("rare_category_threshold must be between 0 and 1.")
        self.method = method
        self.max_one_hot_unique = max_one_hot_unique
        self.rare_category_threshold = rare_category_threshold
        self.verbose = verbose

    def __decide(self, series: pd.Series) -> str:
        if self.method == "auto":
            if 2 < series.dropna().nunique() <= self.max_one_hot_unique:
                return "onehot"
            else:
                return "ordinal"
        else:
            return self.method

    def fit_column(self, series: pd.Series, column_name: str = "") -> dict:
        res = {}                            # Diccionario para guardar resultados que necesita el preprocesador
        strategy = self.__decide(series)    # Decidir estrategia de codificacion
        res["strategy"] = strategy          # Guardar estrategia

        try:                                                # Usar try-except por si la serie esta vacia
            res["mode"] = str(series.mode(dropna=True)[0])  # Calcular moda (valor mas frecuente)
        except Exception:
            res["mode"] = ""                                # Si no se puede calcular la moda, asignar cadena vacia

        if strategy == "ordinal":                                       # ORDINAL ENCODING
            cats = pd.Categorical(series.dropna()).categories           # Obtener categorias unicas (sin NaNs)
            mapping = {str(cat): idx for idx, cat in enumerate(cats)}   # Crear mapeo categoria -> numero
            res['mapping'] = mapping                                    # Guardar mapeo en resultados
            if self.verbose:
                print(f" - Column '{column_name}': Ordinal encoding created with {len(cats)} unique values. ({series.isna().sum()} null values will be filled with '{str(series.mode(dropna=True)[0])}').")
        else:                           # ONE-HOT ENCODING
            non_na = series.dropna()    # Eliminar NaNs para calcular frecuencias
            if len(non_na) == 0:
                unique_vals = []        # Si no hay valores no nulos no hay categorias
            else:
                freqs = non_na.value_counts(normalize=True)                                 # Calcular frecuencias relativas
                if self.rare_category_threshold > 0:                                        # Si se usa umbral para categorias raras
                    top = freqs[freqs >= self.rare_category_threshold].index.to_list()       # Filtrar categorias frecuentes
                    unique_vals = top                                                       # Guardar categorias frecuentes
                    if len(unique_vals) < len(freqs):                                       # Si existen categorias raras
                        unique_vals.append("OTHER")                                         # Guardar nueva categoria "OTHER"
                else:
                    unique_vals = non_na.unique().tolist()                                  # Si no hay umbral, usar todas las categorias
            res['categories'] = [str(c) for c in unique_vals]                               # Guardar categorias

            res['onehot_columns'] = [
                re.sub(r'[\[\]<>]', '_', f"{column_name}__{str(cat)}")
                for cat in unique_vals
            ]   # Guardar nombres de columnas one-hot

            if self.verbose:
                print(f" - Column '{column_name}': One-hot encoding created with {len(unique_vals)} unique values. ({series.isna().sum()} null values will be filled with '{str(series.mode(dropna=True)[0])}').")

        return res

    def transform_column(self, series: pd.Series, params: dict, column_name: str = ""):

        strategy = params.get('strategy')   # Obtener estrategia de codificacion
        if strategy == "ordinal":           # ORDINAL ENCODING
            series_f = series.fillna(params.get('mode')).astype(str)                # Rellenar NaNs con la moda y convertir a str
            mapped = series_f.map(params.get('mapping')).fillna(-1).astype('int64') # Mapear usando el mapeo guardado, NaNs a -1
            return mapped

        elif strategy == "onehot":          # ONE-HOT ENCODING
            series_f = series.fillna(params.get('mode')).astype(str)                        # Rellenar NaNs con la moda y convertir a str
            categories_set = set(params.get('categories'))                                  # Obtener conjunto de categorias guardadas
            if ('OTHER' in categories_set):                                                 # Si se usa categoria "OTHER"
                series_f = series_f.map(lambda x: x if x in categories_set else 'OTHER')    # Mapear categorias raras a "OTHER"

            # Crear dummies (one-hot), sin columna para NaNs, tipo int8 para ahorrar memoria
            dummies_df = pd.get_dummies(series_f, prefix=column_name, prefix_sep="__", dummy_na=False).astype("int8")

            expected = params.get('onehot_columns')                             # Columnas esperadas (guardadas)
            for c in expected:                                                  # Asegurar que todas las columnas esperadas estan en el DataFrame
                if c not in dummies_df.columns:                                 # Si falta alguna columna
                    dummies_df[c] = 0                                           # Agregar columna con ceros
            if expected:
                dummies_df = dummies_df.reindex(columns=expected, fill_value=0) # Reordenar columnas segun las esperadas
            return dummies_df

        else:
            return series
