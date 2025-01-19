import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def importances(forest):
    # Usamos argsort para devolver los indices ordenados, y no los valores
    # Argsort ordena de menor a mayor, por lo que tambien invertiremos el orden del array resultante
    return np.argsort(forest.feature_importances_)[::-1]


def redundances(forest):
    # Inicializamos un array para contar las divisiones de nodos por caracteristica
    redundancia = np.zeros(forest.n_features_in_, dtype=int)

    for tree in forest.estimators_:
        # Obtenemos los indices de las caracteristicas usadas para dividir nodos en este arbol
        feature_indices = tree.tree_.feature
        # Obtenemos las veces que aparece cada caracteristica
        unique, counts = np.unique(feature_indices[feature_indices >= 0], return_counts=True)
        redundancia[unique] += counts

    return np.argsort(redundancia)


def relationships(forest):
    relaciones_dict = {i: set() for i in range(forest.n_features_in_)}

    # Iteramos los arboles que forman el bosque
    for i in forest.estimators_:
        # Accedemos a la estructura del arbol y obtenemos las caracteristicas usadas en cada nodo
        tree = i.tree_
        feature_indices = tree.feature
        n_nodes = tree.node_count

        for node_id in range(n_nodes):
            feature_id = feature_indices[node_id]
            if feature_id != 2:  # -2 indica que el nodo es hoja, y por tanto no tiene caracteristica asociada
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]

                if left_child != -1:  # -1 indica que no tiene hijo izquierdo
                    left_feature_id = feature_indices[left_child]
                    if left_feature_id != -2:
                        # Agregamos al diccionario las dos conexiones, de A a B y de B a A
                        relaciones_dict[feature_id].add(left_feature_id)
                        relaciones_dict[left_feature_id].add(feature_id)

                if right_child != -1:  # -1 indica que no tiene hijo derecho
                    right_feature_id = feature_indices[right_child]
                    if right_feature_id != -2:
                        # Agregamos al diccionario las dos conexiones, de A a B y de B a A
                        relaciones_dict[feature_id].add(right_feature_id)
                        relaciones_dict[right_feature_id].add(feature_id)

    return relaciones_dict


def fitness(X, y, S, alpha):
    if not S:  # Si S esta vacio, devolvemos el peor fitness posible
        return 10

    N = np.size(X, 1)  # Calculamos el total de caracteristicas del conjunto de datos original

    S = list(S)
    X_selected = X[:, S]  # Eliminamos las caracteristicas de X que no se encuentran en S

    # Separamos los datos en entrenamiento y prueba para poder evaluar el modelo
    Xtr, Xte, ytr, yte = train_test_split(X_selected, y, test_size=0.3)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    # Creamos, entrenamos y evaluamos el modelo KNN para calcular su tasa de error
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)
    Er = 1 - knn.score(Xte, yte)

    return alpha * Er + ((1 - alpha) * (len(S) / N))


def RSFSS(X, y, alpha):
    forest = RandomForestClassifier()
    forest.fit(X, y)

    S = set()
    V = 10
    F = importances(forest)
    C = redundances(forest)
    R = relationships(forest)

    # Expansion por importancia
    for f_n in F:
        S_candidate = S | {f_n}  # Intenta agregar la caracteristica f_n
        V_candidate = fitness(X, y, S_candidate, alpha)
        if V_candidate < V:  # Si mejora la valoracion
            V = V_candidate
            S = S_candidate

    # Expansion por relaciones
    for f_n in F:
        if f_n in S:
            related_features = R.get(f_n, [])
            for related_feature in related_features:
                S_candidate = S | {related_feature}
                V_candidate = fitness(X, y, S_candidate, alpha)
                if V_candidate < V:  # Si mejora la valoracion
                    V = V_candidate
                    S = S_candidate

    # Reduccion por redundancias
    for c_n in C:
        if c_n in S:
            S_candidate = S - {c_n}  # Intenta eliminar la característica c_n
            V_candidate = fitness(X, y, S_candidate, alpha)
            if V_candidate < V:  # Si mejora la valoracion
                V = V_candidate
                S = S_candidate

    return sorted(S)