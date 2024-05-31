import pyedflib
import mne

import numpy as np
import pandas as pd
import networkx as nx
from os.path import basename


CHANNELS: tuple[str] = ('Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 
                        'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 
                        'Fz', 'Cz', 'Pz')
N_CHANNELS: int = len(CHANNELS)

N: int = 36
N_GOODS: int = 26
N_BADS: int = 10

# Posición aproximada de los electrodos en la cabeza
pos = {
    'Fp1': (-0.300, 0.954), 'Fp2': (0.300, 0.954),
    'F3': (-0.400, 0.510), 'F4': (0.400, 0.510),
    'F7': (-0.800, 0.600), 'F8': (0.800, 0.600),
    'T3': (-1.000, 0.000), 'T4': (1.000, 0.000),
    'C3': (-0.500, 0.000), 'C4': (0.500, 0.000),
    'T5': (-0.800, -0.600), 'T6': (0.800, -0.600),
    'P3': (-0.400, -0.510), 'P4': (0.400, -0.510),
    'O1': (-0.300, -0.954), 'O2': (0.300, -0.954),
    'Fz': (0.000, 0.500), 'Cz': (0.000, 0.000),
    'Pz': (0.000, -0.500)
}

# Convertir un archivo EDF a CSV
def edf_to_csv(filename: str, output_path: str = '') -> None:
    f: pyedflib.EdfReader = pyedflib.EdfReader(filename)
    
    signals: np.zeros = np.zeros((N_CHANNELS + 1, m:=f.getNSamples()[0]))
    signals[0, :] = np.arange(1, m + 1) # Es la cabecera de las columnas que son numeros desde el 1 al No. Señales
    for i in np.arange(1, N_CHANNELS + 1):
        signals[i, :]= f.readSignal(i - 1)

    np.savetxt(output_path + basename(filename.replace('.edf', '.csv')), signals, delimiter=';')

    f.close()


def resize_all_subject_csv() -> None:
	# Cada indice representa 0.002 segundos del audio
	# Para el 4to sujeto: Antes->85000, Durante->31000
	# Para el 10mo sujeto: Antes->94000, Durante->31000
	# Para el 31vo sujeto: Antes->40000, Durante->31000
	# Para los demás sujetos: Antes->91000, Durante->31000

	for i in range(36):
		subject = f'Subject{i:02}'
		edf_path1 = f'Archivos\\{subject}\\edf\\{subject}_1.edf'
		edf_path2 = f'Archivos\\{subject}\\edf\\{subject}_2.edf'

		csv_path1 = f'Archivos\\{subject}\\csv\\{subject}_1.csv'
		csv_path2 = f'Archivos\\{subject}\\csv\\{subject}_2.csv'

		temp_raw_before = mne.io.read_raw_edf(edf_path1)
		temp_raw_during = mne.io.read_raw_edf(edf_path2)

		temp_raw_before.drop_channels(['EEG A2-A1', 'ECG ECG'])
		temp_raw_during.drop_channels(['EEG A2-A1', 'ECG ECG'])

		if i == 4:
			final_index = 84081
		elif i == 10:
			final_index = 93272
		elif i == 31:
			final_index = 39313
		else:
			final_index = 90000
		
		csv_data1 = np.concatenate([[np.arange(final_index)], temp_raw_before.get_data()[:, :final_index]])
		csv_data2 = np.concatenate([[np.arange(30000)], temp_raw_during.get_data()[:, :30000]])

		np.savetxt(csv_path1, csv_data1, delimiter=';')
		np.savetxt(csv_path2, csv_data2, delimiter=';')


# Funciones a partir de la matriz de distancias
def get_graph_from_df(df_distances: pd.DataFrame) -> nx.Graph:
    g0: nx.Graph = nx.from_numpy_array(df_distances.to_numpy())
    mapping: dict = dict(zip(g0, CHANNELS))
    nx.relabel_nodes(g0, mapping, False)

    return g0


def get_mst_from_df(df_distances: pd.DataFrame) -> nx.Graph:
    return nx.minimum_spanning_tree(get_graph_from_df(df_distances))


def get_centrality_from_df(df_distances: pd.DataFrame) -> pd.DataFrame:
    mst_g0 = get_mst_from_df(df_distances)

    degree_centrality = nx.degree_centrality(mst_g0)
    betweenness_centrality = nx.betweenness_centrality(mst_g0)
    closeness_centrality = nx.closeness_centrality(mst_g0)
    pagerank = nx.pagerank(mst_g0, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)

    return pd.DataFrame({
        'nodes': CHANNELS,
        'degree': degree_centrality.values(),
        'betweenness': betweenness_centrality.values(),
        'closeness': closeness_centrality.values(),
        'pagerank': pagerank.values()
    })


# Funciones a partir de la ruta
def get_distances(path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path, sep=';')

    df = np.transpose(df)
    df.columns = CHANNELS

    # Hacer la matriz de correlación
    df_correlation: pd.DataFrame = df.corr()

    # Obtener las listas de las distancias a partir de las correlaciones
    # Dij = sqrt(2 * (1 - Cij))
    df_distances: pd.DataFrame = pd.DataFrame(np.sqrt(2 * (1 - df_correlation)), index=CHANNELS, columns=CHANNELS)

    return df_distances


def get_graph(path: str) -> nx.Graph:
    df_distances: pd.DataFrame = get_distances(path)
    return get_graph_from_df(df_distances)

    return g0


def get_mst(path: str) -> nx.Graph:
    return nx.minimum_spanning_tree(get_graph(path))


def centrality(path: str) -> pd.DataFrame:
    mst_g0 = get_mst(path)

    degree_centrality = nx.degree_centrality(mst_g0)
    betweenness_centrality = nx.betweenness_centrality(mst_g0)
    closeness_centrality = nx.closeness_centrality(mst_g0)
    pagerank = nx.pagerank(mst_g0, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)

    return pd.DataFrame({
        'nodes': CHANNELS,
        'degree': degree_centrality.values(),
        'betweenness': betweenness_centrality.values(),
        'closeness': closeness_centrality.values(),
        'pagerank': pagerank.values()
    })
