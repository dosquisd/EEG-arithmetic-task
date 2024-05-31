import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

from my_functions.mst_distances import pos


# Define un mapa de colores personalizado para el degradado (rojo a verde)
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FF0000', '#00FF00'], N=100)


def plot_mst_distances(mst_ave_before: nx.Graph, mst_ave_during: nx.Graph, title: str) -> None:
    # Generar la base de los grafos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 7))
    fig.suptitle(title)

    for ax in (ax1, ax2):
        ax.add_patch(Circle((0, 0), radius=1, color='lightgray', fill=False, linestyle='--', linewidth=1.5))
        ax.add_patch(Circle((0, 0), radius=1.3, color='black', fill=False, linewidth=2))

        # Agregar segmentos
        ax.plot([0, 0], [-1.3, 1.3], linestyle='--', color='lightgray', linewidth=1.5, zorder=0)
        ax.plot([-1.3, 1.3], [0, 0], linestyle='--', color='lightgray', linewidth=1.5, zorder=0)

    # Agregar a la izquierda el mst para antes
    ax1.set_title('Before')
    nx.draw_networkx(
        mst_ave_before,
        pos,
        ax = ax1,
        font_size=15,
        node_color='lightgreen'
    )

    nx.draw_networkx_edge_labels(
        mst_ave_before,
        pos,
        {pair: round(weight, 2) for pair, weight in nx.get_edge_attributes(mst_ave_before, 'weight').items()},
        ax=ax1
    )

    # Agregar a la derecha el mst para durante
    ax2.set_title('During')
    nx.draw_networkx(
        mst_ave_during,
        pos,
        ax=ax2,
        font_size=15,
        node_color='lightgreen'
    )

    nx.draw_networkx_edge_labels(
        mst_ave_during,
        pos,
        {pair: round(weight, 2) for pair, weight in nx.get_edge_attributes(mst_ave_during, 'weight').items()},
        ax=ax2
    )

    plt.tight_layout()
    plt.show()


def get_node_color(i: int, measure: str, df_centrality: pd.DataFrame) -> tuple:
    measure_centrality = df_centrality.get(measure)
    return cmap(measure_centrality[i])


def plot_measure(G: nx.Graph, measure: str, df_centrality: pd.DataFrame, ax) -> None:
    temp_labels = {node: f'{node}\n{temp_centrality[measure]}' for node, temp_centrality in G.nodes(data=True)}

    # Agregar la base del grafo
    ax.add_patch(Circle((0, 0), radius=1.3, color='black', fill=False, linewidth=1))
    ax.add_patch(Circle((0, 0), radius=1, color='lightgray', fill=False, linestyle='--', linewidth=1))

    # Agregar segmentos
    ax.plot([0, 0], [-1.3, 1.3], linestyle='--', color='lightgray', linewidth=1, zorder=0)
    ax.plot([-1.3, 1.3], [0, 0], linestyle='--', color='lightgray', linewidth=1, zorder=0)

    nx.draw_networkx(
        G,
        pos,
        node_size=50,
        with_labels=True,
        ax=ax,
        node_color=[get_node_color(i, measure, df_centrality) for i, _ in enumerate(G.nodes)],
        labels=temp_labels
    )

    ax.set_title(f'{measure.title()} centrality')


def plot_all_measures(mst: nx.Graph, df_centrality: pd.DataFrame, title: str) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Degree
    plot_measure(mst, 'degree', df_centrality, ax1)

    # Betweenness
    plot_measure(mst, 'betweenness', df_centrality, ax2)

    # Closeness
    plot_measure(mst, 'closeness', df_centrality, ax3)

    # Pagerank
    plot_measure(mst, 'pagerank', df_centrality, ax4)

    fig.suptitle(title)
    plt.tight_layout()

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, label='Centrality', ax=(ax1, ax2, ax3, ax4))

    plt.show()
    