import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, correlation_matrix, threshold=0.5):
        self.correlation_matrix = correlation_matrix
        self.adjacency_matrix = self.create_adjacency_matrix(threshold)

    def create_adjacency_matrix(self, threshold):
        adjacency_matrix = (self.correlation_matrix.abs() > threshold).astype(int)
        np.fill_diagonal(adjacency_matrix.values, 0)
        return adjacency_matrix

    def visualize(self):
        G = nx.from_pandas_adjacency(self.adjacency_matrix)
        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Graph of Assets", fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        nx.draw_networkx_nodes(G, pos, node_size=10, node_color='black', label='Assets')
        nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.1, edge_color='lightgrey')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.savefig("graph_of_assets.pdf", bbox_inches='tight')
        plt.show()
