import yfinance as yf
import pandas as pd
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import dimod
from tqdm import tqdm
from neal import SimulatedAnnealingSampler
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from prettytable import PrettyTable

plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=16)
plt.rc('axes', grid=False)
warnings.filterwarnings('ignore')


class DataFetcher:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = self.fetch_data_for_tickers()
        self.returns_data = self.calculate_returns()
        self.correlation_matrix = self.calculate_correlations()

    def fetch_data(self, ticker):
        stock_data = yf.Ticker(ticker)
        return stock_data.history(period='1d', start=self.start_date, end=self.end_date)['Close']

    def fetch_data_for_tickers(self):
        price_data = {}
        for ticker in tqdm(self.tickers, desc="Fetching Data"):
            price_data[ticker] = self.fetch_data(ticker)
        return pd.DataFrame(price_data)

    def calculate_returns(self):
        return self.price_data.pct_change().dropna()

    def calculate_correlations(self):
        return self.returns_data.corr()


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
        shells = [list(G.nodes)]
        pos = nx.shell_layout(G, shells)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Graph of Assets", fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', alpha=0.5, label='Assets')
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.3, edge_color='lightgrey')
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.savefig("graph_of_assets.pdf", bbox_inches='tight')
        plt.show()


class QUBOSolver:
    def __init__(self, adjacency_matrix, expected_returns, penalty_parameter, gnn_embeddings=None):
        self.H = self.construct_hamiltonian(adjacency_matrix, expected_returns, penalty_parameter)
        if gnn_embeddings is not None:
            self.H = self.update_hamiltonian_with_gnn(self.H, gnn_embeddings)

    def update_hamiltonian_with_gnn(self, H, gnn_embeddings, alpha=1.0):
        # Update Hamiltonian using GNN embeddings
        updated_H = H + alpha * gnn_embeddings
        return updated_H

    def construct_hamiltonian(self, adjacency_matrix, expected_returns, penalty_parameter):
        n = len(expected_returns)
        H = np.zeros((n, n))
        adjacency_matrix_np = adjacency_matrix.values

        for i in range(n):
            H[i, i] -= expected_returns[i]

        for i in range(n):
            for j in range(n):
                if adjacency_matrix_np[i, j] == 1:
                    H[i, j] += penalty_parameter

        return H

    def solve(self):
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample_qubo(self.H, num_reads=100)
        solution = response.first.sample
        return solution


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_df = tables[0]
    return sp500_df['Symbol'].tolist()


def select_random_tickers(tickers, count=10):
    return random.sample(tickers, count)


def calculate_expected_returns(price_data):
    return price_data.mean()


# GNN Model
class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


"""def train_gnn(adjacency_matrix, epochs=100, hidden_dim=64, learning_rate=0.01):
    # Convert adjacency matrix to edge index format
    G = nx.from_pandas_adjacency(adjacency_matrix)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    # Create a PyTorch Geometric Data object
    data = Data(edge_index=edge_index)

    # Define the GNN model
    num_features = adjacency_matrix.shape[1]  # Assuming features are same as nodes
    num_classes = 1  # Adjust depending on your task
    model = GNNModel(num_features, hidden_dim, num_classes)

    # Define loss function and optimizer
    loss_function = torch.nn.MSELoss()  # Adjust depending on your task
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target)  # Define target based on your task
        loss.backward()
        optimizer.step()

    # Extract the embeddings or any other information you want to use
    learned_embeddings = model.conv1.weight.detach()

    return learned_embeddings"""


def main():
    print("Fetching S&P 500 tickers...")
    sp500_tickers = get_sp500_tickers()
    print("Selecting random tickers...")
    random_tickers = select_random_tickers(sp500_tickers)

    print("Fetching and processing data...")
    data_fetcher = DataFetcher(random_tickers, "2002-01-01", "2022-12-31")
    expected_returns = calculate_expected_returns(data_fetcher.price_data)

    print("Constructing the graph of assets...")
    graph = Graph(data_fetcher.correlation_matrix, threshold=0.1)
    graph.visualize()

    print("Constructing Hamiltonian and solving QUBO...")
    qubo_solver = QUBOSolver(graph.adjacency_matrix, expected_returns, penalty_parameter=5.0)
    solution = qubo_solver.solve()
    print("Solution:", solution)

    # From solution dictionary, visualize the selected assets
    selected_assets = []
    for key, value in solution.items():
        if value == 1:
            selected_assets.append(key)
    print("Selected assets:", selected_assets)
    print("Number of selected assets:", len(selected_assets))

    # Print out name of selected assets
    selected_assets_names = []
    for ticker in selected_assets:
        selected_assets_names.append(data_fetcher.price_data.columns[ticker])
    print("Selected assets names:", selected_assets_names)

    # Table for all assets and selected assets
    table = PrettyTable()
    table.field_names = ["Ticker", "Name", "Selected"]
    for i in range(len(data_fetcher.price_data.columns)):
        table.add_row([i, data_fetcher.price_data.columns[i], "Yes" if i in selected_assets else "No"])
    print(table)


if __name__ == "__main__":
    main()
