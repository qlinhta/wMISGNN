import pandas as pd
import torch
import torch.nn.functional as F
from graph import Graph
from qubo_solver import QUBOSolver
from prettytable import PrettyTable
from gnn_model import ContrastiveGNN, contrastive_loss, prepare_gnn_data


# Function to compute the updated adjacency matrix using learned embeddings
def compute_updated_adjacency(embeddings):
    return embeddings @ embeddings.T


# Function to compute the Hamiltonian using the updated adjacency matrix
def compute_hamiltonian(A_prime, expected_returns, penalty_parameter):
    n = len(expected_returns)
    H = torch.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                H[i, j] = penalty_parameter * A_prime[i, j]
            else:
                H[i, j] = -expected_returns[i]

    return H


def main():
    # Load preprocessed data
    global embeddings
    print("Loading preprocessed data...")
    price_data = pd.read_csv('../data/sp500_price.csv', index_col='Date', parse_dates=True)
    print("Done")

    # Calculate returns and correlations
    print("Calculating returns and correlations...")
    returns_data = price_data.pct_change().dropna()
    correlation_matrix = returns_data.corr()
    print("Done")

    expected_returns = returns_data.mean()

    print("Constructing the graph of assets...")
    graph = Graph(correlation_matrix, threshold=0.1)
    graph.visualize()

    print("Training GNN...")
    data_for_gnn = prepare_gnn_data(graph.adjacency_matrix)

    # Initialize model and optimizer
    model = ContrastiveGNN(num_features=data_for_gnn.num_features, hidden_dim1=64, hidden_dim2=32, embedding_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Number of epochs
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(data_for_gnn)
        loss = contrastive_loss(embeddings, data_for_gnn.edge_index)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    A_prime = compute_updated_adjacency(embeddings)
    print("Constructing Hamiltonian and solving QUBO...")

    # Update the Hamiltonian construction to use A'
    qubo_solver = QUBOSolver(A_prime, expected_returns, penalty_parameter=5.0, gnn_embeddings=embeddings)
    solution = qubo_solver.solve()

    print("Solution:", solution)

    # From solution dictionary, visualize the selected assets
    selected_assets = [key for key, value in solution.items() if value == 1]
    print("Selected assets:", selected_assets)
    print("Number of selected assets:", len(selected_assets))

    # Print out name of selected assets
    selected_assets_names = price_data.columns[selected_assets].tolist()
    print("Selected assets names:", selected_assets_names)


if __name__ == "__main__":
    main()
