import pandas as pd
from graph import Graph
from qubo_solver import QUBOSolver
from gnn_model import GNNModel, prepare_gnn_data, train_gnn
from prettytable import PrettyTable


def main():
    # Load preprocessed data
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
    mean_expected_returns = expected_returns.mean()
    std_expected_returns = expected_returns.std()
    normalized_expected_returns = (expected_returns - mean_expected_returns) / std_expected_returns
    learned_embeddings = train_gnn(data_for_gnn, normalized_expected_returns)

    print("Constructing Hamiltonian and solving QUBO...")
    qubo_solver = QUBOSolver(graph.adjacency_matrix, expected_returns, penalty_parameter=5.0,
                             gnn_embeddings=learned_embeddings)
    solution = qubo_solver.solve()
    print("Solution:", solution)

    # From solution dictionary, visualize the selected assets
    selected_assets = [key for key, value in solution.items() if value == 1]
    print("Selected assets:", selected_assets)
    print("Number of selected assets:", len(selected_assets))

    # Print out name of selected assets
    selected_assets_names = price_data.columns[selected_assets].tolist()
    print("Selected assets names:", selected_assets_names)

    """
    table = PrettyTable()
    table.field_names = ["Ticker", "Name", "Selected"]
    for i, column in enumerate(price_data.columns):
        table.add_row([i, column, "Yes" if i in selected_assets else "No"])
    print(table)
    """


if __name__ == "__main__":
    main()
