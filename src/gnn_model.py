import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_dim1, hidden_dim2, num_classes, dropout_rate=0.5):
        super(GNNModel, self).__init__()

        # First GCN layer
        self.conv1 = GCNConv(num_features, hidden_dim1)

        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        # Third GCN layer (you can continue adding more layers if needed)
        self.conv3 = GCNConv(hidden_dim2, num_classes)

        # Dropout rate
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)

        # Second layer
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)

        # Third layer
        x3 = self.conv3(x2, edge_index)

        return F.log_softmax(x3, dim=1)


def prepare_gnn_data(adjacency_matrix):
    G = nx.from_pandas_adjacency(adjacency_matrix)

    # Create a mapping from node names to integer indices
    node_mapping = {node: i for i, node in enumerate(G.nodes)}

    # Convert edges from string names to integer indices using the mapping
    edge_indices = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in G.edges]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    x = torch.eye(adjacency_matrix.shape[0])  # Node features as identity matrix
    data = Data(x=x, edge_index=edge_index)
    return data


def train_gnn(data, expected_returns, epochs=100, learning_rate=0.05):
    num_features = data.x.size(1)
    model = GNNModel(num_features, hidden_dim1=128, hidden_dim2=128, num_classes=1)
    target = torch.tensor(expected_returns, dtype=torch.float32).view(-1, 1)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    training_losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target)
        loss.backward()
        optimizer.step()

        # Append the loss for this epoch
        training_losses.append(loss.item())

        # Print the training loss
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}")

    # Get the embeddings from the last layer after training
    with torch.no_grad():
        embeddings = model(data).numpy()

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return embeddings
