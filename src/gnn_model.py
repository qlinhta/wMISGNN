import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


class ContrastiveGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim1, hidden_dim2, embedding_dim, dropout_rate=0.5):
        super(ContrastiveGNN, self).__init__()
        # First GCN layer
        self.conv1 = GCNConv(num_features, hidden_dim1)
        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        # Third GCN layer to produce embeddings
        self.conv3 = GCNConv(hidden_dim2, embedding_dim)
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
        # Third layer to produce embeddings
        x3 = self.conv3(x2, edge_index)
        return x3  # Return embeddings


def contrastive_loss(embeddings, edge_index, margin=0.5, num_neg_samples=None):
    # Extract embeddings for source and target nodes of edges
    src_embeddings = embeddings[edge_index[0]]
    tgt_embeddings = embeddings[edge_index[1]]
    # Compute the dot product similarity for positive pairs (adjacent nodes)
    pos_similarity = (src_embeddings * tgt_embeddings).sum(dim=1)
    positive_loss = -torch.log(torch.sigmoid(pos_similarity))

    # If num_neg_samples is None, use the same number as positive samples
    num_neg_samples = num_neg_samples or len(edge_index[0])

    # Randomly sample negative pairs
    all_nodes = set(range(embeddings.shape[0]))
    existing_edges = set(tuple(x) for x in edge_index.t().tolist())
    negative_edges = set()

    while len(negative_edges) < num_neg_samples:
        i, j = torch.randint(0, embeddings.shape[0], (2,))
        if (i.item(), j.item()) not in existing_edges and (i.item(), j.item()) not in negative_edges:
            negative_edges.add((i.item(), j.item()))

    neg_src, neg_tgt = zip(*negative_edges)
    neg_src_embeddings = embeddings[list(neg_src)]
    neg_tgt_embeddings = embeddings[list(neg_tgt)]

    # Compute the dot product similarity for negative pairs (non-adjacent nodes)
    neg_similarity = (neg_src_embeddings * neg_tgt_embeddings).sum(dim=1)
    negative_loss = torch.log(torch.sigmoid(neg_similarity))

    # Combine the positive and negative loss terms
    loss = positive_loss.mean() + torch.clamp(margin - negative_loss, min=0).mean()

    return loss


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
