import numpy as np
from neal import SimulatedAnnealingSampler


class QUBOSolver:
    def __init__(self, adjacency_matrix, expected_returns, penalty_parameter, gnn_embeddings=None):
        self.H = self.construct_hamiltonian(adjacency_matrix, expected_returns, penalty_parameter)
        if gnn_embeddings is not None:
            self.H = self.update_hamiltonian_with_gnn(self.H, gnn_embeddings)

    def update_hamiltonian_with_gnn(self, H, gnn_embeddings, alpha=1.0):
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
