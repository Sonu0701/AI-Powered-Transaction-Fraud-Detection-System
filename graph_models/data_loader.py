import torch
from torch_geometric.data import Data
from collections import defaultdict
import numpy as np

class TransactionGraphBuilder:
    def __init__(self):
        self.node_index = defaultdict(int)
        self.current_id = 0
        self.edges = []
        self.node_features = []
        self.node_types = []

        # FIX: GNN model expects 32 features per node.
        # We encode: 3 one-hot type bits + up to 29 hash-derived identity features.
        self.NUM_FEATURES = 32

    def _make_node_features(self, node_key, node_type):
        """
        Build a 32-dimensional feature vector for a node.
        Dims 0-2  : one-hot node type  (account=0, merchant=1, device=2)
        Dims 3-31 : deterministic hash embedding of the node identity
        """
        vec = np.zeros(self.NUM_FEATURES, dtype=np.float32)
        # One-hot type
        if node_type < 3:
            vec[node_type] = 1.0
        # Hash embedding for remaining 29 dims
        h = hash(str(node_key)) & 0xFFFFFFFF   # unsigned 32-bit
        for i in range(3, self.NUM_FEATURES):
            vec[i] = float((h >> (i % 32)) & 1)
        return vec.tolist()

    def get_node_id(self, node_key, node_type):
        if node_key not in self.node_index:
            self.node_index[node_key] = self.current_id
            self.current_id += 1
            self.node_features.append(self._make_node_features(node_key, node_type))
            self.node_types.append(node_type)
        return self.node_index[node_key]

    def add_transaction(self, transaction):
        # Account node (type 0)
        acc_id      = self.get_node_id(transaction['AccountID'],  0)
        # Merchant node (type 1)
        merchant_id = self.get_node_id(transaction['MerchantID'], 1)
        # Device node (type 2)
        device_id   = self.get_node_id(transaction['DeviceID'],   2)

        # Add edges (bidirectional for better message passing)
        self.edges.append((acc_id, merchant_id))
        self.edges.append((merchant_id, acc_id))
        self.edges.append((acc_id, device_id))
        self.edges.append((device_id, acc_id))

        # FIX: Guard against empty edges list (first call before any edges exist)
        if len(self.edges) < 2:
            # Fallback: self-loop on account node so graph is always valid
            edge_index = torch.tensor([[acc_id], [acc_id]], dtype=torch.long)
        else:
            src, dst = zip(*self.edges)
            edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)

        x = torch.tensor(self.node_features, dtype=torch.float)

        # FIX: Cap graph size to last 500 edges to prevent memory leak
        # on long-running servers
        MAX_EDGES = 500
        if edge_index.shape[1] > MAX_EDGES:
            edge_index = edge_index[:, -MAX_EDGES:]

        return Data(x=x, edge_index=edge_index)