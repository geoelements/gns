import unittest
import torch
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from gns.graph_network import EncodeProcessDecode

class TestModels(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.nparticles = 20
        self.nnode_in_features = 30
        self.nnode_out_features = 2
        self.nedge_in_features = 3
        self.latent_dim = 128
        self.nmessage_passing_steps = 2
        self.nmlp_layers = 2
        self.mlp_hidden_dim = 128
        self.model = EncodeProcessDecode(
            self.nnode_in_features,
            self.nnode_out_features,
            self.nedge_in_features,
            self.latent_dim,
            self.nmessage_passing_steps,
            self.nmlp_layers,
            self.mlp_hidden_dim,
        )
        self.x = torch.rand(self.batch_size, self.nparticles, self.nnode_in_features)
        self.edge_index = torch.randint(0, self.nparticles, (2, self.nparticles))
        self.edge_features = torch.rand(self.batch_size, self.nparticles, self.nedge_in_features)

    def test_encode_process_decode(self):
        output = self.model(self.x, self.edge_index, self.edge_features)
        self.assertEqual(output[0].shape, torch.Size(self.nparticles, self.nnode_out_features))
        self.assertEqual(output[1].shape, (self.batch_size, self.nparticles, self.latent_dim))

if __name__ == "__main__":
    unittest.main()
