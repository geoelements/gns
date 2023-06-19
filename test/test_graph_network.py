import pytest
import torch
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from gns.graph_network import EncodeProcessDecode

@pytest.fixture
def model_data():
    batch_size = 10
    nparticles = 20
    nnode_in_features = 30
    nnode_out_features = 2
    nedge_in_features = 3
    latent_dim = 128
    nmessage_passing_steps = 2
    nmlp_layers = 2
    mlp_hidden_dim = 128
    model = EncodeProcessDecode(
        nnode_in_features,
        nnode_out_features,
        nedge_in_features,
        latent_dim,
        nmessage_passing_steps,
        nmlp_layers,
        mlp_hidden_dim,
    )
    x = torch.rand(batch_size, nparticles, nnode_in_features)
    edge_index = torch.randint(0, nparticles, (2, nparticles))
    edge_features = torch.rand(batch_size, nparticles, nedge_in_features)

    return model, x, edge_index, edge_features, nparticles, nnode_out_features

def test_encode_process_decode(model_data):
    model, x, edge_index, edge_features, nparticles, nnode_out_features = model_data
    output = model(x, edge_index, edge_features)
    assert output[0].shape == (nparticles, nnode_out_features)
    assert output[1].shape == (nparticles, nnode_out_features)