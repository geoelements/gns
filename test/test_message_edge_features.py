from gns.graph_network import *
import torch
from torch_geometric.data import Data
import pytest


@pytest.fixture
def interaction_network_data():
    model = InteractionNetwork(
        nnode_in=2,
        nnode_out=2,
        nedge_in=2,
        nedge_out=2,
        nmlp_layers=2,
        mlp_hidden_dim=2,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)  # node features
    edge_attr = torch.tensor([[1, 1], [2, 2]], dtype=torch.float)  # edge features

    return model, x, edge_index, edge_attr


def test_edge_update(interaction_network_data):
    """Test if edge features are updated and finite and are not simply doubled"""
    model, x, edge_index, edge_attr = interaction_network_data
    old_edge_attr = edge_attr.clone()  # Save the old edge features

    # One message passing step
    _, updated_edge_attr = model(x=x, edge_index=edge_index, edge_features=edge_attr)

    # Check if edge features shape is correct
    assert (
        edge_attr.shape == old_edge_attr.shape
    ), f"Edge features shape is not preserved, changed from {old_edge_attr.shape} to {edge_attr.shape}"
    # Check if edge features are updated
    assert not torch.equal(
        updated_edge_attr, old_edge_attr * 2
    ), "Edge features are simply doubled"
    assert not torch.equal(
        updated_edge_attr, old_edge_attr
    ), "Edge features are not updated"
    # Check if edge features are finite
    assert torch.all(torch.isfinite(edge_attr)), "Edge features are not finite"


def test_gradients_computed(interaction_network_data):
    """Test if gradients are computed and finite"""
    model, x, edge_index, edge_attr = interaction_network_data
    x.requires_grad = True
    edge_attr.requires_grad = True

    # First pass
    aggr, updated_edge_features = model(
        x=x, edge_index=edge_index, edge_features=edge_attr
    )
    updated_node_features = x + aggr
    # Second pass
    aggr, updated_edge_features = model(
        x=updated_node_features,
        edge_index=edge_index,
        edge_features=updated_edge_features,
    )
    updated_node_features = updated_node_features + aggr
    # Compute loss
    loss = (updated_edge_features).sum()
    loss.backward()

    # Check if gradients are computed
    assert x.grad is not None, "Gradients for node features are not computed"
    assert edge_attr.grad is not None, "Gradients for edge features are not computed"
    # Check if gradients are finite
    assert torch.all(
        torch.isfinite(x.grad)
    ), "Gradients for node features are not finite"
    assert torch.all(
        torch.isfinite(edge_attr.grad)
    ), "Gradients for edge features are not finite"
