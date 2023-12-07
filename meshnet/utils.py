import torch
from torch_geometric.data import Data
import enum


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def datas_to_graph(training_example, dt, device):
    # features
    node_coords = training_example[0][0]  # (nnodes, dims)
    node_type = training_example[0][1]  # (nnodes, 1)
    velocity_feature = training_example[0][2]  # (nnodes, dims)
    cells = training_example[0][3]  # (ncells, nnodes_per_cell)
    cells = torch.transpose(cells, 0, 1).type(torch.LongTensor)
    time_vector = training_example[0][4] * dt  # (nnodes, )
    time_vector = time_vector.unsqueeze(1)
    # n_node_per_example = training_example[0][6]

    # aggregate node features
    node_features = torch.hstack((node_type, velocity_feature, time_vector))

    # target velocity
    velocity_target = training_example[1]  # (nnodes, dims)

    # make graph
    graph = Data(x=node_features, face=cells, y=velocity_target, pos=node_coords, device=device)

    return graph


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key == "x":
            x = graph.x
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        else:
            pass

    return (x, edge_index, edge_attr, global_attr)


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)

    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.global_attr = global_attr

    return ret


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
