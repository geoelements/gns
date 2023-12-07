import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected
from torch_geometric.transforms.face_to_edge import FaceToEdge


@functional_transform('my_face_to_edge')
class MyFaceToEdge(FaceToEdge):
    def __call__(self, data: Data) -> Data:
        # Your modified method here
        if hasattr(data, 'face'):
            face = data.face
            if len(face) == 3:
                edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
                edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
            elif len(face) == 4:
                edge_index = torch.cat([face[0:2], face[1:3], face[2:], face[::3]], dim=1)
                edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
            else:
                raise NotImplementedError(f"Only support face with dim = 2 or 3, but {len(face)} provided")

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data
