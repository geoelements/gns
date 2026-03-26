import torch
import torch.nn as nn
from typing import List
from typing import Tuple
from torch import Tensor
from physicsnemo.models.graphcast.graph_cast_processor import GraphCastProcessor
from physicsnemo.models.gnn_layers.graph import CuGraphCSC
from physicsnemo.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP


class Encoder(nn.Module):
    """
    Replace with new Encoder
    """
    def __init__(
          self,
          nnode_in_features: int,
          nnode_out_features: int,
          nedge_in_features: int,
          nedge_out_features: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          ):
        super(Encoder, self).__init__()
        # Encode node features as an MLP
        self.node_mlp = MeshGraphMLP(
            input_dim=nnode_in_features,
            output_dim=nnode_out_features,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=nmlp_layers,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
        )

        # MLP for edge embedding
        self.edge_mlp = MeshGraphMLP(
            input_dim=nedge_in_features,
            output_dim=nedge_out_features,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=nmlp_layers,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
        )


    def forward(
          self,
          nfeat: Tensor,
          efeat: Tensor,
        ):
        return self.node_mlp(nfeat), self.edge_mlp(efeat)

class Processor(nn.Module):
    """
    replace with GraphCastProcessor
    """
    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        ):
        super().__init__()
        self.processor = GraphCastProcessor(
            aggregation="sum",
            processor_layers=nmessage_passing_steps,
            input_dim_nodes=nnode_in,
            input_dim_edges=nedge_in,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=nmlp_layers,
            activation_fn=nn.ReLU(),
            do_concat_trick=False,
        )

    def forward(
        self,
        nfeat: Tensor,              # [n_nodes, nnode_latent]
        efeat: Tensor,      # [n_edges, nedge_latent]
        graph: CuGraphCSC,
    ) -> tuple[Tensor, Tensor]:
        efeat, nfeat = self.processor(
            efeat,
            nfeat,
            graph,
        )
        return efeat, nfeat
    


class Decoder(nn.Module):
    """
    Replace Decoder
    """

    def __init__(
          self,
          nnode_in: int,
          nnode_out: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          ):
        super(Decoder, self).__init__()
        self.node_mlp = MeshGraphMLP(
            input_dim=nnode_in,
            output_dim=nnode_out,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=nmlp_layers,
            activation_fn=nn.ReLU(),
            norm_type=None,
        )
    def forward(self,
              nfeat: Tensor):
        return self.node_mlp(nfeat)

class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        nnode_in_features: int,
        nnode_out_features: int,
        nedge_in_features: int,
        latent_dim: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
    ):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = Encoder(
            nnode_in_features=nnode_in_features,
            nnode_out_features=latent_dim,
            nedge_in_features=nedge_in_features,
            nedge_out_features=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            nnode_in=latent_dim,
            nnode_out=latent_dim,
            nedge_in=latent_dim,
            nedge_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=nnode_out_features,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self,
              nfeat: Tensor,
              efeat: Tensor,
              graph: CuGraphCSC,
              get_on_all_ranks: bool = False,
        ):
        
        nfeat, efeat = self._encoder(nfeat, efeat)
        efeat, nfeat = self._processor(nfeat, efeat, graph)
        nfeat = self._decoder(nfeat)
        nfeat = self.prepare_output(nfeat, graph, get_on_all_ranks)
        return nfeat
    

    def prepare_output(
        self,
        nfeat: Tensor,
        graph: CuGraphCSC,
        get_on_all_ranks: bool = False,
    ) -> Tensor:
        nfeat = graph.get_global_dst_node_features(
            nfeat,
            get_on_all_ranks=get_on_all_ranks,
        )
        if get_on_all_ranks:
            return nfeat
        if graph.dist_graph.graph_partition.partition_rank != 0:
            nfeat.fill_(torch.nan)
        return nfeat
