import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import to_undirected

class TemporalConv(nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))


    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H




class STGCN(nn.Module):
    r"""Spatio-temporal convolution block using ChebConv Graph Convolutions.
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting"
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv)
    with kernel size k. Hence for an input sequence of length m,
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size (int): Size of the kernel considered.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        window: int, 
        kernel_size: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
        one_hot: bool = True,
        undirected: bool = True,
    ):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.window = window 
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self.undirected = undirected
        self.bceloss = BCEWithLogitsLoss()

        if one_hot:
            self.lin = nn.Linear(num_nodes, in_channels)
        else:
            self.lin = nn.Linear(num_features, in_channels)
        
        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)
    
    def set_device(self, device):
        self.device = device
            
    def forward(self,graphs):
        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        x = self.lin(graphs[0].x.to(self.device))
        x = x.unsqueeze(0)  # Taille: (1, N_nodes, in_channels)
        x = x.repeat(len(graphs), 1, 1)  # Taille: (T, N_nodes, in_channels)
        x = x.unsqueeze(0)  # Taille: (1, T, N_nodes, in_channels)
        T_0 = self._temporal_conv1(x)     
        T = torch.zeros_like(T_0).to(T_0.device)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                edge_index =  to_undirected(graphs[t].edge_index.to(self.device)) if self.undirected else graphs[t].edge_index.to(self.device)
                T[b][t] = self._graph_conv(T_0[b][t], edge_index)
        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        #T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3) # B, T, N, F
        T = T.squeeze(0) # T, N, F
        # N, T, F
        T = T.permute(1, 0, 2)
        return T
    
    def get_loss_link_pred(self, feed_dict,graphs):

        node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
        # run gnn
        if self.window > 0:
            tw = max(0,len(graphs)-self.window)
        else:
            tw = 0 
        final_emb = self.forward(graphs[tw:]) # [N, T, F]
        emb_source = final_emb[node_1,-1,:]
        emb_pos  = final_emb[node_2,-1,:]
        emb_neg = final_emb[node_2_negative,-1,:]
        pos_score = torch.sum(emb_source*emb_pos, dim=1)
        neg_score = torch.sum(emb_source*emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + neg_loss
            
        return graphloss, pos_score.detach().sigmoid(), neg_score.detach().sigmoid()
    
    def score_eval(self,feed_dict,graphs):
        with torch.no_grad():
            node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
            if self.window > 0:
                tw = max(0,len(graphs)-self.window)
            else:
                tw = 0 
            final_emb = self.forward(graphs[tw:]) # [N, T, F]
            # time-1 because we want to predict the next time step in eval
            emb_source = final_emb[node_1, -1 ,:]
            emb_pos  = final_emb[node_2, -1 ,:]
            emb_neg = final_emb[node_2_negative, -1 ,:]
            pos_score = torch.sum(emb_source*emb_pos, dim=1)
            neg_score = torch.sum(emb_source*emb_neg, dim=1)        
            return pos_score.sigmoid(),neg_score.sigmoid()