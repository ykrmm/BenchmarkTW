import torch 
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric_temporal import TwitterTennisDatasetLoader
from torch_geometric_temporal.signal.dynamic_graph_temporal_signal import DynamicGraphTemporalSignal

from tw_benchmark.datasets import Discrete_graph



class ListNodeRegDataset(Dataset):
    def __init__(self, 
                 datasets: DynamicGraphTemporalSignal,
                 graphs: list[Discrete_graph],
                 t_max: int,         
                 pred_next: bool = False,
                 ) -> None:
        super().__init__()
        self.graphs = graphs 
        self.datasets = datasets
        self.pred_next = pred_next
        self.t_max = t_max
        
    def get_dataset_t(self,t):
        """
            Return the dataset at time t
        """
        assert t < self.t_max, "t must be smaller than t_max"
        if self.pred_next: 
            graphs = self.graphs[:t]
        else: 
            graphs = self.graphs[:t+1]
        
        dts = NodeRegDataset(self.datasets[t],
                             graphs,
                             t,)
        
        return dts
    
    
class NodeRegDataset(Dataset):
    """
    Class giving the Dataset of a temporal graph for training set and test set
    Edge index is the complete list of edges in the set
    x is the list of features (one-hot or not)
    Neg_edge is the list of negative edges at time t with the appropriate sampling strategies
    """
    def __init__(self,
                 dataset: Data,
                 graphs: list[Discrete_graph],
                 time: torch.LongTensor,
                 ) -> None:
        
        self.dataset = dataset
        self.graphs = graphs
        self.list_nodes = torch.arange(dataset.x.shape[0])
        self.time = time
        
    def __len__(self):
        return len(self.list_nodes)

    def __getitem__(self, idx: int):
        """
            Return the graph at time t no matter if evolving or not
        """
        src_node = self.list_nodes[idx]
        y = self.dataset.y[src_node]
        
        feed_dict = {
            'src_nodes':src_node,
            'y': y,
            'time': self.time,
        }
        return feed_dict
    
    def get_graphs(self):
        return self.graphs


class RegData:
    """
    Node Classification dataset 
    Args:
        dataname (str): Name of the dataset
        n_nodes (int): Number of nodes in the dataset
        split_ratio (list): Ratio of the train, validation and test set
        evolving (bool): If True, the dataset is evolving, else it is static
        one_hot (bool): If True, the node features are the adjacency matrix of the graph, else it is the node features at each time step
        transform (Callable): A function to transform the dynamic graph
    """
    
    def __init__(self,
                 dataname : str,
                 eventname: str,
                 n_nodes: int = None,
                 split_ratio: list = [0.7,0.15,0.15],
                 pred_next: bool = False,):

        super(RegData,self).__init__()
        
        self.dataname = dataname
        self.eventname = eventname
        self.split_ratio = split_ratio
        self.n_nodes = n_nodes
        self.pred_next = pred_next
        
        assert self.dataname in ["TwitterTennis"]
        if self.dataname == "TwitterTennis":
            assert self.eventname in ["rg17","uo17"]
            dataloader = TwitterTennisDatasetLoader(event_id=self.eventname)
            self.dataset = dataloader.get_dataset()
            self.total_snapshot = self.dataset.snapshot_count
        else:
            raise ValueError("Dataset not implemented")

        
    def construct_graph(self):
        """
        Construct the graph from the dataset
        """
        all_graphs = []

        for i in range(self.snapshot_train):
            edge_index = self.dataset[i].edge_index
            weights = self.dataset[i].edge_attr 
            time = torch.tensor(i, dtype=torch.long)
            x = self.dataset[i].x
            score_mat = None
            all_graphs.append(Discrete_graph(edge_index,weights,time,x,score_mat))
        self.train_graphs = all_graphs.copy()
        assert len(self.train_graphs) == self.T_train, "Error in the number of training graphs"
        
        for i in range(self.snapshot_train, self.snapshot_train + self.snapshot_val):
            edge_index = self.dataset[i].edge_index
            weights = self.dataset[i].edge_attr 
            time = torch.tensor(i, dtype=torch.long)
            x = self.dataset[i].x
            score_mat = None
            all_graphs.append(Discrete_graph(edge_index,weights,time,x,score_mat))      
        self.val_graphs = all_graphs.copy()
        assert len(self.val_graphs) == self.T_val, "Error in the number of val graphs"
        
        for i in range(self.snapshot_train + self.snapshot_val, self.total_snapshot):
            edge_index = self.dataset[i].edge_index
            weights = self.dataset[i].edge_attr 
            time = torch.tensor(i, dtype=torch.long)
            x = self.dataset[i].x
            score_mat = None
            all_graphs.append(Discrete_graph(edge_index,weights,time,x,score_mat))
        self.test_graphs = all_graphs.copy()
        assert len(self.test_graphs) == self.T_test, "Error in the number of test graphs"
        
        


    def get_datasets(self):
        """
        Return the list of 'mode' datasets (Discrete_graph_link_pred) for each timestamp
        """
        
        self.snapshot_train = int(self.total_snapshot*self.split_ratio[0])
        self.snapshot_val = int(self.total_snapshot*self.split_ratio[1])
        self.snapshot_test = self.total_snapshot - self.snapshot_train - self.snapshot_val

        self.T_train = self.snapshot_train
        self.T_val = self.snapshot_val + self.snapshot_train
        self.T_test = self.total_snapshot
        
        self.construct_graph()
        
        l_train_datasets = ListNodeRegDataset(
                                                self.dataset,
                                                self.train_graphs,
                                                t_max=self.snapshot_train,
                                                pred_next=self.pred_next,
                                             )
        
        l_val_datasets = ListNodeRegDataset(
                                                self.dataset,
                                                self.val_graphs,
                                                t_max=self.snapshot_val + self.snapshot_train,
                                                pred_next=True,
                                            )
        
        l_test_datasets = ListNodeRegDataset(
                                                self.dataset,
                                                self.test_graphs,
                                                t_max=self.total_snapshot,
                                                pred_next=True
                                            )
        
        
        return l_train_datasets, l_val_datasets, l_test_datasets
