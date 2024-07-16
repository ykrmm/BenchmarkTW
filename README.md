# Temporal receptive field in dynamic graph learning: A comprehensive analysis
![tw](https://github.com/ykrmm/BenchmarkTW/blob/main/tw.png)

## Installation 
```
conda create -n twdgnn python=3.9
conda activate twdgnn
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torchmetrics
pip install torch-geometric-temporal
pip install wandb
pip install hydra-core
pip install hydra-colorlog
pip install -e .
```

## Launch paper experiments 
Exemple launch experiments of EGCN on UNtrade: 
```
# in config/wandb_conf/wandb_default.yaml put your wandb info

wandb login
wandb online
mv scripts/paper_scripts/egcnh/egcn_searchtw_trade.sh scripts/egcn_searchtw_trade.sh
```


## Citation
