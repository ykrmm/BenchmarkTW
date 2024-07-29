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
pip install -e .
```

## Launch paper experiments 
Exemple launch experiments of EGCN on UNtrade: 
```
# in config/wandb_conf/wandb_default.yaml put your wandb info

wandb login
mv scripts/paper_scripts/egcnh/egcn_searchtw_trade.sh scripts/egcn_searchtw_trade.sh
sh scripts/egcn_searchtw_trade.sh
```
## Datasets 
All datasets use in our experiments are in the 'datasets' folder. 

## Citation
Cite as : 
```
@misc{karmim2024temporalreceptivefielddynamic,
      title={Temporal receptive field in dynamic graph learning: A comprehensive analysis}, 
      author={Yannis Karmim and Leshanshui Yang and Raphaël Fournier S'Niehotta and Clément Chatelain and Sébastien Adam and Nicolas Thome},
      year={2024},
      eprint={2407.12370},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.12370}, 
}
```
