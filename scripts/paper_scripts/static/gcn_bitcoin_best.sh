python dgt/run.py \
 gpu=1 \
 dataset=DGB-Bitcoin-OTC \
 lr=0.1 \
 wandb_conf.name=GCN_Best_rnd_Bitcoin \
 task.engine.batch_size=1024 \
 task.engine.n_runs=5 \
 optim.optimizer.weight_decay=0 \
 model=Static \
 model.name=GCN \
 model.evolving=False \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 
python dgt/run.py \
 gpu=1 \
 dataset=DGB-Bitcoin-OTC \
 lr=0.1 \
 wandb_conf.name=GCN_Best_hist_Bitcoin \
 task.sampling=historical \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
 model=Static \
 model.name=GCN \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \