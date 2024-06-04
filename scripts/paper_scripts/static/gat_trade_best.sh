python dgt/run.py \
 gpu=0 \
 dataset=DGB-UNtrade \
 lr=0.01 \
 wandb_conf.name=GAT_Best_rnd_Trade \
 model=Static \
 model.name=GAT \
 model.evolving=False \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 model.link_pred.undirected=False \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \


python dgt/run.py \
 gpu=0 \
 dataset=DGB-UNtrade \
 lr=0.01 \
 wandb_conf.name=GAT_Best_hist_Trade \
 model=Static \
 model.name=GAT \
 model.evolving=False \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 model.link_pred.undirected=False \
 task.engine.n_runs=5 \
 task.sampling=historical \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \