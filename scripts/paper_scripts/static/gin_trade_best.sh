python dgt/run.py \
 gpu=2 \
 dataset=DGB-UNtrade \
 lr=0.001 \
 model=Static \
 model.name=GIN \
 wandb_conf.name=GIN_Best_rnd_Trade \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0005 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \

python dgt/run.py \
 gpu=2 \
 dataset=DGB-UNtrade \
 task.sampling=historical \
 lr=0.001 \
 model=Static \
 model.name=GIN \
 wandb_conf.name=GIN_Best_hist_Trade \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0005 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \