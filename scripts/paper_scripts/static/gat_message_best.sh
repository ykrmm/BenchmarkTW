python dgt/run.py \
 gpu=0 \
 dataset=DGB-UCI-Message \
 lr=0.01 \
 model=Static \
 model.name=GAT \
 wandb_conf.name=GAT_Best_rnd_Message \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \

python dgt/run.py \
 gpu=0 \
 dataset=DGB-UCI-Message \
 lr=0.1 \
 model=Static \
 model.name=GAT \
 wandb_conf.name=GAT_Best_hist_Message \
 task.sampling=historical \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \