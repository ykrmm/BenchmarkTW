python dgt/run.py \
 gpu=1 \
 dataset=DGB-CanParl \
 lr=0.1 \
 model=Static \
 model.name=GCN \
 task.sampling=historical \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \