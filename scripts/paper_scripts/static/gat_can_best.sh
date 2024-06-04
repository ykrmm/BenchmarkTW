python dgt/run.py \
 gpu=0 \
 dataset=DGB-CanParl \
 task.sampling=historical \
 lr=0.01 \
 model=Static \
 model.name=GAT \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \