python dgt/run.py \
 --multirun \
 wandb_conf.name=GCN_NSS_AS733\
 dataset=DGB-AS733 \
 gpu=0 \
 lr=0.001 \
 model=Static \
 model.name=GCN \
 model.evolving=True \
 model.clip_grad=True \
 model.one_hot=True \
 model.link_pred.layers=1 \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=3 \
 task.engine.epoch=200 \
 task.engine.batch_size=1024 \
 task.sampling=random,historical,inductive \