python dgt/run.py \
 --multirun \
 wandb_conf.name=GIN_Search_AS733\
 dataset=DGB-AS733 \
 gpu=2 \
 lr=0.1,0.01,0.001,0.0001,0.0005 \
 model=Static \
 model.name=GIN \
 model.evolving=True \
 model.clip_grad=True \
 model.one_hot=True \
 model.link_pred.layers=1 \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \
 task.engine.batch_size=1024 \
 task.sampling=random \