python dgt/run.py \
 wandb_conf.name=GIN_Hist_SBM \
 dataset=DGB-SBM \
 lr=0.001 \
 gpu=0 \
 model=Static \
 model.name=GIN \
 model.evolving=True \
 model.clip_grad=True \
 model.one_hot=True \
 model.link_pred.layers=1 \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=150 \
 task.engine.batch_size=1024 \
 task.sampling=historical \

 python dgt/run.py \
 wandb_conf.name=GIN_Ind_SBM \
 dataset=DGB-SBM \
 lr=0.001 \
 gpu=0 \
 model=Static \
 model.name=GIN \
 model.evolving=True \
 model.clip_grad=True \
 model.one_hot=True \
 model.link_pred.layers=1 \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=150 \
 task.engine.batch_size=1024 \
 task.sampling=inductive \