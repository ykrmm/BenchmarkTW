python dgt/run.py \
 wandb_conf.name=GCN_Hist_SBM\
 dataset=DGB-SBM \
 gpu=0 \
 lr=0.1 \
 model=Static \
 model.name=GCN \
 model.evolving=True \
 model.clip_grad=True \
 model.one_hot=True \
 model.link_pred.layers=1 \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=150 \
 task.engine.batch_size=16384 \
 task.sampling=historical \

python dgt/run.py \
 wandb_conf.name=GCN_Ind_SBM\
 dataset=DGB-SBM \
 gpu=0 \
 lr=0.1 \
 model=Static \
 model.name=GCN \
 model.evolving=True \
 model.clip_grad=True \
 model.one_hot=True \
 model.link_pred.layers=1 \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=150 \
 task.engine.batch_size=16384 \
 task.sampling=inductive \