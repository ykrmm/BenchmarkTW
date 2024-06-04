python dgt/run.py \
 --multirun \
 wandb_conf.name=GCN_Last_Legis \
 dataset=DGB-USLegis \
 gpu=2 \
 lr=0.1 \
 model=Static \
 model.name=GCN \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
 model.evolving=False \
 model.pred_next=False,True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=1 \