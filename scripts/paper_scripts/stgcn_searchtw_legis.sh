python tw_benchmark/run.py \
 --multirun \
 dataset=DGB-USLegis \
 wandb_conf.name=STGCN_SearchTW_Legis \
 gpu=2 \
 lr=0.001 \
 model=STGCN \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=False \
 model.link_pred.window=1,2,3,4,5,6,7,8,9,10,-1 \
 model.link_pred.kernel_size=1 \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.epoch=50 \
 task.engine.batch_size=1024 \
 task.engine.n_runs=1 \
 