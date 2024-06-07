python tw_benchmark/run.py \
 dataset=DGB-CanParl \
 wandb_conf.name=DCRNN_SearchParam_CanParl \
 gpu=1 \
 lr=0.01 \
 model=DCRNN \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.K=2 \
 model.link_pred.window=3 \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.epoch=50 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=5e-7 \
