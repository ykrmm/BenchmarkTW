python tw_benchmark/run.py \
 --multirun \
 dataset=DGB-Bitcoin-OTC \
 wandb_conf.name=STGCN_SearchParam_OTC \
 gpu=0 \
 lr=0.1,0.01,0.001,0.0001 \
 model=STGCN \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=False \
 model.link_pred.window=3 \
 model.link_pred.kernel_size=1 \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.epoch=50 \
 task.engine.batch_size=1024 \
 task.engine.n_runs=1 \
 