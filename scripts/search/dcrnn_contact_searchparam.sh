python tw_benchmark/run.py \
 --multirun \
 dataset=DGB-Contacts \
 wandb_conf.name=DCRNN_SearchParam_Contacts \
 gpu=0 \
 lr=0.0001,0.001,0.01,0.1 \
 model=DCRNN \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=False \
 model.link_pred.K=2 \
 model.link_pred.window=3 \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.epoch=50 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=5e-7 \
