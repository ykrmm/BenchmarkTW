python tw_benchmark/run.py \
 --multirun \
 dataset=DGB-Bitcoin-Alpha \
 wandb_conf.name=EGCNH_SearchParam_Alpha \
 gpu=1 \
 lr=0.0001,0.001,0.01,0.1 \
 model=EvolveGCNH \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=False \
 model.link_pred.num_layers_rnn=2 \
 model.link_pred.time_length=3 \
 model.link_pred.improved=True \
 model.link_pred.cached=False \
 model.link_pred.add_self_loops=False \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.epoch=30 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
