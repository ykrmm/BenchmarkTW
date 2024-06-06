python tw_benchmark/run.py \
 --multirun \
 wandb_conf.name=EGCNH_SearchTW_Enron \
 dataset=DGB-Enron \
 model=EvolveGCNH \
 gpu=2 \
 lr=0.001 \
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.num_layers_rnn=2 \
 model.link_pred.time_length=1,2,3,4,5,6,7,8,9,10,-1 \
 model.link_pred.improved=True \
 model.link_pred.cached=False \
 model.link_pred.add_self_loops=False \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \