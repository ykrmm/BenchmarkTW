python dgt/run.py \
 dataset=DGB-UCI-Message \
 wandb_conf.name=EGCNH_Best_Rnd_Message \
 gpu=0 \
 lr=0.1 \
 model=EvolveGCNH \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=False \
 model.link_pred.num_layers_rnn=1 \
 model.link_pred.improved=False \
 model.link_pred.cached=False \
 model.link_pred.add_self_loops=True \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0001 \

python dgt/run.py \
 dataset=DGB-UCI-Message \
 wandb_conf.name=EGCNH_Best_Hist_Message \
 gpu=0 \
 lr=0.1 \
 model=EvolveGCNH \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=False \
 model.link_pred.num_layers_rnn=1 \
 model.link_pred.improved=False \
 model.link_pred.cached=False \
 model.link_pred.add_self_loops=True \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 task.sampling=historical \
 optim.optimizer.weight_decay=0.0001 \