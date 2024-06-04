python dgt/run.py \
 dataset=DGB-UNtrade \
 wandb_conf.name=EGCNO_Best_Rnd_Trade \
 gpu=1 \
 lr=0.01 \
 model=EvolveGCNO \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.num_layers_rnn=2 \
 model.link_pred.improved=True \
 model.link_pred.cached=True \
 model.link_pred.add_self_loops=False \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0001 \

python dgt/run.py \
 dataset=DGB-UNtrade \
 wandb_conf.name=EGCNO_Best_Hist_Trade \
 gpu=1 \
 lr=0.01 \
 model=EvolveGCNO \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.num_layers_rnn=2 \
 model.link_pred.improved=True \
 model.link_pred.cached=True \
 model.link_pred.add_self_loops=False \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.sampling=historical \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0001 \