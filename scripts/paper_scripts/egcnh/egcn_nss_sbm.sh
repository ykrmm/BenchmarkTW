python dgt/run.py \
 wandb_conf.name=EGCNH_Hist_SBM \
 dataset=DGB-SBM \
 model=EvolveGCNH \
 gpu=1 \
 lr=0.01 \
 task.engine.batch_size=12288\
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.num_layers_rnn=2 \
 model.link_pred.improved=True \
 model.link_pred.cached=False \
 model.link_pred.add_self_loops=False \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=5e-7 \
 task.engine.n_runs=1 \
 task.engine.epoch=150 \
 task.sampling=historical \

python dgt/run.py \
 wandb_conf.name=EGCNH_Ind_SBM \
 dataset=DGB-SBM \
 model=EvolveGCNH \
 gpu=1 \
 lr=0.01 \
 task.engine.batch_size=12288\
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.num_layers_rnn=2 \
 model.link_pred.improved=True \
 model.link_pred.cached=False \
 model.link_pred.add_self_loops=False \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=5e-7 \
 task.engine.n_runs=1 \
 task.engine.epoch=150 \
 task.sampling=inductive \