python dgt/run.py \
 --multirun \
 dataset=DGB-CanParl \
 wandb_conf.name=EGCNO_Search_Rnd_Can \
 gpu=0 \
 lr=0.001,0.01,0.1 \
 model=EvolveGCNO \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True,False \
 model.link_pred.num_layers_rnn=1,2 \
 model.link_pred.improved=False,True \
 model.link_pred.cached=True,False \
 model.link_pred.add_self_loops=True,False \
 model.link_pred.use_edge_weight=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0,0.0001 \
