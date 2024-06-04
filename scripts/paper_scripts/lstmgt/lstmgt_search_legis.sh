python dgt/run.py \
 --multirun \
 wandb_conf.name=LSTMGT_Search_Legis \
 dataset=DGB-USLegis \
 model=LSTMGT \
 gpu=1 \
 lr=0.1,0.01,0.001,0.0001 \
 task.engine.batch_size=1024 \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=5 \
 model.link_pred.spatial_pe='rwpe' \
 model.link_pred.dim_emb=128 \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.nhead=4 \
 model.link_pred.num_layers_lstm=1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \