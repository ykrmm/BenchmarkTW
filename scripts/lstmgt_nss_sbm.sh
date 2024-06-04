python dgt/run.py \
 --multirun \
 wandb_conf.name=LSTMGT_NSS_SBM \
 dataset=DGB-SBM \
 model=LSTMGT \
 gpu=1 \
 lr=0.001 \
 task.engine.batch_size=12288 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=3 \
 model.link_pred.spatial_pe='rwpe' \
 model.link_pred.dim_emb=128 \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.nhead=8 \
 model.link_pred.num_layers_lstm=1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.epoch=100 \
 task.sampling=random,historical,inductive \
 task.engine.n_runs=1 \