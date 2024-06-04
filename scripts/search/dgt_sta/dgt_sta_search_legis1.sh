python dgt/run.py \
 --multirun \
 wandb_conf.name=DGT_STA_RWPE_NoTE_search_Legis \
 dataset=DGB-USLegis \
 model=DGT_STA \
 gpu=1 \
 lr=0.01,0.001,0.1,0.0001 \
 task.engine.batch_size=1024 \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.bias_lin_pe=False \
 model.link_pred.spatial_pe=rwpe \
 model.link_pred.add_temporal_pe=False \
 model.link_pred.dim_pe=6,12 \
 model.link_pred.dim_feedforward=1024,512 \
 model.link_pred.nhead=2 \
 model.link_pred.temp_pe_drop=0.1 \
 model.link_pred.norm_first=True,False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 