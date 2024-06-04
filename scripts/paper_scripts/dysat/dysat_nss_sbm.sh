python dgt/run.py \
 --multirun \
 wandb_conf.name=DySat_NSS_SBM \
 dataset=DGB-SBM \
 model=DySat \
 gpu=0 \
 lr=0.01 \
 task.engine.batch_size=12288 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.structural_head_config=[16,8,8] \
 model.link_pred.structural_layer_config=[128] \
 model.link_pred.temporal_head_config=[16] \
 model.link_pred.temporal_layer_config=[128] \
 model.link_pred.spatial_drop=0.1 \
 model.link_pred.temporal_drop=0.5 \
 model.link_pred.neg_weight=1.0 \
 model.link_pred.undirected=True \
 model.link_pred.residual=False \
 optim.optimizer.weight_decay=5e-7 \
 task.engine.epoch=100 \
 task.sampling=random,historical,inductive \
 task.engine.n_runs=1 \
 