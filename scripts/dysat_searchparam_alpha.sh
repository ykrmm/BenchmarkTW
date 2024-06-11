python tw_benchmark/run.py \
 wandb_conf.name=DySat_SearchParam_Alpha \
 dataset=DGB-Bitcoin-Alpha \
 model=DySat \
 gpu=0 \
 lr=0.00001 \
 task.engine.batch_size=512 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=1 \
 model.link_pred.structural_head_config=[8,4,4] \
 model.link_pred.structural_layer_config=[128] \
 model.link_pred.temporal_head_config=[8] \
 model.link_pred.temporal_layer_config=[128] \
 model.link_pred.spatial_drop=0.1 \
 model.link_pred.temporal_drop=0.5 \
 model.link_pred.neg_weight=1.0 \
 model.link_pred.undirected=True \
 model.link_pred.residual=False \
 optim.optimizer.weight_decay=0.0001 \
 task.engine.n_runs=1 \
 task.engine.epoch=50 \
 