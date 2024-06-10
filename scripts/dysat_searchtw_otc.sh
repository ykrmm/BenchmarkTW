
python tw_benchmark/run.py \
 --multirun \
 wandb_conf.name=DySat_SearchParams_OTC \
 dataset=DGB-Bitcoin-OTC \
 model=DySat \
 gpu=0 \
 lr=0.01 \
 task.engine.batch_size=512 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=1,2,3,4,5,6,7,8,9,10,-1 \
 model.link_pred.structural_head_config=[8,4,4] \
 model.link_pred.structural_layer_config=[64] \
 model.link_pred.temporal_head_config=[8] \
 model.link_pred.temporal_layer_config=[64] \
 model.link_pred.spatial_drop=0.1 \
 model.link_pred.temporal_drop=0.5 \
 model.link_pred.neg_weight=1.0 \
 model.link_pred.undirected=True \
 model.link_pred.residual=False \
 optim.optimizer.weight_decay=0.0001 \
 task.engine.n_runs=1 \
 task.engine.epoch=50 \
 