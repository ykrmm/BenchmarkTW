python dgt/run.py \
 --multirun \
 wandb_conf.name=DySat_Search_HepPh \
 dataset=DGB-HepPh \
 model=DySat \
 gpu=0 \
 lr=0.1,0.01,0.001,0.0001,0.0005 \
 task.engine.batch_size=1024 \
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
 optim.optimizer.weight_decay=0.0001 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \
 task.sampling=random \
 