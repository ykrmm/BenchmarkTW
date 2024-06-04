python dgt/run.py\
  --multirun \
  gpu=0 \
  model=DySat \
  wandb_conf.name=DySat_Search_rnd_CanParl \
  dataset=DGB-CanParl \
  lr=0.001,0.005 \
  task.engine.n_runs=1 \
  task.engine.batch_size=1024 \
  model.evolving=False \
  model.pred_next=False,True \
  model.one_hot=True \
  model.clip_grad=False,True \
  model.link_pred.structural_head_config=[16,8,8],[8,4,4],[4,2,2] \
  model.link_pred.structural_layer_config=[128] \
  model.link_pred.temporal_head_config=[4],[8],[16] \
  model.link_pred.temporal_layer_config=[128] \
  model.link_pred.spatial_drop=0,0.1 \
  model.link_pred.temporal_drop=0.5 \
  model.link_pred.neg_weight=1.0 \
  model.link_pred.residual=False \
  optim.optimizer.weight_decay=0,0.0001 \