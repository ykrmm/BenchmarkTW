python dgt/run.py \
 --multirun \
 gpu=0 \
 dataset=DGB-UCI-Message \
 lr=0.1,0.01,0.001 \
 model=Static \
 model.name=GIN \
 wandb_conf.name=GIN_Search_rnd_Message \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True,False \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=1 \
 model.link_pred.hidden_dim=128 \