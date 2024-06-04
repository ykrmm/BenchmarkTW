python dgt/run.py \
 wandb_conf.name=GConvLSTM_Best_rnd_Can \
 dataset=DGB-CanParl \
 model=GConvLSTM \
 gpu=0 \
 lr=0.01 \
 model.evolving=False \
 task.engine.batch_size=1024 \
 model.link_pred.K=3 \
 model.pred_next=False \
 model.link_pred.normalization=rw \
 model.link_pred.bias=True \
 model.link_pred.undirected=True \
 model.clip_grad=True \
 optim.optimizer.weight_decay=0.0001 \
 task.engine.n_runs=5 \
 task.engine.epoch=30 \

python dgt/run.py \
 wandb_conf.name=GConvLSTM_Best_Hist_Can \
 dataset=DGB-CanParl \
 model=GConvLSTM \
 gpu=0 \
 lr=0.01 \
 model.evolving=False \
 task.engine.batch_size=1024 \
 model.link_pred.K=3 \
 model.pred_next=False \
 model.link_pred.normalization=rw \
 model.link_pred.bias=True \
 model.link_pred.undirected=True \
 model.clip_grad=True \
 optim.optimizer.weight_decay=0.0001 \
 task.engine.n_runs=5 \
 task.engine.epoch=30 \
 task.sampling=historical \