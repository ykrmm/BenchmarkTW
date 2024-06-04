python dgt/run.py \
 dataset=DGB-UCI-Message \
 wandb_conf.name=GCLSTM_Best_Rnd_Message \
 gpu=1 \
 lr=0.001 \
 model=GCLSTM \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \

python dgt/run.py \
 dataset=DGB-UCI-Message \
 wandb_conf.name=GCLSTM_Best_Hist_Message \
 gpu=1 \
 lr=0.001 \
 model=GCLSTM \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 task.sampling=historical \
 optim.optimizer.weight_decay=0 \
