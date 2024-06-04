python dgt/run.py \
 dataset=DGB-UNtrade \
 wandb_conf.name=DyGrEncoder_Best_Rnd_Trade \
 gpu=1 \
 lr=0.01 \
 model=DyGrEncoder \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.conv_num_layers=2 \
 model.link_pred.conv_aggr=max \
 model.link_pred.lstm_num_layers=1 \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0001 \

python dgt/run.py \
 dataset=DGB-UNtrade \
 wandb_conf.name=DyGrEncoder_Best_Hist_Trade \
 gpu=1 \
 lr=0.01 \
 model=DyGrEncoder \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.conv_num_layers=2 \
 model.link_pred.conv_aggr=max \
 model.link_pred.lstm_num_layers=1 \
 model.link_pred.undirected=True \
 task.engine.n_runs=5 \
 task.engine.batch_size=1024 \
 task.sampling=historical \
 optim.optimizer.weight_decay=0.0001 \

