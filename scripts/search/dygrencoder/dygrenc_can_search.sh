python dgt/run.py \
 --multirun \
 dataset=DGB-CanParl \
 wandb_conf.name=DyGrEncoder_Search_Rnd_Can \
 gpu=0 \
 lr=0.001,0.01,0.1 \
 model=DyGrEncoder \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True,False \
 model.link_pred.conv_num_layers=1,2 \
 model.link_pred.conv_aggr=mean,add,max \
 model.link_pred.lstm_num_layers=1 \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.epoch=20 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0,0.0001 \
