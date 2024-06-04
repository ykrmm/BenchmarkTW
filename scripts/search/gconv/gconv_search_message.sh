python dgt/run.py \
 --multirun \
 wandb_conf.name=GConvLSTM_Search_rnd_Message \
 dataset=DGB-UCI-Message \
 model=GConvLSTM \
 gpu=2 \
 lr=0.01,0.1,0.001 \
 model.evolving=False,True \
 task.engine.batch_size=1024 \
 model.link_pred.K=1,2,3 \
 model.pred_next=False,True \
 model.link_pred.normalization=sym,rw \
 model.link_pred.bias=False,True \
 model.link_pred.undirected=False,True \
 model.clip_grad=False,True \
 optim.optimizer.weight_decay=0,0.0001 \
 task.engine.n_runs=1 \
 task.engine.epoch=30 \