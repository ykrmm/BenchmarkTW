python tw_benchmark/run.py \
 --multirun \
 dataset=DGB-UCI-Message \
 wandb_conf.name=GCLSTM_SearchParam_Message \
 gpu=1 \
 lr=0.1,0.01,0.001,0.0001,0.00001 \
 model=GCLSTM \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.time_length=1 \
 model.link_pred.K=2 \
 model.link_pred.normalization=rw \
 model.link_pred.bias=True \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \