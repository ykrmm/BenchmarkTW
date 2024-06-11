python tw_benchmark/run.py \
 --multirun \
 dataset=DGB-UNtrade \
 wandb_conf.name=GCLSTM_SearchTW_Trade \
 gpu=2 \
 lr=0.1 \
 model=GCLSTM \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 model.link_pred.time_length=1,2,3,4,5,6,7,8,9,10,-1 \
 task.engine.n_runs=1 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0001 \