python tw_benchmark/run.py \
 dataset=DGB-CanParl \
 wandb_conf.name=GCLSTM_SearchTW_Can \
 gpu=0 \
 lr=0.1 \
 model=GCLSTM \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.time_length=3 \
 model.link_pred.K=3 \
 model.link_pred.normalization=rw \
 model.link_pred.bias=True \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0 \