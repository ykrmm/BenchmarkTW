python dgt/run.py \
 --multirun \
 dataset=DGB-USLegis \
 wandb_conf.name=GCLSTM_Search_Rnd_Legis \
 gpu=2 \
 lr=0.001,0.01,0.1 \
 model=GCLSTM \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True,False \
 model.link_pred.K=1,2,3 \
 model.link_pred.normalization=sym,rw \
 model.link_pred.bias=True,False \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.epoch=20 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0,0.0001 \
