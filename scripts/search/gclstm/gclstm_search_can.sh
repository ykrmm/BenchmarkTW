python dgt/run.py \
 --multirun \
 wandb_conf.name=GCLSTM_search_Can \
 dataset=DGB-CanParl \
 model=GCLSTM \
 gpu=2 \
 lr=0.001,0.01,0.1 \
 task.engine.batch_size=1024 \
 model.link_pred.K=1,2,3 \
 model.link_pred.normalization=rw,sym \
 model.link_pred.bias=True,False \
 model.evolving=False,True \
 model.pred_next=False,True \
 model.clip_grad=False,True \
 optim.optimizer.weight_decay=0 \