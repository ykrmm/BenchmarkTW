python dgt/run.py \
 --multirun \
 wandb_conf.name=GCLSTM_NSS_Colab \
 dataset=DGB-Colab \
 model=GCLSTM \
 gpu=2 \
 lr=0.001\
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \
 task.sampling=historical,inductive \