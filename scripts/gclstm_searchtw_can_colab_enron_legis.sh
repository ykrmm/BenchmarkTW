python tw_benchmark/run.py \
 dataset=DGB-CanParl \
 wandb_conf.name=GCLSTM_SearchTW_Can \
 gpu=1 \
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

python tw_benchmark/run.py \
 --multirun \
 wandb_conf.name=GCLSTM_SearchTW_Colab \
 dataset=DGB-Colab \
 model=GCLSTM \
 gpu=1 \
 lr=0.001\
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.time_length=1,2,3,4,5,6,7,8,9,10,-1 \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \

python tw_benchmark/run.py \
 --multirun \
 wandb_conf.name=GCLSTM_SearchTW_Enron \
 dataset=DGB-Enron \
 model=GCLSTM \
 gpu=1 \
 lr=0.0001 \
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.K=2 \
 model.link_pred.time_length=1,2,3,4,5,6,7,8,9,10,-1 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=5e-7 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \

python tw_benchmark/run.py \
 --multirun \
 dataset=DGB-USLegis \
 wandb_conf.name=GCLSTM_SearchTW_Legis \
 gpu=1 \
 lr=0.1 \
 model=GCLSTM \
 model.evolving=False \
 model.pred_next=False \
 model.clip_grad=True \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.time_length=1,2,3,4,5,6,7,8,9,10,-1 \
 model.link_pred.undirected=True \
 task.engine.n_runs=1 \
 task.engine.batch_size=1024 \
 optim.optimizer.weight_decay=0.0001 \