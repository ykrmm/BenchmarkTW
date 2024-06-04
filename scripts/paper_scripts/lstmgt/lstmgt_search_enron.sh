python dgt/run.py \
 --multirun \
 wandb_conf.name=LSTMGT_Search_Enron \
 dataset=DGB-Enron \
 model=LSTMGT \
 gpu=1 \
 lr=0.1,0.01,0.001,0.0001 \
 task.engine.batch_size=128,256 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True,False \
 model.pred_next=False \
 model.link_pred.window=5 \
 model.link_pred.spatial_pe='rwpe' \
 model.link_pred.dim_emb=128 \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512,1024 \
 model.link_pred.nhead=2,4,8 \
 model.link_pred.num_layers_lstm=1,2 \
 model.link_pred.norm_first=False,True \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0,0.0005 \
 task.engine.n_runs=1 \