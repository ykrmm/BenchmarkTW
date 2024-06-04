python dgt/run.py \
 wandb_conf.name=VGRNN_Hist_Enron \
 dataset=DGB-Enron \
 model=VGRNN \
 gpu=0 \
 lr=0.01 \
 task.engine.batch_size=16 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.h_dim=32 \
 model.link_pred.z_dim=16 \
 model.link_pred.n_layers=1 \
 model.link_pred.eps=1e-10 \
 model.link_pred.conv=GCN \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=3 \
 task.sampling=historical \


 python dgt/run.py \
 wandb_conf.name=VGRNN_Ind_Enron \
 dataset=DGB-Enron \
 model=VGRNN \
 gpu=0 \
 lr=0.01 \
 task.engine.batch_size=16 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.h_dim=32 \
 model.link_pred.z_dim=16 \
 model.link_pred.n_layers=1 \
 model.link_pred.eps=1e-10 \
 model.link_pred.conv=GCN \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=3 \
 task.sampling=inductive \