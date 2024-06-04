python dgt/run.py \
 wandb_conf.name=VGRNN_Hist_AS733 \
 dataset=DGB-AS733 \
 model=VGRNN \
 gpu=1 \
 lr=0.01 \
 task.engine.batch_size=1024 \
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
 task.engine.n_runs=1 \
 task.sampling=historical \


 python dgt/run.py \
 wandb_conf.name=VGRNN_Ind_AS733 \
 dataset=DGB-AS733 \
 model=VGRNN \
 gpu=1 \
 lr=0.01 \
 task.engine.batch_size=1024 \
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
 task.engine.n_runs=1 \
 task.sampling=inductive \