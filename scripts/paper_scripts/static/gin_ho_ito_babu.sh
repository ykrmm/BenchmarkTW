python dgt/run.py \
 gpu=2 \
 dataset=DPPIN-Ho \
 lr=0.01\
 model=Static \
 model.name=GIN \
 task.engine.batch_size=512 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=2 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \


python dgt/run.py \
 gpu=2 \
 dataset=DPPIN-Babu \
 lr=0.01\
 model=Static \
 model.name=GIN \
 task.engine.batch_size=512 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=2 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \


python dgt/run.py \
 gpu=2 \
 dataset=DPPIN-Ito \
 lr=0.01\
 model=Static \
 model.name=GIN \
 task.engine.batch_size=64 \
 optim.optimizer.weight_decay=0 \
 model.evolving=True \
 model.clip_grad=False \
 model.one_hot=True \
 model.link_pred.layers=2 \
 model.link_pred.hidden_dim=128 \
 task.engine.n_runs=5 \