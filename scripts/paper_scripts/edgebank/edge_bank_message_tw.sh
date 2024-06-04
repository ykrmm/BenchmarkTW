python dgt/run.py \
 gpu=0 \
 dataset=DGB-UCI-Message \
 task.sampling=random \
 wandb_conf.name=EdgeBankTW-Rnd-Message \
 model=EdgeBank \
 model.link_pred.mode=tw \
 task.engine.batch_size=1024 \
 model.evolving=False \
 task.engine.n_runs=5 \
 task.engine.epoch=5 \

 python dgt/run.py \
 gpu=0 \
 dataset=DGB-UCI-Message \
 task.sampling=historical \
 wandb_conf.name=EdgeBankTW-Hist-Message \
 model=EdgeBank \
 model.link_pred.mode=tw \
 task.engine.batch_size=1024 \
 model.evolving=False \
 task.engine.n_runs=5 \
 task.engine.epoch=5 \