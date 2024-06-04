python dgt/run.py \
 gpu=0 \
 dataset=DGB-UNtrade \
 task.sampling=random \
 wandb_conf.name=EdgeBankTW-Rnd-Trade \
 model=EdgeBank \
 model.link_pred.mode=tw \
 task.engine.batch_size=4096 \
 model.evolving=False \
 task.engine.n_runs=5 \

 python dgt/run.py \
 gpu=0 \
 dataset=DGB-UNtrade \
 task.sampling=historical \
 wandb_conf.name=EdgeBankTW-Hist-Trade \
 model=EdgeBank \
 model.link_pred.mode=tw \
 task.engine.batch_size=4096 \
 model.evolving=False \
 task.engine.n_runs=5 \