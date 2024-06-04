python dgt/run.py \
 gpu=0 \
 dataset=DGB-Bitcoin-OTC \
 task.sampling=random \
 wandb_conf.name=EdgeBankTW-Rnd-Bitcoin \
 model=EdgeBank \
 model.link_pred.mode=tw \
 task.engine.batch_size=1024 \
 model.evolving=False \
 task.engine.epoch=5 \
 task.engine.n_runs=5 \

 python dgt/run.py \
 gpu=0 \
 dataset=DGB-Bitcoin-OTC \
 task.sampling=historical \
 wandb_conf.name=EdgeBankTW-Hist-Bitcoin \
 model=EdgeBank \
 model.link_pred.mode=tw \
 task.engine.batch_size=1024 \
 model.evolving=False \
 task.engine.epoch=5 \
 task.engine.n_runs=5 \