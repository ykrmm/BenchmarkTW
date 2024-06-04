python dgt/run.py \
 gpu=0 \
 dataset=DGB-Bitcoin-OTC \
 task.sampling=random \
 wandb_conf.name=EdgeBank-Rnd-Bitcoin \
 model=EdgeBank \
 task.engine.batch_size=1024 \
 model.evolving=True \
 task.engine.epoch=5 \
 task.engine.n_runs=5 \

 python dgt/run.py \
 gpu=0 \
 dataset=DGB-Bitcoin-OTC \
 task.sampling=historical \
 wandb_conf.name=EdgeBank-Hist-Bitcoin \
 model=EdgeBank \
 task.engine.batch_size=1024 \
 model.evolving=True \
 task.engine.epoch=5 \
 task.engine.n_runs=5 \