python dgt/run.py \
 gpu=0 \
 dataset=DGB-UCI-Message \
 task.sampling=random \
 wandb_conf.name=EdgeBank-Rnd-Message \
 model=EdgeBank \
 task.engine.batch_size=1024 \
 model.evolving=True \
 task.engine.epoch=5 \
 task.engine.n_runs=5 \

 python dgt/run.py \
 gpu=0 \
 dataset=DGB-UCI-Message \
 task.sampling=historical \
 wandb_conf.name=EdgeBank-Hist-Message \
 model=EdgeBank \
 task.engine.batch_size=1024 \
 model.evolving=True \
 task.engine.epoch=5 \
 task.engine.n_runs=5 \