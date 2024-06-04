python dgt/run.py \
 gpu=0 \
 dataset=DGB-UNtrade \
 task.sampling=random \
 wandb_conf.name=EdgeBank-Rnd-Trade \
 model=EdgeBank \
 task.engine.batch_size=4096 \
 model.evolving=True \
 task.engine.n_runs=5 \

 python dgt/run.py \
 gpu=0 \
 dataset=DGB-UNtrade \
 task.sampling=historical \
 wandb_conf.name=EdgeBank-Hist-Trade \
 model=EdgeBank \
 task.engine.batch_size=4096 \
 model.evolving=True \
 task.engine.n_runs=5 \