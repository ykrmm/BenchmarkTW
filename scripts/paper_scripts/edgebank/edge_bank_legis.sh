python dgt/run.py \
 gpu=0 \
 dataset=DGB-USLegis \
 task.sampling=random \
 wandb_conf.name=EdgeBank-Rnd-Legis \
 model=EdgeBank \
 task.engine.batch_size=1024 \
 model.evolving=True \
 task.engine.n_runs=5 \

 python dgt/run.py \
 gpu=0 \
 dataset=DGB-USLegis \
 task.sampling=historical \
 wandb_conf.name=EdgeBank-Hist-Legis \
 model=EdgeBank \
 task.engine.batch_size=1024 \
 model.evolving=True \
 task.engine.n_runs=5 \