python tw_benchmark/run.py \
 --multirun \
 wandb_conf.name=EdgeBank_SearchTW_Alpha \
 gpu=1 \
 dataset=DGB-Bitcoin-Alpha \
 task.sampling=random \
 model=EdgeBank \
 model.evolving=False \
 model.link_pred.window=1,2,3,4,5,6,7,8,9,10,-1 \
 task.engine.n_runs=1 \
 task.engine.batch_size=1024 \
 task.engine.epoch=1 \