from typing import Dict, List, Any
import logging
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import wandb
import omegaconf
from omegaconf import DictConfig
from hydra.utils import instantiate
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tw_benchmark.engine import EngineBase
from tw_benchmark.datasets import RegData, ListLinkPredDataset 
from tw_benchmark.lib import feed_dict_to_device

NoneType = type(None)
ArgsType = List[Any]
KwargsType = Dict[str, Any]


class EngineReg(EngineBase):
    def __init__(
        self,
        config: DictConfig,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        dts: RegData,
        logger: logging.Logger,
        *args: ArgsType,
        **kwargs: KwargsType,
    ) -> NoneType: # type: ignore
        super().__init__(*args, **kwargs)
        logger.info("EngineReg initialization.")
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dts = dts
        self.device = device
        self.logger = logger
    
    
        
    def train(self, datasets, T_train):
        # Train for one epoch 
        self.model.train()
        losses = []
        t_min = 1 if self.pred_next else 0
        for t in range(t_min, T_train):
            dataset_t = datasets.get_dataset_t(t)
            graphs_t = dataset_t.get_graphs()
            
            dataloader_t = instantiate(
                self.config.task.engine.train_loader,
                dataset=dataset_t,
                shuffle=self.config.task.engine.shuffle_train_loader,
            )
            
            for feed_dict in dataloader_t:
                feed_dict = feed_dict_to_device(feed_dict, self.device)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=('cuda'), dtype=torch.float16, enabled=self.use_amp):
                    loss = self.model.get_loss_node_pred(feed_dict, graphs_t)  # [B,C] B: batch size, C: number of classes           
                    losses.append(loss.item())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                if self.config.model.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        mean_loss = sum(losses)/len(losses)
        return mean_loss
    
    
    def eval(self,datasets: ListLinkPredDataset, t_min_max : List[int]):
        self.model.eval()
        scores = {}
        t_min,t_max = t_min_max
        all_pred = []
        all_targets = []
        for t in range(t_min,t_max):
            dataset_t = datasets.get_dataset_t(t)
            graphs_t = dataset_t.get_graphs()
            dataloader_t = instantiate(
                self.config.task.engine.train_loader,
                dataset=dataset_t,
                shuffle=False,
            )
            for feed_dict in dataloader_t:
                feed_dict = feed_dict_to_device(feed_dict,self.device)
                with torch.autocast(device_type=('cuda'), dtype=torch.float16, enabled=self.use_amp):
                    pred = self.model.score_eval_node_reg(feed_dict,graphs_t)
                all_pred.append(pred)
                all_targets.append(feed_dict['y'])
                
        all_pred = torch.cat(all_pred)
        all_targets = torch.cat(all_targets)

        # Convertir les tenseurs en tableaux NumPy
        predictions_np = all_pred.detach().cpu().numpy()
        targets_np = all_targets.detach().cpu().numpy()

        # Calculer la MSE et la MAE pour l'ensemble de l'epoch
        mse = mean_squared_error(targets_np, predictions_np)
        mae = mean_absolute_error(targets_np, predictions_np)

        scores['MSE'] = mse
        scores['MAE'] = mae
        return scores

    def run(self):
        # info 
        self.logger.info("Model: {}".format(self.model.__class__.__name__))
        self.logger.info("Criterion: {}".format(self.criterion.__class__.__name__))
        self.logger.info("Dataset: {}".format(self.config.dataset.name))
        self.logger.info("Number of nodes: {}".format(self.model.num_nodes))
        self.logger.info("Total number of snapshots: {}".format(self.config.dataset.timestamp))
        self.logger.info("Undirected graphs: {}".format(self.config.model.link_pred.undirected))
        self.logger.info("One hot features: {}".format(self.config.model.one_hot))
        
        # wandb 
        config_wandb =  omegaconf.OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        wandb.init(
            entity=self.config.wandb_conf.entity,
            project=self.config.wandb_conf.project_link_pred,
            name=self.config.wandb_conf.name,
            config=config_wandb,
            reinit=True,  # reinit=True is necessary when calling multirun with hydra 
        ) 
        if self.config.multithreading:
            self.logger.info("Using multithreading")
            wandb.init(settings=wandb.Settings(start_method="thread"))

        # amp 
        try: 
            self.use_amp = self.model.flash
        except:
            self.use_amp = False
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Get parameters
        epoch = self.config.task.engine.epoch
        eval_every = self.config.task.engine.eval_every
        n_runs = self.config.task.engine.n_runs
        seeds = np.arange(n_runs)
        all_mse_test = []
        all_mae_test = []

        
        for i in range(n_runs):
            self.logger.info("Run {}/{}".format(i+1,n_runs))
            
            # Handle Reproducibility 
            torch.manual_seed(seeds[i])
            np.random.seed(seeds[i])
            random.seed(seeds[i])
        
            #  dataset and dataloaders
            self.data_name = self.config.dataset.name
            self.logger.info("Constructing the Discrete Dynamic Graph")
            st = time.time()
            train_dts, val_dts, test_dts = self.dts.get_datasets()
            t_test = self.dts.T_test
            t_val = self.dts.T_val
            t_train = self.dts.T_train
            et = time.time()
            self.logger.info("Time to construct the Discrete Dynamic Graph: {:.4f} s".format(et-st))
            self.logger.info("Number of snapshots in train: {}".format(t_train))
            self.logger.info("Number of snapshots in val: {}".format(t_val - t_train))
            self.logger.info("Number of snapshots in test: {}".format(t_test - t_val ))
            self.logger.info("Constructing the dataloaders")
            
            # scheduler 
            scheduler = MultiStepLR(self.optimizer, milestones=[50,100], gamma=0.001)
            self.pred_next = self.config.model.pred_next
                        
            # model and graphs to gpu 
            self.model.to(self.device)
            try: 
                self.model.set_device(self.device) # for models that need to know the device
            except:
                pass

            # log model
            self.logger.info("Model on device: {}".format(self.device))
            
            # Training loop
            self.logger.info("Start training")
            val_mse = []
            train_mse = []
            test_mse = []
            test_mae = []
            min_val_mse = float('inf')
            
            for ep in range(epoch):
                # train

                self.logger.info(f"Epoch {ep}")
                et = time.time()
                mean_loss = self.train(train_dts,t_train)
                st = time.time()
                self.logger.info(f"Time to train one epoch: {st-et:.4f} s")
                if i == 0:
                    # only log the first run
                    wandb.log({"Loss Train "+str(self.data_name): mean_loss},step=ep)
                self.logger.info(f"Loss: {mean_loss}")
                #scheduler.step()

                # eval
                if ep % eval_every == 0:
                    # val
                    scores = self.eval(val_dts,[t_train,t_val])
                    if i == 0 : 
                        for name,score in scores.items():
                            wandb.log({name+" Val "+str(self.data_name): score},step=ep)
                    current_val_mse = scores['MSE']
                    self.logger.info(f"MSE Val: {current_val_mse}")
                    val_mse.append(current_val_mse)
                    
                    # test
                    scores = self.eval(test_dts, [t_val,t_test])
                    if i == 0 : 
                        for name,score in scores.items():
                            wandb.log({name+" Test "+str(self.data_name): score},step=ep)
                    self.logger.info(f"MSE Test: {scores['MSE']}")
                    test_mse.append(scores['MSE'])
                    test_mae.append(scores['MAE'])

                    # early stopping

                    if current_val_mse < min_val_mse:
                        min_val_mse = current_val_mse
                        runs_without_improvement = 0
                    else:
                        runs_without_improvement += 1
                    if runs_without_improvement >= self.config.task.engine.early_stopping: # No early stopping for the first run
                        self.logger.info("Early stopping")
                        break
                

            self.logger.info("Training finished")
            self.logger.info("Best MSE on val: {}".format(max(val_mse)))
            self.logger.info("MSE on test: {}".format(test_mse[val_mse.index(max(val_mse))]))
            all_mse_test.append(test_mse[val_mse.index(max(val_mse))])
            all_mae_test.append(test_mae[val_mse.index(max(val_mse))])
        
        final_mse_test = np.mean(all_mse_test)
        final_mae_test = np.mean(all_mae_test)
        final_mse_test_std = np.std(all_mse_test)
        final_mae_test_std = np.std(all_mae_test)
        self.logger.info("Mean MSE on test: {}".format(final_mse_test))
        self.logger.info("Std MSE on test: {}".format(final_mse_test_std))
        self.logger.info("Mean MAE on test: {}".format(final_mae_test))
        self.logger.info("Std AE on test: {}".format(final_mae_test_std))
        wandb.run.summary["Mean MSE Test"] = final_mse_test
        wandb.run.summary["Mean MAE Test"] = final_mae_test
        wandb.run.summary["Std MSE Test"] = final_mse_test_std
        wandb.run.summary["Std MAE Test"] = final_mae_test_std
                
        
    
