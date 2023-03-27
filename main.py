import cv2
import os
import time
import gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import argparse
import multiprocessing as mp
import hydra
import pickle
import random
import sys
from copy import deepcopy
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.distributed as dist
from datetime import datetime
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path
from rich import print
from tqdm import tqdm
from functools import partial
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from minedojo.minedojo_wrapper import MineDojoEnv
from src.models.simple import SimpleNetwork
from src.utils import negtive_sample, EvalMetric
from src.utils.vision import create_backbone, resize_image
from src.utils.loss import get_loss_fn
from src.data.data_lmdb import LMDBTrajectoryDataset
from src.eval.parallel_eval import ParallelEval
torch.set_float32_matmul_precision('high')

import torch._dynamo
torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True

from src.utils.calculate_overhead import *

def making_exp_name(cfg):
    component = []
    if cfg['model']['use_horizon']:
        component.append('p:ho')
    else:
        component.append('p:bc')
    
    component.append("b:" + cfg['model']['backbone_name'][:4])
    
    today = datetime.now()
    
    component.append(f"{today.month}-{today.day}#{today.hour}-{today.minute}")
    
    return "@".join(component)

from mineclip.mineclip.mineclip import MineCLIP
def accquire_goal_embeddings(clip_path, goal_list, device="cuda"):
    clip_cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 
               'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}
    clip_model = MineCLIP(**clip_cfg)
    clip_model.load_ckpt(clip_path, strict=True)
    clip_model = clip_model.to(device)
    res = {}
    with torch.no_grad():
        for goal in goal_list:
            res[goal] = clip_model.encode_text([goal]).cpu().numpy()
    return res

class Trainer:
    
    def __init__(self, cfg, device, local_rank=0):
        
        self.action_space = [3, 3, 4, 11, 11, 8, 1, 1]
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank
        self.exp_name = making_exp_name(cfg)

        #! accquire goal embeddings
        print("[Progress] [red]Computing goal embeddings using MineClip's text encoder...")
        self.embedding_dict = accquire_goal_embeddings(cfg['pretrains']['clip_path'], cfg['data']['filters'])
        
        if not cfg["eval"]["only"]:
            #! use lmdb type dataset
            print("[Progress] [blue]Loading dataset...")
            self.train_dataset = LMDBTrajectoryDataset(
                in_dir=cfg['data']['train_data'],
                aug_ratio=cfg['optimize']['aug_ratio'] * cfg['optimize']['batch_size'],
                embedding_dict= self.embedding_dict,
                per_data_filters=cfg['data']['per_data_filters'], 
                skip_frame=cfg['data']['skip_frame'],
                window_len=cfg['data']['window_len'],
                padding_pos=cfg['data']['padding_pos'],
            )
        
            # if self.cfg.optimize.parallel:
            # HJ
            if self.cfg['optimize']['parallel']:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            else:
                self.train_sampler = torch.utils.data.sampler.RandomSampler(self.train_dataset)
            
            self.train_loader = DataLoader(
                self.train_dataset, 
                sampler=self.train_sampler, 
                pin_memory=True, 
                batch_size=cfg['optimize']['batch_size'], 
                num_workers=cfg['optimize']['num_workers'], 
            )
        
        backbone = create_backbone(
            name=cfg['model']['backbone_name'], 
            model_path=cfg['model']['model_path'], 
            weight_path=cfg['model']['weight_path'],
            goal_dim=cfg['model']['embed_dim'],
        )
        
        if cfg['model']['name'] == 'simple':
            if cfg['model']['compile']:
                self.model = torch.compile(SimpleNetwork(
                    action_space=self.action_space,
                    state_dim=cfg['model']['state_dim'],
                    goal_dim=cfg['model']['goal_dim'],
                    action_dim=cfg['model']['action_dim'],
                    num_cat=len(cfg['data']['filters']),
                    hidden_size=cfg['model']['embed_dim'],
                    fusion_type=cfg['model']['fusion_type'],
                    max_ep_len=cfg['model']['max_ep_len'],
                    backbone=backbone,
                    frozen_cnn=cfg['model']['frozen_cnn'],
                    use_recurrent=cfg['model']['use_recurrent'],
                    use_extra_obs=cfg['model']['use_extra_obs'],
                    use_horizon=cfg['model']['use_horizon'],
                    use_prev_action=cfg['model']['use_prev_action'],
                    extra_obs_cfg=cfg['model']['extra_obs_cfg'],
                    use_pred_horizon=cfg['model']['use_pred_horizon'],
                    c=cfg['model']['c'],
                    transformer_cfg=cfg['model']['transformer_cfg']
                ))
            else:
                self.model = SimpleNetwork(
                    action_space=self.action_space,
                    state_dim=cfg['model']['state_dim'],
                    goal_dim=cfg['model']['goal_dim'],
                    action_dim=cfg['model']['action_dim'],
                    num_cat=len(cfg['data']['filters']),
                    hidden_size=cfg['model']['embed_dim'],
                    fusion_type=cfg['model']['fusion_type'],
                    max_ep_len=cfg['model']['max_ep_len'],
                    backbone=backbone,
                    frozen_cnn=cfg['model']['frozen_cnn'],
                    use_recurrent=cfg['model']['use_recurrent'],
                    use_extra_obs=cfg['model']['use_extra_obs'],
                    use_horizon=cfg['model']['use_horizon'],
                    use_prev_action=cfg['model']['use_prev_action'],
                    extra_obs_cfg=cfg['model']['extra_obs_cfg'],
                    use_pred_horizon=cfg['model']['use_pred_horizon'],
                    c=cfg['model']['c'],
                    transformer_cfg=cfg['model']['transformer_cfg']
                )
            # torch.save(self.model, "save_model.pt")
        else:
            raise NotImplementedError
        
        self.iter_num = -1
        
        if cfg['model']['load_ckpt_path'] != "":
            state_dict = torch.load(cfg['model']['load_ckpt_path'])
            print(f"[MAIN] load checkpoint from {cfg['model']['load_ckpt_path']}. ")
            print(f"[MAIN] iter_num: {state_dict['iter_num']}, loss: {state_dict['loss']}")
            if cfg['model']['only_load_cnn']:
                backbone_state_dict = self.model.state_dict()
                backbone_state_dict.update({
                    k: v for k, v in state_dict['model_state_dict'].items() if 'backbone' in k
                })
                self.model.load_state_dict(backbone_state_dict)
            else:
                self.model.load_state_dict(state_dict['model_state_dict'])
                self.iter_num = state_dict['iter_num']
        
        self.model = self.model.to(self.device)
        # HJ
        # if self.cfg.optimize.parallel:
        if self.cfg['optimize']['parallel']:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank,
                find_unused_parameters=True
            )
            
        if not cfg["eval"]["only"]:
            
            self.policy_params = [x[1] for x in filter(lambda x: 'backbone' not in x[0], self.model.named_parameters())]
            if hasattr(self.model, 'module'):
                self.backbone_params = list(filter(lambda param: param.requires_grad, self.model.module.backbone.parameters()))
            else:
                self.backbone_params = list(filter(lambda param: param.requires_grad, self.model.backbone.parameters()))
            
            self.optimizer = torch.optim.AdamW([
                    {'params': self.policy_params, 'lr': cfg['optimize']['learning_rate']},
                    {'params': self.backbone_params, 'lr': cfg['optimize']['learning_rate'] * cfg['optimize']['backbone_ratio']},
                ],
                lr=cfg['optimize']['learning_rate'],
                weight_decay=cfg['optimize']['weight_decay'],
            )
            
            if self.cfg['model']['name'] == 'simple':
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lambda steps: min((steps+1)/cfg['optimize']['warmup_steps'], 1)
                )
            else:
                raise NotImplementedError

        assert len(cfg['eval']['goals']) > 0
        
        self.inp_goals = list(zip(cfg['eval']['goals'], cfg['eval']['env_id'])) * cfg['eval']['goal_ratio']
        
        eval_goals = list(set(self.cfg['eval']['goals']))
        print(f"[Prompt] [yellow]Candidate evaluation goals: {eval_goals}")
        
        self.eval_metric = EvalMetric(eval_goals, max_ep_len=self.cfg['eval']['max_ep_len'])
        # Parallel Evaluation
        self.parallel_eval = ParallelEval(
            model=self.model, 
            embedding_dict=self.embedding_dict,
            envs=self.cfg['eval']['envs'], 
            resolution=self.cfg['simulator']['resolution'],
            max_ep_len=self.cfg['eval']['max_ep_len'], 
            num_workers=self.cfg['eval']['num_workers'],
            device=self.device, 
            fps=self.cfg['eval']['fps'], 
            cfg=self.cfg,
        )

        now = datetime.now()

        ckpt_path = Path("./ckpts")
        ckpt_path.mkdir(exist_ok=True)
        print(f"[Prompt] [yellow]Current Checkpoint savig path: {str(ckpt_path)}")

        if cfg['record']['log_to_wandb']:
            wandb.init(
                project="multi-task decision making",
                config=cfg,
                name=self.exp_name,
            )
        
        self.loss_fns = []
        for k, v in cfg['loss'].items():
            if v['enable']:
                self.loss_fns.append({
                    'name': k,
                    'fn': get_loss_fn(v['fn']),
                    'weight': v['weight']
                })


    def run(self):
        
        if self.cfg['eval']['only']:
            self.eval_metric.reset()
            gnames, eval_results = self.parallel_eval.step(0, self.inp_goals)
            for goal_name, eval_result in zip(gnames, eval_results):
                self.eval_metric.add(goal_name, eval_result)
            metric_result = self.eval_metric.precision(k=3)
            
            fig_columns = ['timestep', 'success', 'goal']
            fig_data = []
            for _goal_name, _metric in metric_result.items():
                print(f"goal: {_goal_name}, pricision: {_metric['precision']}, pos: {_metric['pos']}, neg: {_metric['neg']}, tot: {_metric['tot']}, success: {_metric['success']}")

                for fig_t, fig_suc in enumerate(_metric['suc_per_step']):
                    fig_data.append((fig_t, fig_suc, _goal_name))
                
            fig_df = pd.DataFrame(fig_data, columns=fig_columns)
            print(fig_df.head())
            g = sns.relplot(x='timestep', y='success', hue='goal', kind='line', data=fig_df)
            g.savefig('accumulated_success.png', dpi = 300)
        else:
            self.start_time = time.time()
            for iter_num in range(self.iter_num+1, self.cfg['optimize']['max_iters']):
                # if self.cfg.optimize.parallel:
                # HJ
                if self.cfg['optimize']['parallel']:
                    dist.barrier()
                    self.train_loader.sampler.set_epoch(iter_num)
                train_losses = self.train_iteration(iter_num=iter_num+1)
                # if self.cfg.optimize.parallel:
                # HJ
                if self.cfg['optimize']['parallel']:
                    if dist.get_rank() != 0:
                        continue
                
                if (iter_num + 1) % self.cfg['record']['ckpt_freq'] == 0:
                    state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                    torch.save({
                        'iter_num': iter_num, 
                        'model_state_dict': state_dict,
                        'loss': np.mean(train_losses['loss']),
                    }, f"ckpts/ckpt_{iter_num}.pt")
                    
                # gpu_mem_overhead = check_gpu_mem_usedRate()
                # self.log_metrics(iter_num, train_losses, print_logs=True, gpu_mem_overhead=gpu_mem_overhead)
                self.log_metrics(iter_num, train_losses, print_logs=True)
                # sys.exit(0)
                if (iter_num + 1) % self.cfg['eval']['freq'] == 0:
                    self.eval_metric.reset()
                    gnames, eval_results = self.parallel_eval.step(iter_num, self.inp_goals)
                    for goal_name, eval_result in zip(gnames, eval_results):
                        self.eval_metric.add(goal_name, eval_result)
                    metric_result = self.eval_metric.precision(k=3)
                    for _goal_name, _metric in metric_result.items():
                        print(f"goal: {_goal_name}, pricision: {_metric['precision']}, positive: {_metric['pos']}, negative: {_metric['neg']}, total: {_metric['tot']}, horizon: {_metric['hor']}, success: {_metric['success']}")
                        if self.cfg['record']['log_to_wandb']:
                            wandb.log({
                                f'goal/precision/{_goal_name}': _metric['precision'],
                                f'goal/positive/{_goal_name}': _metric['pos'],
                                f'goal/negative/{_goal_name}': _metric['neg'],
                                f'goal/horizon/{_goal_name}': _metric['hor'],
                                f'goal/success/{_goal_name}': _metric['success'],
                            }, step=iter_num)
                        
                    if self.cfg['record']['log_to_wandb']:
                        wandb.log({
                            f'goal/mean-precision': sum(m['precision'] for m in metric_result.values()) / len(metric_result),
                            f'goal/mean-success': sum(m['success'] for m in metric_result.values()) / len(metric_result),
                        }, step=iter_num)
                


    def train_iteration(self, iter_num=0):

        self.model.train()
        train_losses = {}
        train_start = time.time()
        for batch in tqdm(self.train_loader):
            loss_result = self.train_step(batch)
            for key, loss in loss_result.items():
                train_losses[key] = train_losses.get(key, []) + [loss]

            if self.scheduler is not None:
                self.scheduler.step()
        
        return train_losses
    
    def log_metrics(self, iter_num, train_losses, print_logs=False, gpu_mem_overhead=0):
        
        logs = {}
        for key, loss_list in train_losses.items():
            logs[f'training/{key}_mean'] = np.mean(loss_list).item()
            logs[f'training/{key}_std'] = np.std(loss_list).item()
        # HJ
        logs['gpu_mem_overhead']=gpu_mem_overhead

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
        
        if self.cfg['record']['log_to_wandb']:
            for k, v in logs.items():
                try:
                    logs[k] = v.item()
                except:
                    pass
            wandb.log(logs, step=iter_num)


    def train_step(self, data):
        
        goals, states, actions, horizons, timesteps, attention_mask = data
        for k, v in states.items():
            states[k] = v.to(self.device)
        goals = goals.to(self.device) # HJ [batch_size, window_len, goal_dim]
        actions = actions.to(self.device) # HJ [batch_size, window_len, action_dim]
        horizons = horizons.to(self.device) # HJ [batch_size, window_len]
        timesteps = timesteps.to(self.device) #! timesteps is deperacated HJ [batch_size, window_len]
        attention_mask = attention_mask.to(self.device) # HJ [batch_size, window_len]
        
        if self.cfg['model']['name'] == 'simple':
            action_preds, mid_info = self.model(goals, states, horizons, timesteps, attention_mask) # HJ action_preds.shape=torch.Size([batch_size, window_len, 42?]), mid_info['pred_horizons'].shape=torch.Size([batch_size, window_len, 16(mlp_output_dim)])
            state_preds = None
            horizon_preds = None
            goal_preds = None
            neg_action_preds = None
        else:
            raise NotImplementedError

        params = {
            'mid_info': mid_info,
            'actions': actions,
            'states': states,
            'goals': goals,
            'horizons': horizons,
            'action_preds': action_preds,
            'neg_action_preds': neg_action_preds,
            'state_preds': state_preds,
            'horizon_preds': horizon_preds,
            'goal_preds': goal_preds,
            'attention_mask': attention_mask, 
            'action_space': self.action_space, 
            'gamma': self.cfg['optimize']['gamma'], 
        }

        loss = 0
        result = {}

        for loss_fn_cfg in self.loss_fns:
            this_loss = loss_fn_cfg['fn'](params)
            result[loss_fn_cfg['name']] = this_loss.detach().item()
            loss += loss_fn_cfg['weight'] * this_loss
        
        result['loss'] = loss.detach().item() 

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        return result  

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg):
    if cfg.optimize.parallel:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='nccl')
    else:
        local_rank = 0


    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    from rich.console import Console
    from rich.syntax import Syntax
    console = Console()
    syntax = Syntax(str(cfg), "json", theme="monokai", line_numbers=True)
    console.print(cfg)
    
    trainer = Trainer(cfg, device=device, local_rank=local_rank) 
    trainer.run()

if __name__ == "__main__":
    main()
    


