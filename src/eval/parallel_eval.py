import torch
import torch.nn.functional as F
import multiprocessing as mp
import random
import numpy as np
from rich import print
from datetime import datetime
from copy import deepcopy
from functools import partial
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from minedojo.minedojo_wrapper import MineDojoEnv
from src.utils.vision import resize_image
from src.eval.eval_worker import EnvWorker
from src.utils.foundation import discrete_horizon

from PIL import Image, ImageDraw
import cv2
from cv2 import VideoCapture, VideoWriter_fourcc, VideoWriter, cvtColor

def resize_image_numpy(img, target_resolution = (128, 128)):
    img = cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)
    return img

def env_generator_fn(env_names: list, img_size) -> dict:
    envs = {}
    for name in env_names:
        env = MineDojoEnv(
            name=name, 
            img_size=img_size,
            rgb_only=False,
        )
        envs[name] = env
    return envs, env_names

class ParallelEval:
    
    def __init__(self, model, embedding_dict: dict, envs: list, resolution: tuple, 
                 max_ep_len: int, num_workers: int, device: str, fps: int = 1000, cfg={}): 
        self.model = model
        self.embedding_dict = embedding_dict
        self.envs = envs
        self.resolution = resolution
        self.max_ep_len = max_ep_len
        self.num_workers = num_workers
        self.device = device
        self.fps = fps
        self.cfg = cfg
        
        # self.goal_to_num = {name: id for id, name in self.num_to_goal.items()}
        
        self.env_workers = []
        self.pipes = []
        
        for i in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            self.pipes.append(parent_pipe)
            worker = EnvWorker(
                child_pipe, 
                partial(env_generator_fn, self.envs, self.resolution), 
                worker_id = i,
                max_episode_length = self.max_ep_len
            )
            self.env_workers.append(worker)
        
        # Start all workers
        for worker in self.env_workers:
            worker.start()
    
    @torch.no_grad()
    def step(self, iter_num, goal_env_pairs):
        remaining_goals = deepcopy(goal_env_pairs)

        self.model.eval()

        horizon = self.cfg['eval']['horizon']
        #! discrete_horizon
        horizon = discrete_horizon(horizon)
        per_worker_horizon = [deepcopy(horizon) for _ in range(self.num_workers)]

        worker_status = [False for _ in range(self.num_workers)] # 0: idle; 1: busy

        # per_worker_goal = [None for _ in range(self.num_workers)]
        per_worker_target_item = [None for _ in range(self.num_workers)]

        def preprocess_obs(obs: dict):
            res_obs = {}
            rgb = torch.from_numpy(obs['rgb']).unsqueeze(0).to(device=self.device, dtype=torch.float32).permute(0, 3, 1, 2)
            res_obs['rgb'] = resize_image(rgb, target_resolution=(120, 160))
            res_obs['voxels'] = torch.from_numpy(obs['voxels']).reshape(-1).unsqueeze(0).to(device=self.device, dtype=torch.long)
            res_obs['compass'] = torch.from_numpy(obs['compass']).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            res_obs['gps'] = torch.from_numpy(obs['gps'] / np.array([1000., 100., 1000.])).unsqueeze(0).to(device=self.device, dtype=torch.float32)
            res_obs['biome'] = torch.from_numpy(obs['biome_id']).unsqueeze(0).to(device=self.device, dtype=torch.long)
            return res_obs

        def stack_obs(prev_obs: dict, obs: dict):
            stacked_obs = {}
            stacked_obs['rgb'] = torch.cat([prev_obs['rgb'], obs['rgb']], dim = 0)
            stacked_obs['voxels'] = torch.cat([prev_obs['voxels'], obs['voxels']], dim = 0)
            stacked_obs['compass'] = torch.cat([prev_obs['compass'], obs['compass']], dim = 0)
            stacked_obs['gps'] = torch.cat([prev_obs['gps'], obs['gps']], dim = 0)
            stacked_obs['biome'] = torch.cat([prev_obs['biome'], obs['biome']], dim = 0)
            return stacked_obs

        def slice_obs(obs: dict, slice: torch.tensor):
            res = {}
            for k, v in obs.items():
                res[k] = v[slice]
            return res

        per_worker_actions = [
            torch.zeros((1, self.cfg['model']['action_dim']), device=self.device) for _ in range(self.num_workers)
        ]
        per_worker_horizons = [
            torch.tensor([horizon], device=self.device, dtype=torch.long) for _ in range(self.num_workers)
        ]
        per_worker_timesteps = [
            torch.tensor([0], device=self.device, dtype=torch.long) for _ in range(self.num_workers)
        ]
        per_worker_acquire = [
            None for _ in range(self.num_workers)
        ]
        per_worker_goals = [
            None for _ in range(self.num_workers)
        ]
        per_worker_states = [
            None for _ in range(self.num_workers)
        ]
        per_worker_obs = [
            [] for _ in range(self.num_workers)
        ]
        per_worker_pred_horizons = [
            [] for _ in range(self.num_workers)
        ]

        # Give a job to every worker
        assert len(remaining_goals) >= self.num_workers
        for worker_id in range(self.num_workers):
            goal, env = remaining_goals.pop()
            # n_goal = self.goal_to_num[goal]
            self._send_message(worker_id, "eval_begin", (goal, env))

            # per_worker_goal[worker_id] = n_goal
            per_worker_target_item[worker_id] = goal
            worker_status[worker_id] = True

            per_worker_acquire[worker_id] = []

        sf = self.cfg['data']['skip_frame']
        wl = self.cfg['data']['window_len']

        all_goal_names = []
        all_acquired_items = []
        while any(worker_status):
            for worker_id in range(self.num_workers):
                command, args = self._recv_message_nonblocking(worker_id)
                if command is None:
                    continue

                if command == "request_action":
                    goal, robs, t = args
                    # n_goal = self.goal_to_num[goal]
                    # import ipdb; ipdb.set_trace()
                    
                    obs = preprocess_obs(robs)
                    reward = 0
                    env_done = False
                    info = None

                    per_worker_states[worker_id] = obs
                    
                    per_worker_goals[worker_id] = torch.from_numpy(self.embedding_dict[goal]).to(self.device)

                elif command == "request_action_with_info":
                    goal, robs, reward, env_done, info, t = args
                    
                    obs = preprocess_obs(robs)
                else:
                    raise ValueError()

                per_worker_obs[worker_id].append(robs['rgb'])

                # extract frame by every <skip_frame> frames
                rg = torch.arange(t, max(t-sf*(wl-1)-1, -1), -sf).flip(0)

                temp_states = slice_obs(per_worker_states[worker_id], rg)
                temp_states['prev_action'] = per_worker_actions[worker_id][rg]
                if self.cfg['model']['name'] == 'simple':
                    get_action = self.model.module.get_action if hasattr(self.model, 'module') else self.model.get_action
                    action_preds, mid_info = get_action(
                        goals=per_worker_goals[worker_id][rg], 
                        states=temp_states, 
                        horizons=per_worker_horizons[worker_id][rg], 
                    )
                    per_worker_pred_horizons[worker_id].append(mid_info['pred_horizons'].argmax(-1)[0, -1].item())
                else:
                    raise NotImplementedError()

                action_preds = action_preds[:, -1]
                action_space = self.model.module.action_space if hasattr(self.model, 'module') else self.model.action_space
                action_dist = TorchMultiCategorical(action_preds,  None, action_space)
                action = action_dist.sample().squeeze(0)

                if t < self.cfg['eval']['max_ep_len']:
                    self._send_message(worker_id, "eval_step", action.cpu().numpy())

                if info is not None:
                    if len(info['accomplishments']) > 0:
                        acquire = per_worker_acquire[worker_id]
                        acquire = acquire + [ (item, t) for item in info['accomplishments']]
                        per_worker_acquire[worker_id] = acquire
                        
                        # goal_name = self.num_to_goal[goal]
                        if goal in info['accomplishments']:
                            color = 'italic green'
                        else:
                            color = 'italic red'
                        print(f"[eval] acquire: [{color}]{info['accomplishments']}[/{color}] in step {t}!, goal is <{goal}>.")

                per_worker_states[worker_id] = stack_obs(per_worker_states[worker_id], obs)
                per_worker_goals[worker_id] = torch.cat([per_worker_goals[worker_id], per_worker_goals[worker_id][-1:, ...]], dim = 0)
                per_worker_actions[worker_id] = torch.cat([per_worker_actions[worker_id], action.unsqueeze(0)], dim = 0)
                per_worker_horizons[worker_id] = torch.cat([per_worker_horizons[worker_id], torch.tensor([horizon], device=self.device, dtype=torch.long)], dim=0)
                per_worker_timesteps[worker_id] = torch.cat([per_worker_timesteps[worker_id], torch.tensor([t], device=self.device, dtype=torch.long)], dim=0)

                if t >= self.max_ep_len:
                    all_goal_names.append(per_worker_target_item[worker_id])
                    all_acquired_items.append(per_worker_acquire[worker_id])
                    
                    # save gif or video
                    now = datetime.now()
                    timestamp = f"{now.hour}_{now.minute}_{now.second}"
                    file_name = f"{per_worker_target_item[worker_id]}_{timestamp}"
                    imgs = []
                    for id, frame in enumerate(per_worker_obs[worker_id]):
                        frame = resize_image_numpy(frame).astype('uint8')
                        # import ipdb; ipdb.set_trace()
                        horizon_text = per_worker_pred_horizons[worker_id][id]
                        cv2.putText(
                            frame,
                            f"H: {horizon_text}, T: {id}",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )
                        imgs.append(Image.fromarray(frame))
                    # imgs = [Image.fromarray(resize_image_numpy(img).astype('uint8')) for img in per_worker_obs[worker_id]]
                    imgs = imgs[::3]
                    print(f"img length: {len(imgs)}")
                    imgs[0].save(file_name + ".gif", save_all=True, append_images=imgs[1:], optimize=False, quality=0, duration=150, loop=0)

                    if len(remaining_goals) > 0:
                        goal, env = remaining_goals.pop()
                        self._send_message(worker_id, "eval_begin", (goal, env))

                        # per_worker_goal[worker_id] = goal
                        per_worker_target_item[worker_id] = goal
                        worker_status[worker_id] = True

                        per_worker_acquire[worker_id] = []
                        per_worker_obs[worker_id] = []

                        per_worker_actions[worker_id] = torch.zeros((1, self.cfg['model']['action_dim']), device=self.device)
                        per_worker_horizons[worker_id] = torch.tensor([horizon], device=self.device, dtype=torch.long)
                        per_worker_timesteps[worker_id] = torch.tensor([0], device=self.device, dtype=torch.long)
                    else:
                        worker_status[worker_id] = False

        for worker_id in range(self.num_workers):
            self._send_message(worker_id, "close_env")

        return all_goal_names, all_acquired_items

    def _broadcast_message(self, command, args = None):
        for worker_idx in range(self.num_workers):
            self._send_message(worker_idx, command, args = args)

    def _send_message(self, worker_idx, command, args = None):
        self.pipes[worker_idx].send((command, args))

    def _recv_message_nonblocking(self, worker_idx):
        if not self.pipes[worker_idx].poll():
            return None, None

        command, args = self.pipes[worker_idx].recv()

        return command, args
