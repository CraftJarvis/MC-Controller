import numpy as np
import os, math
import random
import lmdb
import json
import pickle
import torch
from torch.utils.data import Dataset
# from src.utils.foundation import discrete_horizon
from tqdm import tqdm
from typing import List
from rich import print
from typing import *
import cv2
from PIL import Image, ImageDraw

def discrete_horizon(horizon):
    '''
    0 - 10: 0
    10 - 20: 1
    20 - 30: 2
    30 - 40: 3
    40 - 50: 4
    50 - 60: 5
    60 - 70: 6
    70 - 80: 7
    80 - 90: 8
    90 - 100: 9
    100 - 120: 10
    120 - 140: 11
    140 - 160: 12
    160 - 180: 13
    180 - 200: 14
    200 - ...: 15
    '''
    # horizon_list = [0]*25 + [1]*25 + [2]*25 + [3]*25 +[4]* 50 + [5]*50 + [6] * 700
    horizon_list = []
    for i in range(10):
        horizon_list += [i] * 10
    for i in range(10, 15):
        horizon_list += [i] * 20
    horizon_list += [15] * 700
    if type(horizon) == torch.Tensor:
        return torch.Tensor(horizon_list, device=horizon.device)[horizon]
    elif type(horizon) == np.ndarray:
        return np.array(horizon_list)[horizon]
    elif type(horizon) == int:
        return horizon_list[horizon]
    else:
        assert False

class LMDBTrajectoryDataset(Dataset):
    
    def __init__(self, 
                 in_dir: Union[str, list], 
                 aug_ratio: float, 
                 embedding_dict: dict, 
                 per_data_filters:list=None,
                 skip_frame: int=3,
                 window_len: int=20,
                 chunk_size: int=8,
                 padding_pos: str='left',
                 random_start: bool=True):
        
        super().__init__()
        if type(in_dir) == str:
            self.base_dirs = [in_dir]
        else:
            self.base_dirs = in_dir
        
        self.aug_ratio = aug_ratio
        self.embedding_dict = embedding_dict
        self.filters = list(embedding_dict.keys())
        self.skip_frame = skip_frame
        self.window_len = window_len
        self.chunk_size = chunk_size
        self.padding_pos = padding_pos
        self.random_start = random_start
        
        self.traj_dirs = [os.path.join(base_dir, "trajs") for base_dir in self.base_dirs]
        self.indices_dirs = [os.path.join(base_dir, "indices") for base_dir in self.base_dirs]

        self.lmdb_traj_envs = [lmdb.open(traj_dir, max_readers=12600, lock=False) for traj_dir in self.traj_dirs]
        self.lmdb_indices_envs = [lmdb.open(indices_dir, max_readers=12600, lock=False) for indices_dir in self.indices_dirs]

        # Build index
        self.trajs_info = {}
        self.trajs_by_goal = {}
        for dataset_id, lmdb_indices_env in enumerate(self.lmdb_indices_envs):
            if per_data_filters is None:
                per_data_filter = []
            else:
                per_data_filter = per_data_filters[dataset_id]
            txn = lmdb_indices_env.begin()
            for key, value in txn.cursor():
                traj_name = key.decode()
                vals = json.loads(value)
                vals["dataset_id"] = dataset_id
                
                record_flag = True
                if len(per_data_filter) > 0:
                    flag = False
                    for accomplishment in vals["accomplishments"]:
                        if accomplishment in per_data_filter:
                            flag = True
                            break
                    if not flag:
                        record_flag = False
                
                if record_flag:
                    self.trajs_info[traj_name] = vals

                    for accomplishment in vals["accomplishments"]:
                        if accomplishment not in self.trajs_by_goal:
                            self.trajs_by_goal[accomplishment] = []
                        self.trajs_by_goal[accomplishment].append(traj_name)

    def __len__(self):
        return self.aug_ratio

    def padding(self, goal, state, action, horizon, timestep):
        
        window_len = self.window_len
        traj_len = goal.shape[0]
        
        rgb_dim = state['rgb'].shape[1:]
        voxels_dim = state['voxels'].shape[1:]
        compass_dim = state['compass'].shape[1:]
        gps_dim = state['gps'].shape[1:]
        biome_dim = state['biome'].shape[1:]
        
        action_dim = action.shape[1:]
        goal_dim = goal.shape[1:]
        
        if self.padding_pos == 'left':
            state['rgb'] = np.concatenate([np.zeros((window_len - traj_len, *rgb_dim)), state['rgb']], axis=0)
            state['voxels'] = np.concatenate([np.zeros((window_len - traj_len, *voxels_dim)), state['voxels']], axis=0)
            state['compass'] = np.concatenate([np.zeros((window_len - traj_len, *compass_dim)), state['compass']], axis=0)
            state['gps'] = np.concatenate([np.zeros((window_len - traj_len, *gps_dim)), state['gps']], axis=0)
            state['biome'] = np.concatenate([np.zeros((window_len - traj_len, *biome_dim)), state['biome']], axis=0)
            state['prev_action'] = np.concatenate([np.zeros((window_len - traj_len, *action_dim)), state['prev_action']], axis=0)
            goal = np.concatenate([np.zeros((window_len - traj_len, *goal_dim)), goal], axis=0)
            action = np.concatenate([np.zeros((window_len - traj_len, *action_dim)), action], axis=0)
            horizon = np.concatenate([np.zeros((window_len - traj_len)), horizon], axis=0)
            timestep = np.concatenate([np.zeros((window_len - traj_len)), timestep], axis=0)
            mask = np.concatenate([np.zeros((window_len - traj_len)), np.ones((traj_len))], axis=0)
            
        elif self.padding_pos == 'right':
            state['rgb'] = np.concatenate([state['rgb'], np.zeros((window_len - traj_len, *rgb_dim))], axis=0)
            state['voxels'] = np.concatenate([state['voxels'], np.zeros((window_len - traj_len, *voxels_dim))], axis=0)
            state['compass'] = np.concatenate([state['compass'], np.zeros((window_len - traj_len, *compass_dim))], axis=0)
            state['gps'] = np.concatenate([state['gps'], np.zeros((window_len - traj_len, *gps_dim))], axis=0)
            state['biome'] = np.concatenate([state['biome'], np.zeros((window_len - traj_len, *biome_dim))], axis=0)
            state['prev_action'] = np.concatenate([state['prev_action'], np.zeros((window_len - traj_len, *action_dim))], axis=0)
            goal = np.concatenate([goal, np.zeros((window_len - traj_len, *goal_dim))], axis=0)
            action = np.concatenate([action, np.zeros((window_len - traj_len, *action_dim))], axis=0)
            horizon = np.concatenate([horizon, np.zeros((window_len - traj_len))], axis=0)
            timestep = np.concatenate([timestep, np.zeros((window_len - traj_len))], axis=0)
            mask = np.concatenate([np.ones((traj_len)), np.zeros((window_len - traj_len))], axis=0)
        
        else:
            assert False
        
        state['rgb'] = torch.from_numpy(state['rgb']).float()
        state['voxels'] = torch.from_numpy(state['voxels']).long()
        state['compass'] = torch.from_numpy(state['compass']).float()
        state['gps'] = torch.from_numpy(state['gps']).float()
        state['biome'] = torch.from_numpy(state['biome']).long()
        state['prev_action'] = torch.from_numpy(state['prev_action']).float()
        action = torch.from_numpy(action).float()
        goal = torch.from_numpy(goal).float()
        horizon = torch.from_numpy(horizon).long()
        timestep = torch.from_numpy(timestep).long()
        mask = torch.from_numpy(mask).long()
        
        return goal, state, action, horizon, timestep, mask

    def __getitem__(self, idx):
        
        if self.filters is not None:
            goal = random.choice(self.filters)
        else:
            goal = random.choice(list(self.trajs_by_goal.keys()))
        
        traj_id = np.random.randint(0, len(self.trajs_by_goal[goal]))
        traj_name = self.trajs_by_goal[goal][traj_id]
        traj_metadata = self.trajs_info[traj_name]
        dataset_id = traj_metadata["dataset_id"]

        num_chunks = traj_metadata["num_chunks"]
        horizon = traj_metadata["horizon"]
        
        while horizon <=2:
            traj_id = np.random.randint(0, len(self.trajs_by_goal[goal]))
            traj_name = self.trajs_by_goal[goal][traj_id]
            traj_metadata = self.trajs_info[traj_name]
            dataset_id = traj_metadata["dataset_id"]

            num_chunks = traj_metadata["num_chunks"]
            horizon = traj_metadata["horizon"]
        
        assert horizon > 1, f"[ERROR] horizon must bigger than 1"
        t_goal = horizon - 1
        if self.random_start:
            si = 0
            while si % self.chunk_size == 0:
                #! avoid sampling additional chunk
                si = random.randint(1, t_goal-1)
        else:
            si = 1
        traj_len = min(math.ceil((t_goal - si) / self.skip_frame), self.window_len)
        ei = si + (traj_len - 1) * self.skip_frame
        
        s_chunk = math.floor(si / self.chunk_size)
        e_chunk = math.floor(ei / self.chunk_size)
        
        pile_chunks = []
        for chunk_id in range(s_chunk, e_chunk + 1):
            txn = self.lmdb_traj_envs[dataset_id].begin()
            chunk_name = traj_name + "_" + str(chunk_id)
            serialized_chunk = txn.get(chunk_name.encode())
            chunk = pickle.loads(serialized_chunk)
            pile_chunks.extend(chunk)
        _si = si - s_chunk * self.chunk_size
        _ei = ei - s_chunk * self.chunk_size
        
        obs, action, reward, done, info, prev_action = [], [], [], [], [], []
        for frame_id in range(_si,_ei+1,self.skip_frame):
            f_obs, f_action, f_reward, f_done, f_info = pile_chunks[frame_id]
            f_prev_action = pile_chunks[frame_id - 1][1]
            obs.append(f_obs)
            action.append(f_action)
            reward.append(f_reward)
            done.append(f_done)
            info.append(f_info)
            prev_action.append(f_prev_action)
        
        state = {}
        state['rgb'] = np.stack([o['rgb'] for o in obs])
        state['voxels'] = np.stack([o['voxels'] for o in obs])
        state['compass'] = np.stack([o['compass'] for o in obs])
        state['gps'] = np.stack([o['gps'] for o in obs]) / np.array([[1000., 100., 1000.]])
        state['biome'] = np.stack([o['biome_id'] for o in obs])
        state['prev_action'] = np.stack(prev_action)

        action = np.stack(action)
        goal = np.repeat(self.embedding_dict[goal], traj_len, 0)
        horizon = np.arange(t_goal-si, t_goal-si-self.skip_frame*traj_len+(self.skip_frame-1),-self.skip_frame)
        horizon = discrete_horizon(horizon)
        timestep = np.arange(0, traj_len)
        
        return self.padding(goal, state, action, horizon, timestep)
        
        
    # def __getitem__(self, idx):
        
    #     if self.filters is not None:
    #         goal = random.choice(self.filters)
    #     else:
    #         goal = random.choice(list(self.trajs_by_goal.keys()))
    #     traj_id = np.random.randint(0, len(self.trajs_by_goal[goal]))
    #     traj_name = self.trajs_by_goal[goal][traj_id]
    #     traj_metadata = self.trajs_info[traj_name]
    #     dataset_id = traj_metadata["dataset_id"]

    #     num_chunks = traj_metadata["num_chunks"]
    #     horizon = traj_metadata["horizon"]
        
    #     txn = self.lmdb_traj_envs[dataset_id].begin()
    #     chunk_id = np.random.randint(0, num_chunks)
    #     chunk_name = traj_name + "_" + str(chunk_id)
    #     serialized_chunk = txn.get(chunk_name.encode())
    #     chunk = pickle.loads(serialized_chunk)
        
    #     if len(chunk) == 1:
    #         #! resample a chunk
    #         chunk_id = np.random.randint(0, num_chunks - 1)
    #         chunk_name = traj_name + "_" + str(chunk_id)
    #         serialized_chunk = txn.get(chunk_name.encode())
    #         chunk = pickle.loads(serialized_chunk)
        
    #     frame_id = np.random.randint(0, len(chunk))
    #     if chunk[frame_id][1] is None:
    #         frame_id = np.random.randint(0, len(chunk) - 1)
        
    #     obs, action, reward, done, info = chunk[frame_id]
    #     if frame_id != 0:
    #         prev_action = chunk[frame_id-1][1]
    #     elif chunk_id != 0:
    #         prev_chunk_name = traj_name + "_" + str(chunk_id-1)
    #         prev_serialized_chunk = txn.get(prev_chunk_name.encode())
    #         prev_trunk = pickle.loads(prev_serialized_chunk)
    #         prev_action = prev_trunk[-1][1]
    #     else:
    #         prev_action = 0 * action
        
    #     state = {}
    #     state['rgb'] = obs['rgb'].reshape(1, *obs['rgb'].shape)
    #     state['voxels'] = obs['voxels'].reshape(1, *obs['voxels'].shape)
    #     state['compass'] = obs['compass'].reshape(1, *obs['compass'].shape)
    #     state['gps'] = obs['gps'].reshape(1, *obs['gps'].shape) / np.array([[1000., 100., 1000.]])
    #     state['biome'] = obs['biome_id'].reshape(1, *obs['biome_id'].shape)
        
        
    #     action = action.reshape(1, *action.shape)
    #     prev_action = prev_action.reshape(1, *prev_action.shape)
    #     # goal = np.array([self.goal_mapping[goal]])
    #     goal = self.embedding_dict[goal]
    #     horizon_to_goal = horizon - (8 * chunk_id + frame_id)
    #     horizon = np.array([horizon_to_goal])
    #     horizon = discrete_horizon(horizon)
    #     timestep = np.array([0])
    #     mask = np.array([1.])
        
    #     state['rgb'] = torch.from_numpy(state['rgb']).float()
    #     state['voxels'] = torch.from_numpy(state['voxels']).long()
    #     state['compass'] = torch.from_numpy(state['compass']).float()
    #     state['gps'] = torch.from_numpy(state['gps']).float()
    #     state['biome'] = torch.from_numpy(state['biome']).long()
    #     action = torch.from_numpy(action).float()
    #     prev_action = torch.from_numpy(prev_action).float()
    #     goal = torch.from_numpy(goal)
    #     horizon = torch.from_numpy(horizon).long()
    #     timestep = torch.from_numpy(timestep).long()
    #     mask = torch.from_numpy(mask).long()
        
    #     return state, action, prev_action, goal, horizon, timestep, mask

if __name__ == "__main__":
    
    bound = 2000
    embedding_dict = {
        'log': np.random.rand(1, 512), 
        'sheep': np.random.rand(1, 512),
    }
    nb_eps = 40
    skip_frame = 1
    window_len = 600
    batch_size = 128 // window_len
    print(f"skip_frame: {skip_frame}, window_len: {window_len}, batch_size: {batch_size}")
    dataset = LMDBTrajectoryDataset(
        in_dir="/scratch/public/minecraft-data/plains_lmdb", 
        aug_ratio = bound,
        embedding_dict=embedding_dict,
        skip_frame=skip_frame,
        window_len=window_len,
        random_start=True)
    
    for i in tqdm(range(nb_eps)):
        eps = dataset[i]
        import ipdb; ipdb.set_trace()
        imgs = []
        for id, frame in enumerate(eps[1]['rgb']):
            if frame.sum() <= 0:
                continue
            # import ipdb; ipdb.set_trace()
            frame = frame.numpy().astype('uint8')
            horizon_text = eps[3][id].item()
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
        # imgs = [Image.fromarray(frame.numpy().astype(np.uint8)) for frame in eps[1]['rgb'] if frame.sum() > 0]
        imgs[0].save(f'gifs/{i}.gif', save_all=True, append_images=imgs[1:], optimize=False, quality=0, duration=10, loop=0)
        
    
    
    # from torch.utils.data.dataloader import DataLoader
    # dataloader = DataLoader(
    #         dataset, 
    #         shuffle=False, 
    #         pin_memory=True, 
    #         batch_size=batch_size,
    #         num_workers=8,
    #     )
    
    # for _ in range(1):
    #     for data in tqdm(dataloader):
    #         goal, state, action, horizon, timestep, mask = data
    #         import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()