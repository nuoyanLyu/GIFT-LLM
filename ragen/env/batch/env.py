from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import nashpy as nash  
import random
import gymnasium as gym
from .config import BatchConfig
from ragen.env.base import BaseLanguageBasedEnv, seed_everything, timed, TRAIN_STEPS
from ragen.env.env_factory import create_success_info, clean_success_info

class BatchEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: Optional[BatchConfig] = None):
        BaseLanguageBasedEnv.__init__(self) 
        self.config = config or BatchConfig()
        self.seed = int(self.config.seed)
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'
        base_envs = self.config.base_env_list
        self.env_names = base_envs
        self.k_list = self.config.k_list

        assert len(self.k_list) == len(self.env_names)
        
        mode = self.config.mode
        # 初始化训练阶段
        self.phase = None
        # 初始化嵌套的子env
        self.sub_envs = []
        self.env_success_info = []

        # 延迟导入，避免循环依赖
        from ragen.env import REGISTERED_ENV_CONFIGS, REGISTERED_ENVS
        
        for i in range(len(base_envs)):
            env_name = base_envs[i]
            Config = REGISTERED_ENV_CONFIGS[env_name]
            if hasattr(Config, 'mode'):
                env_config = Config(mode=mode)
            elif self.config.player_nums[i]:
                player_info = self.config.player_infos[i]
                env_config = Config(player_info=player_info)
            else:
                env_config = Config()
            env = REGISTERED_ENVS[env_name](config=env_config)
            self.sub_envs.append(env)
            self.env_success_info.append(create_success_info(env_name, max_num=self.config.max_num))
        # self.reset(self.seed)
    
    def reset(self, seed, **kwargs):
        seed_everything(seed)
        for env in self.sub_envs:
            env.reset(seed, **kwargs)
        # 计算任务难度——方便后续任务reward分配
        # self.diff_list = [1 - success.get() for success in self.env_success_info]
        # 选取子任务，改为rollout阶段的时候要读取子任务、分为了不同的batch
        k0 = (sum(TRAIN_STEPS) - 1) % sum(self.k_list) + 1
        # 初始化任务阶段
        for i in range(len(self.k_list)):
            if k0 <= sum(self.k_list[:i + 1]):
                self.phase = i
                break
        # 初始化子任务成功率
        self.sub_rewards = []
        self.sub_success = []
        # 成功率不清零从而记录之前的历史信息

    def render(self) -> str:
        env = self.sub_envs[self.phase]
        return env.render()

    def step(self, action):
        # 找到当前step对应环境并调用对应的step函数
        # print(action)
        env = self.sub_envs[self.phase]
        prompt, reward, done, info = env.step(action)
        # 子任务未结束、继续子任务
        if not done:
            return prompt, reward, done, info
        
        # 记录子任务成功率
        self.env_success_info[self.phase].record(info['success'])
        self.env_success_info[self.phase].update()
        # 计算总体的reward以及info等信息
        # 这个地方的设计可能不靠谱——都对应该设置为1，只有一个对才应该是一半的奖励
        # 求平均值对两个都作对的奖励可能不够？整体reward都偏小了
        # print(self.sub_rewards)
        # reward = sum(self.sub_rewards) / len(self.sub_rewards)
        prompt = self.render()
        done = True
        success = info['success']

        info = dict(
            action_is_valid=True,
            action_is_effective=True,
            success=success,
        )
        # 记录子任务成功情况，方便查看模型训练情况
        info[f'{self.env_names[self.phase]}_success'] = success
        # 记录任务成功率方便查看模型情况——直接用difficult打印避免混乱
        for i in range(len(self.env_names)):
            info[f'{self.env_names[i]}_success_rate'] = self.env_success_info[i].get()
        return prompt, reward, done, info

    def close(self):
        for env in self.sub_envs:
            env.close()
        self.sub_envs = None
        for env_name in self.env_names:
            clean_success_info(env_name)


    
if __name__ == '__main__':
    config = BatchConfig()
    env = BatchEnv(config)
    r = 1
    while r:
        env.reset(random.randint(0, 1000))
        done = False
        while not done:
            print(env.render())
            print('game', r)
            action = input('Input action: ')
            if action == 'q':
                r = -1
                break
            prompt, reward, done, info = env.step(action)
            print(prompt, reward, done, info)
        r += 1
        