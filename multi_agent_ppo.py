import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import multi_agent_Snake
import numpy as np
import random

import ray
from ray.rllib.env.vector_env import VectorEnv
from ray.tune.registry import register_env

from gym.wrappers.monitoring.video_recorder import VideoRecorder

import os


import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


'''
   Note :Suffix _ld means list of dicts , _d is dict ,_l is list of tensors for all variables used below
'''

def CNN(output_dim):
	model =  nn.Sequential(
		nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,padding='valid'),
		nn.LeakyReLU(inplace=True),
		nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding='valid'),
		nn.LeakyReLU(inplace=True),
		nn.Flatten(),
		nn.Linear(16*16*16,32),
		nn.ReLU(inplace=True),
		nn.Linear(32,16),
		nn.ReLU(inplace=True),
		nn.Linear(16,output_dim)
	)
	return model

class MLPActorCrtic(nn.Module):
	def __init__ (self,act_dim,num_envs):
		super().__init__()
		self.pi_logits_net = CNN(act_dim)
		self.v_logits_net =  CNN(1)

	def step(self,obs,a=None,grad_condition=False):
		with torch.set_grad_enabled(grad_condition):
			pi_logits = self.pi_logits_net(obs)
			pi = Categorical(logits = pi_logits)
			if a == None:
				a = pi.sample()
				logp_a = pi.log_prob(a)
			else:
				logp_a = pi.log_prob(a)	
			v_logits = self.v_logits_net(obs)
			v = torch.squeeze(v_logits,-1)
		return a,v,logp_a,pi.entropy()


class PPO: 
	def __init__(self):
		self.num_envs  = 32
		self.num_updates = 32000
		self.num_timesteps = 32 
		self.gamma = 0.99
		self.lamda = 0.95
		self.mini_batch_size = 4 
		self.learning_rate = 2.5e-4
		self.clip_coef = 0.2
		self.entropy_coef=0.01
		self.value_coef= 0.5
		self.max_grad_norm =0.5
		self.update_epochs=4

		self.snakeids = ["snake1","snake2"]
		self.update = 0 
		self.episodeid = 0
		self.total_reward_d = {"snake1":0 ,"snake2":0} 

		self.snake_dead_d = {"snake1":False ,"snake2":False} 



	def capped_cubic_video_schedule(self,episode_id: int) -> bool:
		if self.update  < self.num_updates-3: 
			if episode_id < 1000:
				return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
			else:
				return episode_id % 1000 == 0
		else:
			return episode_id % 1 == 0  #For last 3 updates, video for every 100th episode will be recorded


	def calculate_gae(self,last_values,last_dones,id):
		next_nonterminal = None
		last_gae = 0 

		last_index = self.num_envs-1

		for step in reversed(range(self.batch_size)):
			if step >= self.batch_size-self.num_envs and step <= self.batch_size-1: 
				next_nonterminal = 1.0-last_dones[last_index]
				next_values = last_values[last_index]
				last_index = last_index-1
			else:
				next_nonterminal = 1.0-self.batch_dones[id][step+1]
				next_values = 1.0-self.batch_values[id][step+1]
	
			delta = self.batch_rewards[id][step]+self.gamma*next_nonterminal*next_values-self.batch_values[id][step] 

			self.batch_advantages[id][step] = last_gae = delta +self.gamma*next_nonterminal*self.lamda*last_gae
			
		self.batch_returns[id] = self.batch_advantages[id]+self.batch_values[id] 	



	def step(self):
		#print("Step function enter:")
		
		#Below are list of tensors which will be converted to tensors each having size = batch_size after collecting 
		#data
		list_actions = {"snake1":[],"snake2":[]}
		list_values = {"snake1":[],"snake2":[]}
		list_logprobs_ac = {"snake1":[],"snake2":[]}
		list_entropies_agent = {"snake1":[],"snake2":[]}

		list_obs = {"snake1":[],"snake2":[]}
		list_rewards = {"snake1":[],"snake2":[]}
		list_dones = {"snake1":[],"snake2":[]}

		list_values = {"snake1":[],"snake2":[]}
		list_dones = {"snake1":[],"snake2":[]}


		for i in range(0,self.num_timesteps):
			actions_d = {} 
			actions_ld = []

			values_d = {}
			logprobs_ac_d = {}
			entropies_agent_d = {}

			#print("----------------------------------TIMESTEP NO:%d---------------------------------------"%i)	
		
			for li in range(0,self.num_envs):
				for id in self.snakeids:
					list_obs[id].append(torch.from_numpy(self.next_obs_ld[li][id]).type(torch.float32))
					list_dones[id].append(torch.tensor(self.next_dones_ld[li][id],dtype=torch.bool))


					actions_d[id],values_d[id],logprobs_ac_d[id],entropies_agent_d[id] = self.actor_critic_d[id].step(
						torch.as_tensor(self.next_obs_ld[li][id].reshape(1,1,self.next_obs_ld[li][id].shape[0],self.next_obs_ld[li][id].shape[1]),dtype = torch.float32))

					list_actions[id].append(actions_d[id])
					list_values[id].append(values_d[id])
					list_logprobs_ac[id].append(logprobs_ac_d[id])
					list_entropies_agent[id].append(entropies_agent_d[id])


					actions_d[id] = actions_d[id].numpy()[0]


				actions_ld.append(actions_d)

			#print(actions_ld)		
			self.next_obs_ld,rewards_ld,self.next_dones_ld,infos_ld = self.snake_game_envs.vector_step(actions_ld)

			if(self.capped_cubic_video_schedule(self.episodeid)):
				self.video_recorder.capture_frame()

	
			for li in range(0,self.num_envs):
				for id in self.snakeids:
					if li==0:
		
						if(self.snake_dead_d[id] != True):
							self.total_reward_d[id] += rewards_ld[li][id]


						self.snake_dead_d[id] = self.next_dones_ld[li][id]
						
						#print("reward for agent:",id)
						#print(rewards_ld[li][id])

						#print("total reward:",id)
						#print(self.total_reward_d[id])

					list_rewards[id].append(torch.tensor(rewards_ld[li][id],dtype=torch.float32))

				if(self.next_dones_ld[li]["snake1"] and self.next_dones_ld[li]["snake2"]):
			
					#Reset the sub environment if both snake1 and snake2 die 		
					if li == 0:
						filestr = "\n"+"Episode_"+str(self.episodeid)+":"+"Snake1"+":"+str(self.total_reward_d["snake1"])+","+"Snake2"+":"+str(self.total_reward_d["snake2"])
						#print(filestr)
						self.trainf.write(filestr)

						if(self.capped_cubic_video_schedule(self.episodeid)):
							self.video_recorder.close()

						self.episodeid += 1
						#print("------------------------------New episode started id:-----------------",self.episodeid)
	
						if(self.capped_cubic_video_schedule(self.episodeid)):
							self.video_path = os.path.abspath(os.getcwd())+"/video/"+"Episode_"+str(self.episodeid)+".mp4"
							self.video_recorder = VideoRecorder(self.first_env,self.video_path)
	
						self.total_reward_d["snake1"] = 0
						self.total_reward_d["snake2"] = 0


					obs = self.snake_game_envs.reset_at(li)
					self.next_obs_ld[li] = obs


		self.batch_size	= self.num_timesteps*self.num_envs	#will be same for 2 snake agents


		list_last_values = {"snake1":[],"snake2":[]}
		list_last_dones =  {"snake1":[],"snake2":[]}

		for li in range(0,self.num_envs):
			next_values = {}
			for id in self.snakeids:		
				_,next_values[id],_,_ = self.actor_critic_d[id].step(torch.as_tensor(
					self.next_obs_ld[li][id].reshape(1,1,self.next_obs_ld[li][id].shape[0],self.next_obs_ld[li][id].shape[1]),dtype = torch.float32))
				list_last_values[id].append(next_values[id])
				list_last_dones[id].append(self.next_dones_ld[li][id])


		self.batch_actions = {}
		self.batch_values = {}
		self.batch_logprobs_ac = {}
		self.batch_entropies_agent = {}
		self.batch_obs ={}
		self.batch_rewards ={}
		self.batch_dones = {}	
		self.batch_advantages ={}
		self.batch_returns={}	

		for id in self.snakeids:		
			self.batch_actions[id] = torch.Tensor(self.batch_size)
			torch.cat(list_actions[id], out=self.batch_actions[id])


			self.batch_values[id] = torch.Tensor(self.batch_size)
			torch.cat(list_values[id], out=self.batch_values[id])


			self.batch_logprobs_ac[id] = torch.Tensor(self.batch_size)
			torch.cat(list_logprobs_ac[id], out=self.batch_logprobs_ac[id])


			self.batch_entropies_agent[id] = torch.Tensor(self.batch_size)
			torch.cat(list_entropies_agent[id], out=self.batch_entropies_agent[id])


			self.batch_obs[id] = torch.Tensor(self.batch_size,20,20)
			torch.cat(list_obs[id], out = self.batch_obs[id])


			self.batch_rewards[id] = torch.Tensor(list_rewards[id])

			self.batch_dones[id] = torch.Tensor(list_dones[id])
			
		
			self.batch_advantages[id] = torch.zeros_like(self.batch_values[id])
			self.batch_returns[id] = torch.zeros_like(self.batch_values[id])

			self.calculate_gae(list_last_values[id],list_last_dones[id],id)	

		#print("Step function exit:")

	

	def train(self):

		seed = 0
		torch.manual_seed(seed)
		np.random.seed(seed)
		
		self.trainf = open('TrainLog.txt','a')

		ray.init()

		config = {
			"env": multi_agent_Snake.MultiAgentSnakeGameEnv,
			"num_workers": self.num_envs,
			"num_envs_per_worker": 1,
			"remote_worker_envs": False,
			"framework": "torch"
		}	

		def env_creator(index):
			return multi_agent_Snake.MultiAgentSnakeGameEnv(config)


		register_env("MultiAgentSnakeGameEnv", lambda config: env_creator)

		obj = multi_agent_Snake.MultiAgentSnakeGameEnv(config)

		self.snake_game_envs = VectorEnv.vectorize_gym_envs(
			env_creator,
			num_envs = self.num_envs,
			action_space= obj.action_space,
			observation_space= obj.observation_space,
			restart_failed_sub_environments = False
		)



		self.first_env = self.snake_game_envs.get_sub_environments()[0]
	
		self.video_path = os.path.abspath(os.getcwd())+"/video/"+"Episode_"+str(self.episodeid)+".mp4"

		self.video_recorder = VideoRecorder(self.first_env,self.video_path)
		

		print("single_observation_space.shape=")
		print(self.snake_game_envs.observation_space.shape)

		print("number_of_actions=")
		print(self.snake_game_envs.action_space.n)


		self.actor_critic_d = {}
		self.actor_critic_optimizer_d = {}


		for id in self.snakeids:
			self.actor_critic_d[id] = MLPActorCrtic(self.snake_game_envs.action_space.n,self.num_envs)
			

		self.next_dones_ld = [{"snake1":False ,"snake2":False} for i in range(self.num_envs)]

		self.next_obs_ld = self.snake_game_envs.vector_reset()
		

		for update in range(1,self.num_updates+1):
			self.update = update
			print("************Multi_Agent_PPO*****UpdateNum*******:",update)

			multiple = 1.0-(update-1.0)/self.num_updates
			self.lr_current = multiple*self.learning_rate
			
			self.step() #step the environment and actor critic to get one batch of data
			

			self.batch_size = len(self.batch_actions)
			
			self.batch_indices = [i for i in range(0,self.batch_size)]
			
			random.shuffle(self.batch_indices)

			torch.set_num_threads(1)

			if __name__ == "__main__":
				size = 32
				processes = []
				for rank in range(size):
					p = mp.Process(target= init_process, args=(rank,size,compute_gradients_and_optimize))
					p.start()
					processes.append(p)

				for p in processes:
					p.join()
		
		self.video_recorder.close()	
		self.trainf.close()


def init_process(rank, size, fn, backend = "gloo"):
		""" Initialize the distributed environment. """
		os.environ['MASTER_ADDR'] = '127.0.0.1'
		os.environ['MASTER_PORT'] = '29503'
		dist.init_process_group(backend, rank=rank, world_size=size)
		fn(rank,size)


def compute_gradients_and_optimize(rank,size):
	ddp_models_d = {}
	optimizers_d = {}

	for id in ppo_obj.snakeids:
		ddp_models_d[id] = DDP(ppo_obj.actor_critic_d[id])
		optimizers_d[id] = Adam(ddp_models_d[id].parameters(),lr = ppo_obj.lr_current)
	
	#print("***********************************Enter compute_gradients_and_optimize***********************")
	
	#Below 3 params are same for snake 1 and snake 2
	sub_batch_size =  ppo_obj.batch_size//size
	sub_batch_train_start_index = rank*sub_batch_size 
	sub_batch_train_stop_index = sub_batch_train_start_index+sub_batch_size

	for epoch in range(ppo_obj.update_epochs):
		i = sub_batch_train_start_index 

		while (i < sub_batch_train_stop_index):

			start = i
			end = i+ ppo_obj.mini_batch_size
			slice = ppo_obj.batch_indices[start:end]

			for id in ppo_obj.snakeids:	

				mini_batch_obs = ppo_obj.batch_obs[id][slice]

				mini_batch_actions = ppo_obj.batch_actions[id][slice]

				mini_batch_logp_a = ppo_obj.batch_logprobs_ac[id][slice]

				mini_batch_returns = ppo_obj.batch_returns[id][slice]

				mini_batch_values = ppo_obj.batch_values[id][slice]
					
				mb_obs_size = list(mini_batch_obs.size())	

				_,new_v,new_logp_a,entropy = ppo_obj.actor_critic_d[id].step(
					mini_batch_obs.reshape(mb_obs_size[0],1,mb_obs_size[1],mb_obs_size[2]),mini_batch_actions,grad_condition=True)


				mini_batch_advantages = ppo_obj.batch_advantages[id][slice]
				mini_batch_advantages_mean = mini_batch_advantages.mean()
				mini_batch_advantages_std = mini_batch_advantages.std()
				mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages_mean)/(mini_batch_advantages_std + 1e-8)


				logratio = new_logp_a-mini_batch_logp_a

				ratio = logratio.exp()
				
				ploss1 = -mini_batch_advantages*ratio
				ploss2 = -mini_batch_advantages* torch.clamp(ratio, 1 - ppo_obj.clip_coef, 1 + ppo_obj.clip_coef) 
				ploss = torch.max(ploss1,ploss2).mean()

				vloss1 = (new_v-mini_batch_returns)**2
				vloss2 = (torch.clamp(new_v-mini_batch_values,-ppo_obj.clip_coef, ppo_obj.clip_coef)-mini_batch_returns)**2
				vloss = 0.5*torch.max(vloss1,vloss2).mean()

				entropy_loss = entropy.mean()


				loss = ploss - ppo_obj.entropy_coef*entropy_loss + ppo_obj.value_coef*vloss


				ppo_obj.actor_critic_optimizer_d[id].zero_grad()
				loss.backward()


				nn.utils.clip_grad_norm_(ppo_obj.actor_critic[id].parameters(), ppo_obj.max_grad_norm)
				ppo_obj.actor_critic_optimizer_d[id].step()


			i = i+ppo_obj.mini_batch_size	


	
ppo_obj = PPO() 
ppo_obj.train()


