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

def layer_init(m,std=np.sqrt(2)):
	#print("within init_weights_and_biases")
	nn.init.orthogonal_(m.weight,std)
	nn.init.constant_(m.bias.data,0)
	return m

class MLPActorCrtic(nn.Module):
	def __init__ (self,act_dim):
		super().__init__()	
		self.network = nn.Sequential(
			layer_init(nn.Conv2d(3, 32, 8, stride=4)),
			nn.ReLU(),
			layer_init(nn.Conv2d(32, 64, 4, stride=2)),
			nn.ReLU(),
			layer_init(nn.Conv2d(64, 64, 3, stride=1)),
			nn.ReLU(),
			nn.Flatten(),
			layer_init(nn.Linear(46*46*64, 512)),
			nn.ReLU(),
		)
		self.policy = layer_init(nn.Linear(512, 4), std=0.01)
		self.value =  layer_init(nn.Linear(512, 1), std=1)


	def step(self,obs,a=None,grad_condition=False):
		with torch.set_grad_enabled(grad_condition):
			pi_logits = self.policy(self.network(obs))
			pi = Categorical(logits = pi_logits)
			if a == None:
				a = pi.sample()
				logp_a = pi.log_prob(a)
			else:
				logp_a = pi.log_prob(a)
			v_logits = self.value(self.network(obs))
			v = torch.squeeze(v_logits,-1)
		return a,v,logp_a,pi.entropy()

class PPO: 
	def __init__(self):
		self.num_envs  = 4
		self.num_updates = 4000
		self.num_timesteps = 128
		self.gamma = 0.99
		self.lamda = 0.95
		self.mini_batch_size = 4 
		self.learning_rate = 2.5e-4
		self.clip_coef = 0.2
		self.entropy_coef=0.01
		self.value_coef= 0.5
		self.max_grad_norm =0.5
		self.epochs=4

		self.snakeids = ["snake1","snake2"]
		self.update = 0 
		self.episodeid = 0
		self.total_reward_d = {"snake1":0 ,"snake2":0} 

		self.snake_dead_d = {"snake1":False ,"snake2":False} 

		self.episode_len = 0 


	def capped_cubic_video_schedule(self,episode_id: int) -> bool:
		if self.update  < self.num_updates-30: 
			if episode_id < 1000:
				return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
			else:
				return episode_id % 1000 == 0
		else:
			return episode_id % 1 == 0  #For last 3 updates, video for every episode will be recorded


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
		
					list_obs[id].append(torch.from_numpy(self.next_obs_ld[li][id]).type(torch.float32).reshape(1,self.next_obs_ld[li][id].shape[0],self.next_obs_ld[li][id].shape[1],self.next_obs_ld[li][id].shape[2]).to(self.device))

					list_dones[id].append(torch.tensor(self.next_dones_ld[li][id],dtype=torch.bool).to(self.device))

					#print(self.next_obs_ld[li][id].shape)

					actions_d[id],values_d[id],logprobs_ac_d[id],entropies_agent_d[id] = self.actor_critic_d[id].step(torch.as_tensor(self.next_obs_ld[li][id].reshape(1,self.next_obs_ld[li][id].shape[2],self.next_obs_ld[li][id].shape[0],self.next_obs_ld[li][id].shape[1]),dtype = torch.float32).to(self.device))

					list_actions[id].append(actions_d[id])
					list_values[id].append(values_d[id])
					list_logprobs_ac[id].append(logprobs_ac_d[id])
					list_entropies_agent[id].append(entropies_agent_d[id])


					actions_d[id] = actions_d[id].cpu().numpy()[0]


				actions_ld.append(actions_d)

			#print(actions_ld)		
			self.next_obs_ld,rewards_ld,self.next_dones_ld,infos_ld = self.snake_game_envs.vector_step(actions_ld)


			#Video Recorder capture only for 1st env 
			if(self.capped_cubic_video_schedule(self.episodeid)):
				self.video_recorder.capture_frame()

	
			for li in range(0,self.num_envs):
				if li ==0:
					self.episode_len+=1

				for id in self.snakeids:
					if li==0:
		
						if(self.snake_dead_d[id] != True):
							self.total_reward_d[id] += rewards_ld[li][id]


						self.snake_dead_d[id] = self.next_dones_ld[li][id]
						
						#print("reward for agent:",id)
						#print(rewards_ld[li][id])

						#print("total reward:",id)
						#print(self.total_reward_d[id])

					list_rewards[id].append(torch.tensor(rewards_ld[li][id],dtype=torch.float32).to(self.device))

				if(self.next_dones_ld[li]["snake1"] and self.next_dones_ld[li]["snake2"]):
			
					#Reset the sub environment if both snake1 and snake2 die 		
					if li == 0:
						filestr = "\n"+"Episode_"+str(self.episodeid)+":"+"Episode_len:"+str(self.episode_len)+":"+"Snake1 rew"+":"+str(self.total_reward_d["snake1"])+","+"Snake2 rew"+":"+str(self.total_reward_d["snake2"])
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

						self.snake_dead_d["snake1"] = False
						self.snake_dead_d["snake2"] = False

						self.episode_len = 0

					obs = self.snake_game_envs.reset_at(li)
					self.next_obs_ld[li] = obs


		list_last_values = {"snake1":[],"snake2":[]}
		list_last_dones =  {"snake1":[],"snake2":[]}

		for li in range(0,self.num_envs):
			next_values = {}
			for id in self.snakeids:		
				_,next_values[id],_,_ = self.actor_critic_d[id].step(torch.as_tensor(
					self.next_obs_ld[li][id].reshape(1,self.next_obs_ld[li][id].shape[2],self.next_obs_ld[li][id].shape[0],self.next_obs_ld[li][id].shape[1]),dtype = torch.float32).to(self.device))
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


		self.batch_size = self.num_timesteps*self.num_envs	#will be same for 2 snake agents


		self.data_store_dict["batch_size"] = self.batch_size

		for id in self.snakeids:

			self.batch_actions[id] = torch.Tensor(self.batch_size).to(self.device)
			torch.cat(list_actions[id], out=self.batch_actions[id])
			self.data_store_dict[id]["batch_actions"] = self.batch_actions[id]


			self.batch_values[id] = torch.Tensor(self.batch_size).to(self.device)
			torch.cat(list_values[id], out=self.batch_values[id])
			self.data_store_dict[id]["batch_values"] = self.batch_values[id]


			self.batch_logprobs_ac[id] = torch.Tensor(self.batch_size).to(self.device)
			torch.cat(list_logprobs_ac[id], out=self.batch_logprobs_ac[id])
			self.data_store_dict[id]["batch_logprobs_ac"] = self.batch_logprobs_ac[id]


			self.batch_entropies_agent[id] = torch.Tensor(self.batch_size).to(self.device)
			torch.cat(list_entropies_agent[id], out=self.batch_entropies_agent[id])
			self.data_store_dict[id]["batch_entropies_agent"] = self.batch_entropies_agent[id]


			self.batch_obs[id] = torch.Tensor(self.batch_size,400,400,3).to(self.device)
			torch.cat(list_obs[id], out = self.batch_obs[id])
			self.data_store_dict[id]["batch_obs"]= self.batch_obs[id]



			self.batch_rewards[id] = torch.Tensor(list_rewards[id]).to(self.device)

			self.batch_dones[id] = torch.Tensor(list_dones[id]).to(self.device)
			
		
			self.batch_advantages[id] = torch.zeros_like(self.batch_values[id]).to(self.device)
			self.batch_returns[id] = torch.zeros_like(self.batch_values[id]).to(self.device)

			self.calculate_gae(list_last_values[id],list_last_dones[id],id)	

			self.data_store_dict[id]["batch_advantages"]=self.batch_advantages[id]
			self.data_store_dict[id]["batch_returns"] = self.batch_returns[id]

		#print("Step function exit:")

	

	def train(self):

		seed = 0
		torch.manual_seed(seed)
		np.random.seed(seed)
		
		self.trainf = open('TrainLog.txt','a')

		#ray.init()

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

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.actor_critic_d = {}

		for id in self.snakeids:
			self.actor_critic_d[id] = MLPActorCrtic(self.snake_game_envs.action_space.n).to(self.device)
			self.actor_critic_d[id].share_memory()

		self.data_store_dict = {"snake1":{} ,"snake2":{}}

		self.next_dones_ld = [{"snake1":False ,"snake2":False} for i in range(self.num_envs)]

		self.next_obs_ld = self.snake_game_envs.vector_reset()
		

		self.data_store_dict["epochs"] = self.epochs

		self.data_store_dict["mini_batch_size"] = self.mini_batch_size

		self.data_store_dict["max_grad_norm"] = self.max_grad_norm

		self.data_store_dict["entropy_coef"] = self.entropy_coef

		self.data_store_dict["value_coef"] = self.value_coef

		self.data_store_dict["clip_coef"] = self.clip_coef


		for update in range(1,self.num_updates+1):
			self.update = update
			print("************Multi_Agent_PPO*****UpdateNum*******:",update)

			multiple = 1.0-(update-1.0)/self.num_updates
			self.lr_current = multiple*self.learning_rate
			self.data_store_dict["lr_current"] = self.lr_current
			
			self.step() #step the environment and actor critic to get one batch of data

			self.batch_indices = [i for i in range(0,self.batch_size)]

			self.data_store_dict["batch_indices"] = self.batch_indices
			
			random.shuffle(self.batch_indices)

			torch.set_num_threads(1)

			if __name__ == "__main__":
				size = 1
				run_method(compute_gradients_and_optimize,size,self.actor_critic_d,self.data_store_dict)
		
		self.video_recorder.close()	
		self.trainf.close()


def setup(rank,world_size):
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29530'
	dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run_method(fn,world_size,actor_critic_d,data_store_dict):
	mp.spawn(fn,args=(world_size,actor_critic_d,data_store_dict),nprocs = world_size,join = True)


def compute_gradients_and_optimize(rank,world_size,actor_critic_d,data_store_dict):
	setup(rank, world_size)
	ddp_models_d = {}
	optimizers_d = {}
	snakeids = ["snake1","snake2"] 

	for id in snakeids:
		actor_critic_d[id] = actor_critic_d[id].to(rank)
		ddp_models_d[id] = DDP(actor_critic_d[id],device_ids= [rank])
		#ddp_models_d[id] = DDP(actor_critic_d[id])
		optimizers_d[id] = Adam(ddp_models_d[id].parameters(),lr = data_store_dict["lr_current"])
	
	#print("***********************************Enter compute_gradients_and_optimize***********************")
	
	#Below 3 params are same for snake 1 and snake 2
	sub_batch_size = data_store_dict["batch_size"] //world_size
	sub_batch_train_start_index = rank*sub_batch_size 
	sub_batch_train_stop_index = sub_batch_train_start_index+sub_batch_size

	epochs = data_store_dict["epochs"]
	mini_batch_size = data_store_dict["mini_batch_size"]

	for epoch in range(epochs):
		i = sub_batch_train_start_index 

		while (i < sub_batch_train_stop_index):

			start = i
			end = i+ mini_batch_size
			slice = data_store_dict["batch_indices"][start:end]

			for id in snakeids:	

				#mini_batch_obs = data_store_dict[id]["batch_obs"][slice]
				mini_batch_obs = data_store_dict[id]["batch_obs"][slice].to(rank)

				#mini_batch_actions = data_store_dict[id]["batch_actions"][slice]
				mini_batch_actions = data_store_dict[id]["batch_actions"][slice].to(rank)


				#mini_batch_logp_a = data_store_dict[id]["batch_logprobs_ac"][slice]
				mini_batch_logp_a = data_store_dict[id]["batch_logprobs_ac"][slice].to(rank)

				#mini_batch_returns = data_store_dict[id]["batch_returns"][slice]
				mini_batch_returns = data_store_dict[id]["batch_returns"][slice].to(rank)

				#mini_batch_values = data_store_dict[id]["batch_values"][slice]
				mini_batch_values = data_store_dict[id]["batch_values"][slice].to(rank)
					
				mb_obs_size = list(mini_batch_obs.shape)	
				
				#_,new_v,new_logp_a,entropy = actor_critic_d[id].step(
				#	mini_batch_obs.reshape(mb_obs_size[0],mb_obs_size[3],mb_obs_size[1],mb_obs_size[2]),mini_batch_actions,grad_condition=True)


				_,new_v,new_logp_a,entropy = actor_critic_d[id].step(mini_batch_obs.reshape(mb_obs_size[0],mb_obs_size[3],mb_obs_size[1],mb_obs_size[2]).to(rank),mini_batch_actions,grad_condition=True)


				#mini_batch_advantages = data_store_dict[id]["batch_advantages"][slice]
				mini_batch_advantages = data_store_dict[id]["batch_advantages"][slice].to(rank)

				mini_batch_advantages_mean = mini_batch_advantages.mean()
				mini_batch_advantages_std = mini_batch_advantages.std()
				mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages_mean)/(mini_batch_advantages_std + 1e-8)


				logratio = new_logp_a-mini_batch_logp_a

				ratio = logratio.exp()
				
				ploss1 = -mini_batch_advantages*ratio
				ploss2 = -mini_batch_advantages* torch.clamp(ratio, 1 - data_store_dict["clip_coef"], 1 + data_store_dict["clip_coef"]) 
				ploss = torch.max(ploss1,ploss2).mean()

				vloss1 = (new_v-mini_batch_returns)**2
				vloss2 = (torch.clamp(new_v-mini_batch_values,-data_store_dict["clip_coef"], data_store_dict["clip_coef"])-mini_batch_returns)**2
				vloss = 0.5*torch.max(vloss1,vloss2).mean()

				entropy_loss = entropy.mean()


				loss = ploss - data_store_dict["entropy_coef"]*entropy_loss + data_store_dict["value_coef"]*vloss


				optimizers_d[id].zero_grad()
				loss.backward()


				nn.utils.clip_grad_norm_(ddp_models_d[id].parameters(),data_store_dict["max_grad_norm"])
				optimizers_d[id].step()


			i = i+mini_batch_size	


if __name__ == '__main__':	
	ppo_obj = PPO() 
	ppo_obj.train()


