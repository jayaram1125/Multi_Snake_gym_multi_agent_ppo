import gym 
from gym import logger, spaces

import numpy as np
import Box2D
from enum import IntEnum
import math
import random
from getkey import getkey,keys


from typing import Optional


from Box2D import b2Vec2
    
from Box2D.b2 import(
	fixtureDef,
	polygonShape)

'''import ray
from ray.rllib.env.vector_env import VectorEnv
from ray.tune.registry import register_env
from gym.wrappers.monitoring.video_recorder import VideoRecorder'''

import pygame
import os


BLACK = (0,0,0) #background 

DARK_GREEN = (0,100,0)#Snake1 head color
GREEN = (0, 255, 0) #Snake1 body color

DARK_BLUE = (39,64,139) #Snake2 head color
BLUE = (135,206,250) #Snake2 body color 

RED = (255, 0, 0) #Fruit color
BROWN = (156,102,31) #Grid color

DEG2RAD = math.pi/180.0
RAD2DEG = 180/math.pi

COLOR_CODE_BLACK = 65535

COLOR_CODE_DARK_GREEN = ((0<<16)|(100<<8)|0)
COLOR_CODE_GREEN =  ((0<<16)|(255<<8)|0)

COLOR_CODE_DARK_BLUE = ((39<< 16) |(64<< 8) | 139)
COLOR_CODE_BLUE = ((135<< 16) |(206<< 8) | 250)

COLOR_CODE_RED = ((255<<16)|(0<<8)|0)
COLOR_CODE_BROWN = ((156<<16)|(102<<8)|31)

COLOR_CODE_BLUE = ((135<<16)|(206<<8)|250)
COLOR_CODE_DARK_BLUE = ((39<<16)|(64<<8)|139)


MIN_OBS_V = COLOR_CODE_DARK_GREEN
MAX_OBS_V = COLOR_CODE_RED 


class MultiAgentScenario(IntEnum):
	COOPERATIVE = 0, 
	COMPETITIVE = 1


class SnakeDirection(IntEnum):
	LEFT =0,
	RIGHT =1,
	UP = 2,
	DOWN = 3



class MultiAgentSnakeGameEnv(gym.Env):	
	"""
	**********Action Space:***************
	The action is a `ndarray` with shape `(1,)` which can take values `{0,1,2,3}` indicating the direction
	of the head of the Snake to be steered.
	 Num    Action                                      Value
	  0      Steer Snake head to the left direction       0
	  1      Steer Snake head to the right                1  
	  2      Steer Snake head to the Upward direction     2 
	  3      Steer Snake head to the Downward direction   3		


	**********Observation Space:**********
	A vector of length 402 values . The first 400 values represent object values in the 20x20 grid 

     Num       Observation                           Observation Value 
      
      0        Empty location in grid                  COLOR_CODE_BLACK         
      
      1        Maze           						   COLOR_CODE_BROWN
      
      2        Fruit                                   COLOR_CODE_RED       

      3        Body of Snake1                          COLOR_CODE_GREEN

      4  	   Head of Snake1                          COLOR_CODE_DARK_GREEN

      5.       Body of Snake2                          COLOR_CODE_BLUE

      6        Head of Snake2                          COLOR_CODE_DARK_BLUE


	  Note: 
	  1.Snake's body length can range from 0 to 383 units.
	  2.The object values have to be normalized to have zero mean , unit variance  and be in the interval [-10,10]
	    before passing into the neural network


     ******Rewards:**********************
	  +10 reward if snake eats fruit and grows its body unit 
	  
	  if dcurrent is not = -1
	  (dprev -dcurrent)  reward if snake is alive and away from fruit
		dcurrent = current shortest path length from snake head to fruit
		dprev = previous episode step shortest path length from snake head to fruit
	  
	  -1 if  dcurrent is -1 (snake blocked it self ,it could unblock or might not)


      -100 reward if snake dies 


	  In addition to the above, if the environment is competitive , one snake agent is negatively rewarded for other snake agent's good reward
	  and vice versa
		
	  +100	reward to other snake if the snake dies by touching the body of the other snake .
	   Other snake positioned itself competitively to kill Snake

  	  -10 reward if other snake eats fruit and grows its body unit  




	 ******Episode Termination:**********
     The episode terminates  if:
	    1) Termination:The snake head crosses the maze (unsuccessful end of the game)
	    2) Termination:The snake head crosses any one of the body units ,another snake's head or body (unsuccessful end of the game)
	    3) Termination: Snake fills the entire play area with its head and body and no place to position the fruit. It touches its body (successful end of the game)	
	"""

	

	def __init__(self,config,render_mode: Optional[str] = "rgb_array"):

		self.metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps":2}

		self.config = config

		#print("render_mode=",render_mode)
		assert render_mode is None or render_mode in self.metadata["render_modes"]

		self.render_mode = render_mode

		self.displayX = 400.0
		self.displayY = 400.0

		self.window =None
		if self.render_mode == "human":
			#print("___ENTER___")
			import pygame

			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((self.displayX, self.displayY))
			self.clock = pygame.time.Clock()



		self.body_width = 20.0
		self.scale = 10.0
		self.world = Box2D.b2World(gravity=(0,0))



		self.screen = None
		self.action_space = spaces.Discrete(4)

		rows = int(self.displayX/self.body_width)
		cols = int(self.displayY/self.body_width)

		low = np.full((rows, cols),MIN_OBS_V)
		high = np.full((rows,cols),MAX_OBS_V)


		self.observation_space = spaces.Box(low, high)   

		self.create_maze()
		
		self.fruit = None
		self.scenario = MultiAgentScenario.COOPERATIVE
		self.multisnakeobj = None

		self.snakeids = ["snake1","snake2"]
		self.head_obs_values = {"snake1": COLOR_CODE_DARK_GREEN,"snake2": COLOR_CODE_DARK_BLUE}
		self.body_obs_values = {"snake1": COLOR_CODE_GREEN,"snake2": COLOR_CODE_DARK_GREEN}


		self.obs_n = {}
		self.is_snake_dead_n = {"snake1":False,"snake2":False}
		self.fruit_distances = {}
		self.snake_collided_with_other_snake = {"snake1":False,"snake2":False}

		self.snake_head_colors = {"snake1":DARK_GREEN ,"snake2": DARK_BLUE}
		self.snake_body_colors = {"snake1":GREEN ,"snake2":BLUE}

		self.dcurrent = {}
		self.fruit_eaten = {"snake1":False ,"snake2":False}



	def create_maze(self):
		vertices = [b2Vec2(0.0,0.0), b2Vec2(self.displayX/self.scale,0.0), b2Vec2(self.displayX/self.scale,self.displayY/self.scale),b2Vec2(0.0,self.displayY/self.scale),
				b2Vec2(0.0,0.0), b2Vec2(self.body_width/self.scale,self.body_width/self.scale),b2Vec2(self.displayX/self.scale-self.body_width/self.scale,self.body_width/self.scale),b2Vec2(self.displayX/self.scale-self.body_width/self.scale,self.displayY/self.scale-self.body_width/self.scale),b2Vec2(self.body_width/self.scale,self.displayY/self.scale-self.body_width/self.scale),b2Vec2(self.body_width/self.scale,self.body_width/self.scale)]
		chain = Box2D.b2ChainShape(vertices_chain=vertices)

		
		#displayX == displayY assumed in the design .All statements below work only if this assumption is considered		
		self.maze_collision_bound_1 = self.body_width/self.scale/2
		self.maze_collision_bound_2 = self.displayX/self.scale-self.body_width/self.scale/2
		self.maze = self.world.CreateStaticBody(fixtures=fixtureDef(shape=chain))

		#Maze indices in the observations 2D array
		self.maze_indices = []

		start = 0
		end = int(self.displayX/self.body_width)
		
		#top side of maze
		for j in range(start,end):
			self.maze_indices.append([start,j])


		#left side of the maze
		begin = start+1
		for i in range(begin, end):
			self.maze_indices.append([i,start])


		#bottom side of maze	
		for j in range(begin, end):
			self.maze_indices.append([end-1,j])


		#right side of the maze
		begin = start+1
		for i in range(begin,end-1):
			self.maze_indices.append([i,end-1])

		
		self.play_area_set = set()

		play_area_bound_1 =  int(self.body_width/self.scale + 1)
		play_area_bound_2 =  int(self.displayX/self.scale-self.body_width/self.scale -1)   
		self.cols  = play_area_bound_2//2

		self.area_pos_dict = {}

		for i in range(play_area_bound_1,play_area_bound_2+2,2):
			for j in range(play_area_bound_1,play_area_bound_2+2,2):
				#print("(%d,%d)"%(i,j))
				#i*self.cols+j helps to create unique key in the play area set
				key = int(i//2+(j//2)*self.cols)
				self.play_area_set.add(key)
				self.area_pos_dict[key] =(i,j)

				

	class MultiSnake:
		def __init__(self,env_ref,list_occupied_area_sets):
			#print("enter snake __init__")
			self.env_ref = env_ref
			self.heads = {"snake1":None,"snake2":None}
			self.bodies = {"snake1":[],"snake2":[]}
			self.head_pos_sets = {"snake1":set(),"snake2":set()} 			
			self.body_pos_sets = {"snake1":set(),"snake2":set()} 
			self.create_heads_of_snakes(list_occupied_area_sets)



		def create_heads_of_snakes(self,list_occupied_area_sets):
			for id in self.env_ref.snakeids:
				sampled_pos = self.env_ref.sample_position_from_play_area(list_occupied_area_sets)
				sampled_angle = self.env_ref.sample_angle()
			
				self.heads[id] = self.env_ref.world.CreateStaticBody(position = b2Vec2(sampled_pos[0],sampled_pos[1]),angle = sampled_angle[0] ,fixtures=fixtureDef(shape=polygonShape(box=(1,1)))) 
				self.head_pos_sets[id].add(int(self.heads[id].position[0]//2+(self.heads[id].position[1]//2)*self.env_ref.cols))			 	

				#print(self.head_pos_sets[id])

				if id == "snake1":  # after head pos is sampled for snake2 , it is not needed to add it to the list 
					list_occupied_area_sets.append(self.head_pos_sets[id])
					#print(list_occupied_area_sets)

		

		def move_snake(self,snakeid,next_direction):
			if(self.heads[snakeid] != None):
				prev_head_position =  self.heads[snakeid].position.copy()
				prev_head_angle = self.heads[snakeid].angle
		

				if next_direction == SnakeDirection.UP and round(self.heads[snakeid].angle*RAD2DEG) != 270: 
					self.heads[snakeid].position[1] = self.heads[snakeid].position[1]-self.env_ref.body_width/self.env_ref.scale
					self.heads[snakeid].angle = math.pi/2

				elif next_direction == SnakeDirection.DOWN and round(self.heads[snakeid].angle*RAD2DEG) != 90: 
					self.heads[snakeid].position[1] = self.heads[snakeid].position[1]+self.env_ref.body_width/self.env_ref.scale
					self.heads[snakeid].angle = 3*math.pi/2
						
				elif next_direction == SnakeDirection.RIGHT and round(self.heads[snakeid].angle*RAD2DEG) != 180: 				 
					self.heads[snakeid].position[0] = self.heads[snakeid].position[0]+self.env_ref.body_width/self.env_ref.scale
					self.heads[snakeid].angle = 0
				
				elif next_direction == SnakeDirection.LEFT and round(self.heads[snakeid].angle*RAD2DEG) != 0:
					self.heads[snakeid].position[0] = self.heads[snakeid].position[0]-self.env_ref.body_width/self.env_ref.scale
					self.heads[snakeid].angle = math.pi


				#print("Head angle =:"+str(round(self.head.angle*RAD2DEG)))
		

				self.head_pos_sets[snakeid].clear()
				self.head_pos_sets[snakeid].add(int(self.heads[snakeid].position[0]//2+(self.heads[snakeid].position[1]//2)*self.env_ref.cols))


				


				#print("@@@@@@@@@@@@@@@@@"+snakeid+"@@@@@@@@@@@@@@@@@@@@@@@@@@")


				#print("***************Before****************:")	
				

				#print("----HEAD---")
				#print(self.heads[snakeid].position)
	
			
				'''print("----BODIES---")
				for i in range(0,len(self.bodies[snakeid])):
					print(self.bodies[snakeid][i].position)'''


				#Update the body positions only if the head is moved .Snake cannot move in opposite direction	
				if(prev_head_position != self.heads[snakeid].position and len(self.bodies[snakeid])>0):
					
					self.body_pos_sets[snakeid].clear()

					for i in range(len(self.bodies[snakeid])- 1, 0, -1):
						self.bodies[snakeid][i].position = self.bodies[snakeid][i-1].position
						self.bodies[snakeid][i].angle = self.bodies[snakeid][i-1].angle
						self.body_pos_sets[snakeid].add(int(self.bodies[snakeid][i].position[0]//2+(self.bodies[snakeid][i].position[1]//2)*self.env_ref.cols))

					#print("prev_head_position"+str(prev_head_position))		
					#print("prev_head_angle"+str(prev_head_angle))	

					self.bodies[snakeid][0].position = prev_head_position
					self.bodies[snakeid][0].angle =  prev_head_angle
					self.body_pos_sets[snakeid].add(int(self.bodies[snakeid][0].position[0]//2+(self.bodies[snakeid][0].position[1]//2)*self.env_ref.cols))
		

				'''print("*********************After********************:")	
			
				print("----HEAD---")
				print(self.heads[snakeid].position)
			

				print("----BODIES---")
			
				for i in range(0,len(self.bodies[snakeid])):
					print(self.bodies[snakeid][i].position)


				print(self.head_pos_sets[snakeid])
				print(self.body_pos_sets[snakeid])'''	


		def destroy_snake(self,snakeid):
			if(self.heads[snakeid] != None):					
				self.env_ref.world.DestroyBody(self.heads[snakeid])
				self.heads[snakeid] = None
				self.head_pos_sets[snakeid].clear()
			for i in range(0,len(self.bodies[snakeid])):
				#print("enter destroy body:")
				self.env_ref.world.DestroyBody(self.bodies[snakeid][i])
			self.bodies[snakeid].clear()
			self.body_pos_sets[snakeid].clear()
			#print("Snake Destroyed:")


		def increase_snake_length(self,snakeid):

			new_body_unit_position_x = 0
			new_body_unit_position_y = 0

			last_unit_angle = round(self.bodies[snakeid][-1].angle*RAD2DEG) if len(self.bodies[snakeid])>0 else round(self.heads[snakeid].angle*RAD2DEG) 
			last_unit_position_x = self.bodies[snakeid][-1].position[0] if len(self.bodies[snakeid])>0 else  self.heads[snakeid].position[0]
			last_unit_position_y = self.bodies[snakeid][-1].position[1] if len(self.bodies[snakeid])>0 else  self.heads[snakeid].position[1]

			distance_delta = self.env_ref.body_width/self.env_ref.scale

			if(last_unit_angle == 0):
				new_body_unit_position_x = last_unit_position_x-distance_delta
				new_body_unit_position_y = last_unit_position_y

			elif(last_unit_angle == 90):
				new_body_unit_position_x = last_unit_position_x
				new_body_unit_position_y = last_unit_position_y+distance_delta

			elif(last_unit_angle == 180):
				new_body_unit_position_x = last_unit_position_x+distance_delta
				new_body_unit_position_y = last_unit_position_y

			elif(last_unit_angle == 270):	
				new_body_unit_position_x = last_unit_position_x
				new_body_unit_position_y = last_unit_position_y-distance_delta

			self.bodies[snakeid].append(self.env_ref.world.CreateStaticBody(position = b2Vec2(new_body_unit_position_x,new_body_unit_position_y),angle= last_unit_angle,fixtures=fixtureDef(shape=polygonShape(box=(1,1)))))
			self.body_pos_sets[snakeid].add(int(new_body_unit_position_x//2+(new_body_unit_position_y//2)*self.env_ref.cols))



	def sample_position_from_play_area(self,list_occupied_area_sets):
		remaining_area_set = self.play_area_set -list_occupied_area_sets[0]
		for i in range(1,len(list_occupied_area_sets)):
			remaining_area_set = remaining_area_set-list_occupied_area_sets[i]

		#print("remaining_area_set=")
		#print(remaining_area_set)
		
		sampled_pos = None
		if(len(remaining_area_set)!=0):
			output = random.sample(remaining_area_set,1)
			sampled_pos = self.area_pos_dict[output[0]]

		return sampled_pos	 

	def sample_angle(self):
		angles = [0.0,math.pi/2,3*math.pi/2,math.pi]
		sampled_angle = random.sample(angles,1)
		return sampled_angle	  		


	def create_fruit(self):
		sampled_pos = self.sample_position_from_play_area([set()])
		self.fruit = self.world.CreateStaticBody(position=b2Vec2(sampled_pos[0],sampled_pos[1]),angle= 0,fixtures=fixtureDef(shape=polygonShape(box=(1,1))))
	
	def destroy_fruit(self):
		self.world.DestroyBody(self.fruit)


	def move_fruit_to_another_location(self):
		list_occupied_area_sets = []
		for id in self.snakeids:
			list_occupied_area_sets.append(self.multisnakeobj.head_pos_sets[id])
			list_occupied_area_sets.append(self.multisnakeobj.body_pos_sets[id])


		sampled_pos = self.sample_position_from_play_area(list_occupied_area_sets)   
		if(sampled_pos):	
			self.fruit.position[0] = sampled_pos[0]
			self.fruit.position[1] = sampled_pos[1]		
			#print('new fruit position= (%d,%d)'%(self.fruit.position[0],self.fruit.position[1]))
		#destroy fruit object when play area is filled with snake
		else:
			self.destroy_fruit()


	def create_snake(self):
		#print("Enter Create Snake method")
		fruit_area_set = set()
		#print(int(self.fruit.position[0]//2 + (self.fruit.position[1]//2)*self.cols))
		fruit_area_set.add(int(self.fruit.position[0]//2 + (self.fruit.position[1]//2)*self.cols))
		self.multisnakeobj = self.MultiSnake(self,[fruit_area_set])
		#if(self.multisnakeobj):
			#print("Not none")

	def check_contact(self,snakeid,othersnakeid):
		if(self.multisnakeobj.heads[snakeid] != None):
			snake_collided_with_maze = (self.multisnakeobj.heads[snakeid].position[0]== self.maze_collision_bound_1 or self.multisnakeobj.heads[snakeid].position[1]==self.maze_collision_bound_1 or
				self.multisnakeobj.heads[snakeid].position[0]== self.maze_collision_bound_2 or self.multisnakeobj.heads[snakeid].position[1]== self.maze_collision_bound_2)

			snake_head_pos = int(self.multisnakeobj.heads[snakeid].position[0]//2+(self.multisnakeobj.heads[snakeid].position[1]//2)*self.cols)
			
			snake_collided_with_itself = snake_head_pos in self.multisnakeobj.body_pos_sets[snakeid] 

			if(self.multisnakeobj.heads[othersnakeid] != None): 
				self.snake_collided_with_other_snake[snakeid] = (snake_head_pos in self.multisnakeobj.head_pos_sets[othersnakeid]) or (snake_head_pos in self.multisnakeobj.body_pos_sets[othersnakeid]) 

			#Checking contact with Maze or snake itself
			if(snake_collided_with_maze or snake_collided_with_itself or self.snake_collided_with_other_snake[snakeid]):
				self.multisnakeobj.destroy_snake(snakeid)
				self.is_snake_dead_n[snakeid] = True
			#Checking contact with fruit	
			elif (self.fruit.position[0] == self.multisnakeobj.heads[snakeid].position[0] and self.fruit.position[1] == self.multisnakeobj.heads[snakeid].position[1]):
				self.multisnakeobj.increase_snake_length(snakeid)
				self.move_fruit_to_another_location()
				self.fruit_eaten[snakeid] = True
				self.dcurrent[snakeid] = 0 


	def create_observations(self):
		
		obs = np.zeros((self.observation_space.shape[0],self.observation_space.shape[0]))

		'''
			Row index and col index of the matrix are adjusted so as matrix represents the 
			image frame.  
			Graphics coordinate system of Box2d and Pygame  is not same as matrix indexing

		'''

		for i in range(0,len(self.maze_indices)):
			obs[self.maze_indices[i][0]][self.maze_indices[i][1]] = COLOR_CODE_BROWN

		if(self.fruit != None):	
			obs[int(self.fruit.position[1]//2)][int(self.fruit.position[0]//2)] =  COLOR_CODE_RED

		for id in self.snakeids:	
			for i in range(0,len(self.multisnakeobj.bodies[id])):
				if(self.multisnakeobj.bodies[id][i] != None):
					obs[int(self.multisnakeobj.bodies[id][i].position[1]//2)][int(self.multisnakeobj.bodies[id][i].position[0]//2)] = self.body_obs_values[id]


			if(self.multisnakeobj.heads[id] != None):
				obs[int(self.multisnakeobj.heads[id].position[1]//2)][int(self.multisnakeobj.heads[id].position[0]//2)] = self.head_obs_values[id]  		
		
		self.obs_n["snake1"] = obs
		self.obs_n["snake2"] = obs		
				

		return self.obs_n	


	def calculate_distance_head_and_fruit(self,obs,snakeid):
		
		headX = int(self.multisnakeobj.heads[snakeid].position[1]//2)
		headY = int(self.multisnakeobj.heads[snakeid].position[0]//2)

		source = [headX, headY,0]    

		visited = []

		for i in range(len(obs)):
			visited.append([])
			for j in range(len(obs[0])): 
				visited[i].append(False)

		delta =[[-1,0],[1,0],[0,-1],[0,1]]

		queue = []
		queue.append(source)
		visited[source[0]][source[1]] = True


		while len(queue) != 0:
			source = queue.pop(0)
 
        	#Fruit found,return its distance from snake head
			#print(obs[source[0]][source[1]])
			if (obs[source[0]][source[1]] == COLOR_CODE_RED):
				return source[2]

			for i in range(0,len(delta)):
				x = source[0]+delta[i][0]
				y = source[1]+delta[i][1]  
				if (x>=0 and y>=0 and x< len(obs) and y < len(obs[0]) and (obs[x][y] == COLOR_CODE_BLACK or obs[x][y] == COLOR_CODE_RED) and visited[x][y] == False):
					queue.append([x,y,source[2] + 1])
					visited[x][y] = True

		return -1	


	def reset(self, seed=None, return_info=False, options=None):	
		#print("--------------------------RESET CALLED -----------------")
		if(self.fruit):
			self.destroy_fruit()
			self.fruit = None

		for id in self.snakeids: 	
			if(self.multisnakeobj and self.multisnakeobj.heads[id]):
				self.multisnakeobj.destroy_snake(id)

		self.create_fruit()

		self.create_snake()

		self.fruit_eaten = {"snake1":False ,"snake2":False}

		self.obs_n = self.create_observations()

		for id in self.snakeids: 
			self.dcurrent[id] = self.calculate_distance_head_and_fruit(self.obs_n[id],id)
			self.is_snake_dead_n[id] = False
			self.snake_collided_with_other_snake[id] = False

		obs =  self.obs_n["snake1"] #snake1 and snake2 have same observation values at any time	

		obs_norm = obs.copy()
		obs_norm = (obs_norm-obs_norm.mean())/obs_norm.std() 

		return_dict ={}
		return_dict["snake1"] =obs_norm 
		return_dict["snake2"]= obs_norm  

		return return_dict 	



	def step(self,actions_n):
		
		rewards_n = {"snake1":0,"snake2":0}


		for id in self.snakeids:

			assert self.action_space.contains(actions_n[id]),"action = %d is invalid" %action	

			if(self.multisnakeobj.heads[id]):

				self.multisnakeobj.move_snake(id,actions_n[id])
				
				othersnakeid = None

				if id == "snake1":
					othersnakeid = "snake2"
				elif id == "snake2":		
					othersnakeid = "snake1"
				 

				self.check_contact(id,othersnakeid)

		
		self.obs_n = self.create_observations()


				
		for id in self.snakeids:
			
			if id == "snake1":
				othersnakeid = "snake2"
			elif id == "snake2":		
				othersnakeid = "snake1"

			if(self.is_snake_dead_n[id]):
				rewards_n[id] = -100.0
				'''if(self.scenario == MultiAgentScenario.COMPETITIVE and self.snake_collided_with_other_snake[id] 
					and self.multisnakeobj.heads[othersnakeid]):
					rewards_n[othersnakeid] += 100
					self.snake_collided_with_other_snake[id] = False'''
			elif(self.fruit_eaten[id]):
				rewards_n[id] = 10
				self.fruit_eaten[id] = False 
				'''if(self.scenario == MultiAgentScenario.COMPETITIVE and self.multisnakeobj.heads[othersnakeid]):
					rewards_n[othersnakeid] -= 10'''
			else:

				dprev = self.dcurrent[id]
				self.dcurrent[id] = self.calculate_distance_head_and_fruit(self.obs_n[id],id)	

				if(self.dcurrent[id] ==-1):
					rewards_n[id] = -1.0   #snake blocked its path to fruit, -ve reward for wrong move
				else:
					rewards_n[id] = (dprev-self.dcurrent[id])

				'''if(self.scenario == MultiAgentScenario.COMPETITIVE and self.multisnakeobj.heads[othersnakeid]):
					if(self.dcurrent[id] > self.dcurrent[othersnakeid]):
						rewards_n[id] /= 2''' 

		obs_norm = self.obs_n["snake1"].copy()

		obs_norm = (obs_norm-obs_norm.mean())/obs_norm.std() 
		
		info_n = {"snake1":None,"snake2": None}

		return_obs ={}

		return_obs["snake1"] = obs_norm

		return_obs["snake2"] = obs_norm
		

		return return_obs,rewards_n,self.is_snake_dead_n,info_n
	

	def render(self, mode: str="rgb_array"):
		#print("----RENDER method Called------------------------------------------------------")
		# create the display surface object
		# of specific dimension..e(X,Y).

		assert mode is not None

		import pygame


		surf = pygame.Surface((self.displayX, self.displayY))
		surf.fill(BLACK)
		pygame.transform.scale(surf, (self.scale, self.scale))
		

		#In pygame positions are passed as (X,Y) coordinate. (Box2D has similar coordinate system)
		#This represents the number of pixels to the right, and the number of pixels down to place the image.

		pygame.draw.rect(surf,BROWN,pygame.Rect(0,0,self.displayX,self.displayY),int(self.body_width))

		

		if self.fruit:
			pygame.draw.rect(surf,RED,pygame.Rect(self.fruit.position[0]*self.scale-self.body_width/2,self.fruit.position[1]*self.scale-self.body_width/2,self.body_width,self.body_width))
		
		for id in self.snakeids: 
			if self.multisnakeobj.heads[id] :			
				pygame.draw.rect(surf, self.snake_head_colors[id],pygame.Rect(self.multisnakeobj.heads[id].position[0]*self.scale-self.body_width/2,self.multisnakeobj.heads[id].position[1]*self.scale-self.body_width/2,self.body_width,self.body_width))  
		
			for i in range(0,len(self.multisnakeobj.bodies[id])):
				if self.multisnakeobj.bodies[id][i]:
					pygame.draw.rect(surf,self.snake_body_colors[id],pygame.Rect(self.multisnakeobj.bodies[id][i].position[0]*self.scale-self.body_width/2,self.multisnakeobj.bodies[id][i].position[1]*self.scale-self.body_width/2,self.body_width,self.body_width))  
		

		if mode == "human":
			assert self.window is not None
			self.window.blit(surf, surf.get_rect())
			pygame.event.pump()
			pygame.display.update()

			self.clock.tick(self.metadata["render_fps"])
		else:
			return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))


	def close(self):
		if self.window is not None:
			import pygame

			pygame.display.quit()
			pygame.quit()


# Test code
if __name__ =="__main__":

	'''ray.init()

	config = {
        # Also try common gym envs like: "CartPole-v0" or "Pendulum-v1".
		"env": MultiAgentSnakeGameEnv,

		"video_dir": '~/video',
        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
    	# env and only the 1st sub-env in a vectorized env.
		"render_env": True,
		
		"num_workers": 2,

        # Use a vectorized env with 2 sub-envs.
		"num_envs_per_worker": 2,

		"remote_worker_envs": False,

		"framework": "torch"
	}


	def env_creator(index):
		return MultiAgentSnakeGameEnv(config)


	register_env("MultiAgentSnakeGameEnv", lambda config: env_creator)


	obj = MultiAgentSnakeGameEnv(config)

	envs = VectorEnv.vectorize_gym_envs(
		env_creator,
		num_envs = 2,
		action_space= obj.action_space,
		observation_space= obj.observation_space,
		restart_failed_sub_environments = False
	)


	envs.vector_reset()

	first_env = envs.get_sub_environments()[0]
	

	dict1 = {"snake1":1,"snake2":0}
	dict2 = {"snake1":1,"snake2":0}


	before_training = os.path.abspath(os.getcwd())+"/video/"+"before_training.mp4"

	video = VideoRecorder(first_env,before_training)

	for i in range(0,100):
		print(i)

		envs.vector_step([dict1,dict2])
		
		video.capture_frame()

	video.close()
	first_env.close()

	print(os.path.abspath(os.getcwd()))	




	def render():
		running = True
		
		pygame.init()
		pygame.display.init()
		window = pygame.display.set_mode((400,400))


		while running:
			surf = pygame.Surface((400,400))
			surf.fill(BLACK)
			pygame.transform.scale(surf,(10,10))

			
			pygame.draw.rect(surf,BROWN,pygame.Rect(0,0,400,400),20)

			if obj.fruit:
				pygame.draw.rect(surf,RED,pygame.Rect(obj.fruit.position[0]*obj.scale-obj.body_width/2,obj.fruit.position[1]*obj.scale-obj.body_width/2,obj.body_width,obj.body_width))
			
			for id in obj.snakeids: 
				if obj.multisnakeobj.heads[id] :			
					pygame.draw.rect(surf, obj.snake_head_colors[id],pygame.Rect(obj.multisnakeobj.heads[id].position[0]*obj.scale-obj.body_width/2,obj.multisnakeobj.heads[id].position[1]*obj.scale-obj.body_width/2,obj.body_width,obj.body_width))  
			
				for i in range(0,len(obj.multisnakeobj.bodies[id])):
					if obj.multisnakeobj.bodies[id][i]:
						pygame.draw.rect(surf,obj.snake_body_colors[id],pygame.Rect(obj.multisnakeobj.bodies[id][i].position[0]*obj.scale-obj.body_width/2,obj.multisnakeobj.bodies[id][i].position[1]*obj.scale-obj.body_width/2,obj.body_width,obj.body_width))  

			
			window.blit(surf, surf.get_rect())
			pygame.event.pump()
			pygame.display.update()

			
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					running = False

				
	actions_n ={"snake1":0 ,"snake2":1}


	# creating a bool value which checks
	# if game is running

 
	actions_n ={"snake1":0,"snake2":0}

	while(True):
		render()

		for id in obj.snakeids:
			key = getkey()
			if key == keys.LEFT:
				print('LEFT')
				actions_n[id] = 0 
			elif key == keys.RIGHT:
				print('RIGHT')
				actions_n[id] = 1
			elif key == keys.UP:
				print('UP')
				actions_n[id] = 2
			elif key == keys.DOWN:
				print('DOWN')
				actions_n[id] = 3

		a,b,c,d = obj.step(actions_n)
		print(a["snake1"])

		
	print("OK")'''
