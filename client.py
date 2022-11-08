from time import time
import models, torch, copy
import numpy as np
import random


class Client(object):

	def __init__(self, conf, train_dataset, id = -1):
		
		self.conf = conf
		
		self.local_model = models.get_model(self.conf["model_name"]) 
		
		self.client_id = id
		
		self.train_dataset = train_dataset

		datasize = 5000

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                  		batch_size=conf["batch_size"], 
														sampler=torch.utils.data.sampler.RandomSampler(self.train_dataset, replacement=True, num_samples=datasize))
	
		
	def local_train(self, model):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	

		optimizer = torch.optim.SGD(self.local_model.parameters(), 
                              		lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
			
				optimizer.step()
			
			
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
	
			
		return diff


	def model_update(self, candidates, global_model):
		for name, data in candidates.local_model.state_dict().items():
			candidates.local_model.state_dict()[name].copy_(global_model.state_dict()[name])
		return candidates


	def uav_assignment_2(conf, sim_server, register, new_coords, index0, index1, label_coords0, label_coords1):
		for i in range(conf['no_models']):
			register[i]['Drone']['Coordinate'] = new_coords[i] # Update the coordinates
   
		for index in range(2):
			if index == 0:
				sim_server[index] = register[index0]['Drone']['client']
			else:
				sim_server[index] = register[index1]['Drone']['client']
    
		new_clients0, new_clients1 = [], []
		for l in label_coords0:
			new_clients0.append(register[l]['Drone']['client'])
		for l in label_coords1:
			new_clients1.append(register[l]['Drone']['client'])

		return	new_clients0, new_clients1, sim_server
	
	def uav_assignment_1(conf, register, new_coords):
		for i in range(conf['no_models']):
			register[i]['Drone']['Coordinate'] = new_coords[i] # Update the coordinates
    
		new_clients0 = []
		for l in range(conf["no_models"]):
			new_clients0.append(register[l]['Drone']['client'])

		return	new_clients0