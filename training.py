import random
import models, torch
import argparse, json
import datetime
import os, sys
import logging
from time import time
import numpy as np

from aggregate_server import *
from Energy_record import *


class training(object):

	def __init__(self, conf, eval_datasets):
			
			self.conf = conf
			
			self.eval_datasets = eval_datasets

		
	def normal_train(self, clients, register, accuracy_list, loss_list, Battery_Left, record_list_Bused):
		for e in range(self.conf["global_epochs"]):
			for c in clients:
				# local training
				lt_start_time = time()
				diff = c.local_train(c.local_model)		
				lr_time = time() - lt_start_time
				record_list_Bused = update_U(self.conf, np.round((lr_time*59)/9864, 6), c, record_list_Bused)
				register[c.client_id]['Drone']['Battery'] -= np.round((lr_time*59)/9864, 6)
				register, Battery_Left = update_B(register, c, Battery_Left)
    		
				aggregate_client_server = aggregate_server(c.local_model, self.conf, self.eval_datasets)
				acc, loss = aggregate_client_server.model_eval()
				
				accuracy_list[c.client_id].append(acc)
				loss_list[c.client_id].append(loss)
				
				print("Global Epoch %d, Client %d | accuracy: %f, loss: %f\n" % (e+1, c.client_id, acc, loss))
    
		return accuracy_list, loss_list, Battery_Left, record_list_Bused

	def one_server_train(self, new_clients0, sim_server, e, register, record_list_Bleft, record_list_Bused, send_list, receive_list):
		if self.conf["Drone Selection"] == 'Fix':
			pass
		else:
			id_list0 = [new_clients0[n].client_id for n in range(len(new_clients0))]
			copy_clients_id0 = list(filter(lambda a: a != sim_server[0].client_id, id_list0))

			if len(copy_clients_id0) == 1:
				random_value0 = copy_clients_id0
			else:
				random_value0 = random.sample(copy_clients_id0, k = random.randint(1, len(copy_clients_id0)))
			copy_client0 = []
			for m0 in new_clients0:
				for m1 in random_value0:
					if m0.client_id == m1:
						copy_client0.append(m0)
			copy_client0.append(sim_server[0])
			new_clients0 = copy_client0

   
		global_train_start = time()
		random_choice_client = sim_server
		aggregate_client_server = aggregate_server(random_choice_client[0].local_model, self.conf, self.eval_datasets)

		weight_accumulator_0 = {}
		for name, params in aggregate_client_server.global_model.state_dict().items():
			weight_accumulator_0[name] = torch.zeros_like(params) 

		# local training
		for c in new_clients0:
			lt_start_time = time()
			diff = c.local_train(aggregate_client_server.global_model)
			lr_time = time() - lt_start_time
			record_list_Bused = update_U(self.conf, np.round((lr_time*59)/9864, 6), c, record_list_Bused)	
			register[c.client_id]['Drone']['Battery'] -= np.round((lr_time*59)/9864, 6)
   
			# Record battery state
			register, record_list_Bleft = update_B(register, c, record_list_Bleft)
   
			Temp_list =  np.array([], dtype=np.int64)
			for name, params in aggregate_client_server.global_model.state_dict().items():
				filesize = sys.getsizeof(diff[name].storage())/1000 #Kb
				Temp_list = np.append(Temp_list, filesize)
			if c.client_id == random_choice_client[0].client_id:
				receive_list[c.client_id].append(np.sum(Temp_list) * (len(new_clients0)-1))
			else:
				send_list[c.client_id].append(np.sum(Temp_list))
				eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
				energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
				register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
				record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
				register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
    
				register[c.client_id]['Drone']['Battery'] -= energy_cost
				record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
				register, record_list_Bleft = update_B(register, c, record_list_Bleft)
				

			
			for name, params in aggregate_client_server.global_model.state_dict().items():
				weight_accumulator_0[name].add_(diff[name])
		
		aggregate_client_server.model_aggregate(weight_accumulator_0)
  
		Temp_list =  np.array([], dtype=np.int64)
		for name, params in aggregate_client_server.global_model.state_dict().items():
			Temp_list = np.append(Temp_list, sys.getsizeof(aggregate_client_server.global_model.state_dict()[name].storage()))
		for num,c in enumerate(new_clients0):
			new_clients0[num] = c.model_update(c, aggregate_client_server.global_model)
			if c.client_id == random_choice_client[0].client_id:
				send_list[c.client_id].append(np.sum(Temp_list) * (len(new_clients0)-1))
			else:
				eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
				energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
				register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
				record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
				register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)

				register[c.client_id]['Drone']['Battery'] -= energy_cost
				record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
				register, record_list_Bleft = update_B(register, c, record_list_Bleft)
    
				receive_list[c.client_id].append(np.sum(Temp_list))


		acc0, loss0 = aggregate_client_server.model_eval()

		print("For cluster 0: Round %d, accuracy: %f, loss: %f\n" % (e+1, acc0, loss0))
		global_train_time = time() - global_train_start
		print("Global Epoch {} | Time: {} min\n\n".format(e+1,np.around(((global_train_time)/60),2)))
  
		return acc0, loss0, register, record_list_Bleft, record_list_Bused, send_list, receive_list
		
	def final_round_aggregate(self, sim_server0, sim_server1):
		final_weight_sum = {}

		for name, data in sim_server1.local_model.state_dict().items():
			final_weight_sum[name] = (data - sim_server0.local_model.state_dict()[name])
		return final_weight_sum

	def DFL_UN_train(self, new_clients0, new_clients1, sim_server, e, register, record_list_Bleft, record_list_Bused, send_list, receive_list):
		if self.conf['Method'] == 'N':
			limited = list(range(self.conf['Local round'], self.conf['Local round'] + self.conf['Global round']))
		elif self.conf['Method'] == 'N1':
			limited = list(range(self.conf['Global round'] + self.conf['Local round']))
   
		# Check whether all drones participate
		if self.conf["Drone Selection"] == 'Fix':
			pass
		else:
			id_list0 = [new_clients0[n].client_id for n in range(len(new_clients0))]
			id_list1 = [new_clients1[n].client_id for n in range(len(new_clients1))]
			copy_clients_id0 = list(filter(lambda a: a != sim_server[0].client_id, id_list0))
			copy_clients_id1 = list(filter(lambda a: a != sim_server[1].client_id, id_list1))

			if len(copy_clients_id0) == 1 or len(copy_clients_id1) == 1:
				random_value0 = copy_clients_id0
				random_value1 = copy_clients_id1
			else:
				random_value0 = random.sample(copy_clients_id0, k = random.randint(1, len(copy_clients_id0)))
				random_value1 = random.sample(copy_clients_id1, k = random.randint(1, len(copy_clients_id1)))
			copy_client0, copy_client1 = [],[]
			for m0,n0 in zip(new_clients0, new_clients1):
				for m1, n1 in zip(random_value0, random_value1):
					if m0.client_id == m1:
						copy_client0.append(m0)
					if n0.client_id == n1:
						copy_client1.append(n0)
			copy_client0.append(sim_server[0])
			copy_client1.append(sim_server[1])
			new_clients0 = copy_client0
			new_clients1 = copy_client1

		global_train_start = time()
		random_choice_client = sim_server
		for i in range(2):
			aggregate_client_server = aggregate_server(random_choice_client[i].local_model, self.conf, self.eval_datasets)
			if i == 0:
				weight_accumulator_0 = {}
				for name, params in aggregate_client_server.global_model.state_dict().items(): 
					weight_accumulator_0[name] = torch.zeros_like(params) 
	
				# local training
				for c in new_clients0:
					lt_start_time = time()
					diff = c.local_train(aggregate_client_server.global_model)
					lr_time = time() - lt_start_time
					record_list_Bused = update_U(self.conf, np.round((lr_time*59)/9864, 6), c, record_list_Bused)	
					register[c.client_id]['Drone']['Battery'] -= np.round((lr_time*59)/9864, 6)
     
					# Record battery state
					register, record_list_Bleft = update_B(register, c, record_list_Bleft)
   
					
					for name, params in aggregate_client_server.global_model.state_dict().items():
						weight_accumulator_0[name].add_(diff[name])
     
					#Send the weight difference to the server
					Temp_list =  np.array([], dtype=np.int64)
					for name, params in aggregate_client_server.global_model.state_dict().items():
						Temp_list = np.append(Temp_list, sys.getsizeof(diff[name].storage()))
					if c.client_id == random_choice_client[0].client_id:
						receive_list[c.client_id].append(np.sum(Temp_list) * (len(new_clients0)-1))
					else:
						send_list[c.client_id].append(np.sum(Temp_list))
      
						eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
						energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
						register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
						record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
						register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
      
						register[c.client_id]['Drone']['Battery'] -= energy_cost
						record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
						register, record_list_Bleft = update_B(register, c, record_list_Bleft)

				aggregate_client_server.model_aggregate(weight_accumulator_0)
				if e%(self.conf['Global round'] + self.conf['Local round']) in limited:
					pass
				else:
					Temp_list =  np.array([], dtype=np.int64)
					for name, params in aggregate_client_server.global_model.state_dict().items():
						Temp_list = np.append(Temp_list, sys.getsizeof(aggregate_client_server.global_model.state_dict()[name].storage()))
					for num,c in enumerate(new_clients0):
						mu_start_time = time()
						new_clients0[num] = c.model_update(c, aggregate_client_server.global_model)
						mu_time = time() - mu_start_time

						
						if c.client_id == random_choice_client[0].client_id:		
							send_list[c.client_id].append(np.sum(Temp_list) * (len(new_clients0)-1))
						else:
							eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
							energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
							register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
							register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
		
							register[c.client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
							register, record_list_Bleft = update_B(register, c, record_list_Bleft)

							receive_list[c.client_id].append(np.sum(Temp_list))

	
				acc0, loss0 = aggregate_client_server.model_eval()
	
				print("For cluster 0: Round %d, accuracy: %f, loss: %f\n" % (e+1, acc0, loss0))

			else:
				weight_accumulator_1 = {}
				for name, params in aggregate_client_server.global_model.state_dict().items():
					weight_accumulator_1[name] = torch.zeros_like(params)
			
				for c in new_clients1:
					lt_start_time = time()
					diff = c.local_train(aggregate_client_server.global_model)
					lr_time = time() - lt_start_time
					record_list_Bused = update_U(self.conf, np.round((lr_time*59)/9864, 6), c, record_list_Bused)
					register[c.client_id]['Drone']['Battery'] -= np.round((lr_time*59)/9864, 6)
					register, record_list_Bleft = update_B(register, c, record_list_Bleft)
     
					for name, params in aggregate_client_server.global_model.state_dict().items():
						weight_accumulator_1[name].add_(diff[name])
     
					Temp_list =  np.array([], dtype=np.int64)
					for name, params in aggregate_client_server.global_model.state_dict().items():
						Temp_list = np.append(Temp_list, sys.getsizeof(diff[name].storage()))
					if c.client_id == random_choice_client[1].client_id:
						receive_list[c.client_id].append(np.sum(Temp_list) * (len(new_clients1)-1))
					else:
						send_list[c.client_id].append(np.sum(Temp_list))
						eucli_dis = np.linalg.norm(register[random_choice_client[1].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
						energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
						register[random_choice_client[1].client_id]['Drone']['Battery'] -= energy_cost
						record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[1], record_list_Bused)
						register, record_list_Bleft = update_B(register, random_choice_client[1], record_list_Bleft)

						register[c.client_id]['Drone']['Battery'] -= energy_cost
						record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
						register, record_list_Bleft = update_B(register, c, record_list_Bleft)
      
				aggregate_client_server.model_aggregate(weight_accumulator_1) # aggregate in that aggregate_server
				if e%(self.conf['Global round'] + self.conf['Local round']) in limited:
					pass
				else:
					Temp_list =  np.array([], dtype=np.int64)
					for name, params in aggregate_client_server.global_model.state_dict().items():
						Temp_list = np.append(Temp_list, sys.getsizeof(aggregate_client_server.global_model.state_dict()[name].storage()))
					for num,c in enumerate(new_clients1):
						new_clients1[num] = c.model_update(c, aggregate_client_server.global_model)

						if c.client_id == random_choice_client[1].client_id:		
							send_list[c.client_id].append(np.sum(Temp_list) * (len(new_clients1)-1))
						else:
							eucli_dis = np.linalg.norm(register[random_choice_client[1].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
							energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
							register[random_choice_client[1].client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[1], record_list_Bused)
							register, record_list_Bleft = update_B(register, random_choice_client[1], record_list_Bleft)
		
							register[c.client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
							register, record_list_Bleft = update_B(register, c, record_list_Bleft)
		
							receive_list[c.client_id].append(np.sum(Temp_list))
					
				acc1, loss1 = aggregate_client_server.model_eval()
    
				print("For cluster 1: Round %d, accuracy: %f, loss: %f\n" % (e+1, acc1, loss1))

				acc_f, loss_f = 1, 1
				if e%(self.conf['Global round'] + self.conf['Local round']) in limited:
					fa_start_time = time()
					weight_final = self.final_round_aggregate(random_choice_client[0], random_choice_client[1]) #if 0 is server to aggregate
					aggregate_client_server_01 = aggregate_server(random_choice_client[0].local_model, self.conf, self.eval_datasets)
					aggregate_client_server_01.model_aggregate(weight_final)
					fa_time = time() - fa_start_time
     
					Temp_list =  np.array([], dtype=np.int64)
					for name, params in random_choice_client[1].local_model.state_dict().items():
						Temp_list = np.append(Temp_list, sys.getsizeof(random_choice_client[1].local_model.state_dict()[name].storage()))
					for x in range(self.conf['no_models']):
						if random_choice_client[1].client_id == x:
							send_list[x].append(np.sum(Temp_list))
							receive_list[x].append(0)
       
							eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[random_choice_client[1].client_id]['Drone']['Coordinate']) * 0.001
							energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
							register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
							register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
						elif random_choice_client[0].client_id == x:
							send_list[x].append(0)
							receive_list[x].append(np.sum(Temp_list))
							register[random_choice_client[1].client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[1], record_list_Bused)
							register, record_list_Bleft = update_B(register, random_choice_client[1], record_list_Bleft)
						else:
							send_list[x].append(0)
							receive_list[x].append(0) 

					acc_f_01, loss_f_01 = aggregate_client_server_01.model_eval() #compare 0 with 1
     
					fa_start_time = time()
					weight_final = self.final_round_aggregate(random_choice_client[1], random_choice_client[0])
					aggregate_client_server_10 = aggregate_server(random_choice_client[1].local_model, self.conf, self.eval_datasets)#if 1 is server to aggregate
					aggregate_client_server_10.model_aggregate(weight_final)
					fa_time = time() - fa_start_time
	
					Temp_list =  np.array([], dtype=np.int64)
					for name, params in random_choice_client[0].local_model.state_dict().items():
						Temp_list = np.append(Temp_list, sys.getsizeof(random_choice_client[0].local_model.state_dict()[name].storage()))
					for x in range(self.conf['no_models']):
						if random_choice_client[0].client_id == x:
							send_list[x].append(np.sum(Temp_list))
							receive_list[x].append(0)
       
							eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[random_choice_client[1].client_id]['Drone']['Coordinate']) * 0.001
							energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
							register[random_choice_client[1].client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[1], record_list_Bused)
							register, record_list_Bleft = update_B(register, random_choice_client[1], record_list_Bleft)
						elif random_choice_client[1].client_id == x:
							send_list[x].append(0)
							receive_list[x].append(np.sum(Temp_list))

							register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
							record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
							register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
						else:
							send_list[x].append(0)
							receive_list[x].append(0) 

					acc_f_10, loss_f_10 = aggregate_client_server_10.model_eval() #compare 1 with 0

					if acc_f_01 > acc_f_10:
						acc_f = acc_f_01
						loss_f = loss_f_01
						Temp_list =  np.array([], dtype=np.int64)
						for name, params in aggregate_client_server_01.global_model.state_dict().items():
							Temp_list = np.append(Temp_list, sys.getsizeof(aggregate_client_server_01.global_model.state_dict()[name].storage()))
						for num,c in enumerate(new_clients1):
							new_clients1[num] = c.model_update(c, aggregate_client_server_01.global_model)
							if c.client_id == random_choice_client[1].client_id:
								receive_list[c.client_id].append(np.sum(Temp_list))
								send_list[c.client_id].append(0)
        
								eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
								energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
								register[random_choice_client[1].client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[1], record_list_Bused)
								register, record_list_Bleft = update_B(register, random_choice_client[1], record_list_Bleft)
							else:
								receive_list[c.client_id].append(np.sum(Temp_list))
								send_list[c.client_id].append(0)
							
								eucli_dis = np.linalg.norm(register[random_choice_client[1].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
								energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
								register[c.client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
								register, record_list_Bleft = update_B(register,c, record_list_Bleft)
						for num,c in enumerate(new_clients0):
							new_clients0[num] = c.model_update(c, aggregate_client_server_01.global_model)
							if c.client_id == random_choice_client[0].client_id:
								receive_list[c.client_id].append(0)
								send_list[c.client_id].append(np.sum(Temp_list) * (self.conf['no_models']-1))

								register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
								register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
							else:
								receive_list[c.client_id].append(np.sum(Temp_list))
								send_list[c.client_id].append(0)
        
								eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
								energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
								register[c.client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
								register, record_list_Bleft = update_B(register, c, record_list_Bleft)
        
								register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
								register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
						print("Aggregate stage: accuracy: {}, loss: {}\n".format(acc_f, loss_f))
					else:
						acc_f = acc_f_10
						loss_f = loss_f_10
						Temp_list =  np.array([], dtype=np.int64)
						for name, params in aggregate_client_server_10.global_model.state_dict().items():
							Temp_list = np.append(Temp_list, sys.getsizeof(aggregate_client_server_10.global_model.state_dict()[name].storage()))
						for num,c in enumerate(new_clients1):
							new_clients1[num] = c.model_update(c, aggregate_client_server_10.global_model)
							if c.client_id == random_choice_client[1].client_id:
								receive_list[c.client_id].append(0)
								send_list[c.client_id].append(np.sum(Temp_list) * (self.conf['no_models']-1))
							else:
								receive_list[c.client_id].append(np.sum(Temp_list))
								send_list[c.client_id].append(0)
								eucli_dis = np.linalg.norm(register[random_choice_client[1].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
								energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
								# client
								register[c.client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
								register, record_list_Bleft = update_B(register, c, record_list_Bleft)
								#cluster head
								register[random_choice_client[1].client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[1], record_list_Bused)
								register, record_list_Bleft = update_B(register, random_choice_client[1], record_list_Bleft)
						for num,c in enumerate(new_clients0):
							new_clients0[num] = c.model_update(c, aggregate_client_server_10.global_model)
							if c.client_id == random_choice_client[0].client_id:
								receive_list[c.client_id].append(np.sum(Temp_list))
								send_list[c.client_id].append(0)
								
								eucli_dis = np.linalg.norm(register[random_choice_client[1].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
								energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
								register[random_choice_client[0].client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, random_choice_client[0], record_list_Bused)
								register, record_list_Bleft = update_B(register, random_choice_client[0], record_list_Bleft)
							else:
								receive_list[c.client_id].append(np.sum(Temp_list))
								send_list[c.client_id].append(0)
        
								eucli_dis = np.linalg.norm(register[random_choice_client[0].client_id]['Drone']['Coordinate']-register[c.client_id]['Drone']['Coordinate']) * 0.001
								energy_cost = np.round(((np.sum(Temp_list)/(20*np.log2(1+((-(128.1+37.6*np.log10(eucli_dis)))*np.power(1/eucli_dis,8)*10)/(-174*20))))*10)/9864000, 6)
								#client
								register[c.client_id]['Drone']['Battery'] -= energy_cost
								record_list_Bused = update_U(self.conf, energy_cost, c, record_list_Bused)
								register, record_list_Bleft = update_B(register, c, record_list_Bleft)
        
						print("Aggregate stage: accuracy: {}, loss: {}\n".format(acc_f, loss_f))

				global_train_time = time() - global_train_start
				print("Global Epoch {} | Time: {} min\n\n".format(e+1,np.around(((global_train_time)/60),2)))

		return acc0, loss0, acc1, loss1, acc_f, loss_f, register, record_list_Bleft, record_list_Bused, send_list, receive_list