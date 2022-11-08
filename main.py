import argparse, json
import datetime
import os, sys
import logging
import torch, random
import matplotlib.pyplot as plt
import numpy as np
import models, datasets
from scipy.spatial.distance import cdist

from training import *
from client import *
from UAV import *
from Energy_record import *
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	

	with open('utils/conf.json') as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	tr = training(conf, eval_datasets) # pre-store the value
	
	# UAV initial generate
	coords = np.array([(random.uniform(-5.0,5.0), random.uniform(-5.0,5.0), random.uniform(-5.0,5.0)) for _ in range(conf["no_models"])])
	if conf["Cluester Head Situation"] != 'S':
		index0, index1, label_coords0, label_coords1 = uav_register(coords)	
	else:
		#The cluster head
		index0 = np.where(sum(cdist(coords, coords)) == min(sum(cdist(coords, coords))))[0][0]
	
	sim_server, clients, new_clients0, new_clients1 = [], [], [], []
 	
	# Set the initial model for all clients
	for c in range(conf["no_models"]):
		clients.append(Client(conf, train_datasets, c))
	
	# Regist all of the information into one list
	register = [{'Drone':{'Coordinate': coords[i], 'client': clients[i], 'index': i, 'Battery': 100}}for i in range(conf["no_models"])] #register[2]['Drone']['Coordinate'] to access data	
 
	# Record the Energy state
	file_name = 'UAV_'+str(conf['no_models'])+str(conf["Method"])+str(conf["Local round"])+str(conf["Global round"])+".xls"
	sheet_name = ['Battery Left','Battery used', 'A & L', 'Bandwidth']
	title = [x for x in range(len(register))]
	title.insert(0, 'Drone Index')
	write_excel(file_name, sheet_name, title)
 
	if conf['Method'] == 'O':
		A_l = []
	elif conf["Cluester Head Situation"] == 'M':
		# Store the simulate server information
		sim_server.append(register[index0]['Drone']['client'])
		sim_server.append(register[index1]['Drone']['client'])
  
		# Store the clients with simulate server
		for l in label_coords0:
			new_clients0.append(register[l]['Drone']['client'])
		for l in label_coords1:
			new_clients1.append(register[l]['Drone']['client'])
   
		# Store the cluster head
		possible_head = list(range(conf['no_models']))
		count_of_times = [0]*len(possible_head)
  
		# Create list to store accuracy and loss for each cluster
		A_l = [[] * 1 for _ in range(4)]
  
		cluster_head = {}
		for possible_head, count_of_times in zip(possible_head, count_of_times):
			cluster_head[possible_head] = count_of_times
   
		for key, value in cluster_head.items():
			if sim_server[0].client_id == key:
				cluster_head[key] += 1
			elif sim_server[1].client_id == key:
				cluster_head[key] += 1
		print('The Initial cluster heads are Drone {} and Drone {}'.format(sim_server[0].client_id, sim_server[1].client_id))
  
	elif conf["Cluester Head Situation"] == 'S':
		sim_server.append(register[index0]['Drone']['client'])
		for l in range(conf['no_models']):
			new_clients0.append(register[l]['Drone']['client'])
   
		# Store the cluster head
		possible_head = list(range(conf['no_models']))
		count_of_times = [0]*len(possible_head)
  
		# Create list to store accuracy and loss for each cluster
		A_l = [[] * 1 for _ in range(2)]
  
		cluster_head = {}
		for possible_head, count_of_times in zip(possible_head, count_of_times):
			cluster_head[possible_head] = count_of_times
		for key, value in cluster_head.items():
			if sim_server[0].client_id == key:
				cluster_head[key] += 1
		print('The Initial cluster head is Drone {}'.format(sim_server[0].client_id))
 
	accuracy_list_0, accuracy_list_1, loss_list_0, loss_list_1 = [], [], [], []
	
	print("\n")
 
	# Record battery state for each training
	record_list_Bleft = [[0] * 1 for _ in range(len(register))]
	for i in range(len(register)):
		record_list_Bleft[i][0] = register[i]['Drone']['Battery']
	record_list_Bused = [[] * 1 for _ in range(len(register))]
 
	# Build the list for record file size
	receive_list = [[] * 1 for _ in range(len(register))]
	send_list = [[] * 1 for _ in range(len(register))]
 
	if conf['Method'] == 'O':
		accurate_list = [[] * 1 for _ in range(len(register))]
		loss_list = [[] * 1 for _ in range(len(register))]
		accurate_list, loss_list, Battery_Left, record_list_Bused = tr.normal_train(clients, register, accurate_list, loss_list, record_list_Bleft, record_list_Bused)

		
		for i in range(len(accurate_list)):
			A_l.append(accurate_list[i])
		for i in range(len(loss_list)):
			A_l.append(loss_list[i])
		append_excel(file_name, Battery_Left, 0)
		append_excel(file_name, record_list_Bused, 1)
		append_excel(file_name, A_l, 2)

		# Plot accuracy and loss
		plt.figure()
		for i in range(len(accurate_list)): 
			plt.plot(range(len(accurate_list[i])), accurate_list[i], label= 'Client'+str(i))
		plt.ylabel('Accuracy%')
		plt.xlabel('Epoch')
		x_factor=plt.MultipleLocator(5)
		y_factor=plt.MultipleLocator(5)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		axis.yaxis.set_major_locator(y_factor)
		
		plt.grid()
		plt.legend()
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['local_epochs'])+'l_'+str(conf['global_epochs'])
				+'g_Accuracy_no_server_'+str(conf["no_models"])+'_Drones.png')
	
		plt.figure()
		for i in range(len(accurate_list)):
			plt.plot(range(len(loss_list[i])), loss_list[i], label= 'Client'+str(i))
		plt.ylabel('loss')
		plt.xlabel('Epoch')
		x_factor=plt.MultipleLocator(5)
		y_factor=plt.MultipleLocator(0.2)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		axis.yaxis.set_major_locator(y_factor)
		
		plt.grid()
		plt.legend()
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['local_epochs'])+'l_'+str(conf['global_epochs'])
				+'g_Loss_no_server_'+str(conf["no_models"])+'_Drones.png')
	
		# Plot Energy usage
		Fin_Battery = []
		for i in range(len(register)):
			Fin_Battery.append(np.round(Battery_Left[i][conf['global_epochs']],2))
		plt.figure()
		plt.bar(range(len(Fin_Battery)), Fin_Battery)
		for a,b in zip(list(range(len(Fin_Battery))),Fin_Battery):
			plt.text(a, b+0.05, '%.2f' % b, ha='center', va= 'bottom',fontsize=8)
		plt.ylabel('Bettery left')
		plt.xlabel('Drone Index')
		x_factor=plt.MultipleLocator(1)
		y_factor=plt.MultipleLocator(5)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		axis.yaxis.set_major_locator(y_factor)
		plt.xlim(-0.8,conf['no_models']-0.2)
		plt.grid(axis = 'y')
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['local_epochs'])+'l_'+str(conf['global_epochs'])
				+'g_Battery_usage_'+str(conf["no_models"])+'_Drones.png')
		plt.show()
	else:
		i = 0
		while i < conf['global_epochs']:
			if conf["Cluester Head Situation"] == 'S':
				acc0, loss0, register, record_list_Bleft, record_list_Bused, send_list, receive_list= tr.one_server_train(new_clients0, sim_server, 
																						i, register, record_list_Bleft, record_list_Bused, send_list, receive_list)
			else:
				acc0, loss0, acc1, loss1, acc_f, loss_f, register, record_list_Bleft, record_list_Bused, send_list, receive_list = tr.DFL_UN_train(new_clients0, 
                                                                                        new_clients1, sim_server, i, register, 
                                                                                        record_list_Bleft, record_list_Bused, send_list, receive_list)
				accuracy_list_1.append(acc1)
				loss_list_1.append(loss1)
				
			accuracy_list_0.append(acc0)
			loss_list_0.append(loss0)
			
			# Set new position for UAV
			new_coords = np.array([(random.uniform(-5.0,5.0), random.uniform(-5.0,5.0), random.uniform(-5.0,5.0)) for _ in range(conf["no_models"])])
			if conf["Cluester Head Situation"] == 'M':
				index0, index1, label_coords0, label_coords1 = uav_register(new_coords)
				new_clients0, new_clients1, sim_server = Client.uav_assignment_2(conf, sim_server, register, new_coords, 
																		index0, index1, label_coords0, label_coords1) # Get the new coordinate and assign new clients and servers
				if i != conf['global_epochs'] - 1:
					for key, value in cluster_head.items():
						if sim_server[0].client_id == key:
							cluster_head[key] += 1
						elif sim_server[1].client_id == key:
							cluster_head[key] += 1
					print('After {} round the cluster heads are Drone {} and Drone {}'.format(i+1, sim_server[0].client_id, sim_server[1].client_id))
			else:
				index0= np.where(sum(cdist(new_coords, new_coords)) == min(sum(cdist(new_coords, new_coords))))[0][0]
				sim_server[0] = register[index0]['Drone']['client']
				new_clients0 = Client.uav_assignment_1(conf, register, new_coords)
				if i != conf['global_epochs'] - 1:
					for key, value in cluster_head.items():
						if sim_server[0].client_id == key:
							cluster_head[key] += 1
					print('After {} round the cluster head is Drone {} '.format(i+1, sim_server[0].client_id))
			i += 1

		print('\n')
		if conf["Cluester Head Situation"] == 'M':
			print('The final model which aggregate the newest two model with {0:.6f}% Accuracy and {1:.6f} loss'.format(acc_f,loss_f))
		
		with open('test.txt','a') as file0:
			print(conf["Method"]+'_'+str(conf['Local round'])+'l_'+str(conf['Global round'])+'g_'+str(conf["no_models"])+'_Drones',file=file0)
			for key, value in cluster_head.items():
				print('Drone {} has been the cluster head of {} times'.format(key, value),file=file0)
   
		send_list = [sum(i) for i in send_list]
		receive_list = [sum(i) for i in receive_list]
		overall_list = [x + y for x,y in zip(send_list,receive_list)]
		S_R = [[] * 1 for _ in range(3)]
		for i in range(3):
			if i == 0:
				for j in range(len(send_list)):
					S_R[i].append(int(send_list[j]))
			elif i == 1:
				for j in range(len(receive_list)):
					S_R[i].append(int(receive_list[j]))
			elif i == 2:
				for j in range(len(overall_list)):
					S_R[i].append(int(overall_list[j]))
     
		if conf['global_epochs'] == 100:
			pass
		else:
			if conf["Cluester Head Situation"] == 'S':
				for i in range(2):
					if i == 0:
						for j in range(len(accuracy_list_0)):
							A_l[i].append(accuracy_list_0[j])
					elif i == 1:
						for j in range(len(loss_list_0)):
							A_l[i].append(loss_list_0[j])
			else:					
				for i in range(4):
					if i == 0:
						for j in range(len(accuracy_list_0)):
							A_l[i].append(accuracy_list_0[j])
						A_l[i].append(acc_f)
					elif i == 1:
						for j in range(len(accuracy_list_0)):
							A_l[i].append(accuracy_list_1[j])
					elif i == 2:
						for j in range(len(loss_list_0)):
							A_l[i].append(loss_list_0[j])
						A_l[i].append(loss_f)
					elif i == 3:
						for j in range(len(loss_list_1)):
							A_l[i].append(loss_list_1[j])

			append_excel(file_name, record_list_Bleft, 0)
			append_excel(file_name, record_list_Bused, 1)
			append_excel(file_name, A_l, 2)
			append_excel(file_name, S_R, 3)
	
		# Plot accuracy and loss
		plt.figure()
		if conf["Cluester Head Situation"] == 'S':
			plt.plot(range(len(accuracy_list_0)), accuracy_list_0[:],'b', label='cluster0')
		else:
			plt.plot(range(len(accuracy_list_0)), accuracy_list_0[:],'b', label='cluster0')
			plt.plot(range(len(accuracy_list_1)), accuracy_list_1[:],'r', label='cluster1')
		plt.ylabel('Accuracy%')
		plt.xlabel('Epoch')
		x_factor=plt.MultipleLocator(5)
		y_factor=plt.MultipleLocator(5)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		axis.yaxis.set_major_locator(y_factor)
		plt.xlim(-0.5,np.ceil(conf['global_epochs'] / 5) * 5)
		
		plt.grid()
		plt.legend()
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['Local round'])+'l_'+str(conf['Global round'])
				+'g_Accuracy_multi_server_'+str(conf["no_models"])+'_Drones.png')
	
		plt.figure()
		if conf["Cluester Head Situation"] == 'S':
			plt.plot(range(len(loss_list_0)), loss_list_0[:],'b', label='loss0')
		else:
			plt.plot(range(len(loss_list_0)), loss_list_0[:],'b', label='loss0')
			plt.plot(range(len(loss_list_1)), loss_list_1[:],'r', label='loss1') 
		plt.ylabel('loss')
		plt.xlabel('Epoch')
		x_factor=plt.MultipleLocator(5)
		y_factor=plt.MultipleLocator(0.2)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		axis.yaxis.set_major_locator(y_factor)
		plt.xlim(-0.5, np.ceil(conf['global_epochs'] / 5) * 5)
		plt.grid()
		plt.legend()
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['Local round'])+'l_'+str(conf['Global round'])
				+'g_Loss_multi_server_'+str(conf["no_models"])+'_Drones.png')
	
		# Plot Energy usage
		Fin_Battery = []
		for i in range(len(register)):
			Fin_Battery.append(np.round(register[i]['Drone']['Battery'],2))
		plt.figure()
		plt.bar(range(len(Fin_Battery)), Fin_Battery)
		for a,b in zip(list(range(len(Fin_Battery))),Fin_Battery):
			plt.text(a, b+0.05, '%.2f' % b, ha='center', va= 'bottom',fontsize=8)
		plt.ylabel('Battery left')
		plt.xlabel('Drone Index')
		x_factor=plt.MultipleLocator(1)
		y_factor=plt.MultipleLocator(5)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		axis.yaxis.set_major_locator(y_factor)
		plt.xlim(-0.8,conf['no_models']-0.2)
		plt.grid(axis = 'y')
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['Local round'])+'l_'+str(conf['Global round'])
				+'g_Battery_usage_'+str(conf["no_models"])+'_Drones.png')
  
		#Plot the communication overhead
		plt.figure()
		for num in range(len(send_list)):
			plt.bar(num, send_list[num])
		for a,b in zip(list(range(len(send_list))),send_list):
			plt.text(a, b+0.05, '%.2E' % b, ha='center', va= 'bottom',fontsize=8)
		plt.ylabel('The file size send by drone/Bytes')
		plt.xlabel('Drone Index')
		plt.xlim(-0.8,conf['no_models']-0.2)
		plt.grid(axis = 'y')
		x_factor=plt.MultipleLocator(1)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['Local round'])+'l_'+str(conf['Global round'])
				+'g_Send_record_'+str(conf["no_models"])+'_Drones.png')
		plt.figure()
		for num in range(len(receive_list)):
			plt.bar(num, receive_list[num])
		for a,b in zip(list(range(len(receive_list))),receive_list):
			plt.text(a, b+0.05, '%.2E' % b, ha='center', va= 'bottom',fontsize=8)
		plt.ylabel('The file size receive by drone/Bytes')
		plt.xlabel('Drone Index')
		plt.xlim(-0.8,conf['no_models']-0.2)
		plt.grid(axis = 'y')
		axis=plt.gca()
		x_factor=plt.MultipleLocator(1)
		axis.xaxis.set_major_locator(x_factor)
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['Local round'])+'l_'+str(conf['Global round'])
				+'g_receive_record_'+str(conf["no_models"])+'_Drones.png')
  
		fig, ax = plt.subplots()
		ax.bar([num for num in range(len(send_list))], send_list, label='Send')
		ax.bar([num for num in range(len(receive_list))], receive_list, bottom=send_list,
			label='Receive')

		ax.set_ylabel('The file size receive&send by drone/Bytes')
		ax.set_xlabel('Drone Index')
		ax.legend()
		for a,b in zip(list(range(len(overall_list))),overall_list):
			plt.text(a, b+0.05, '%.2E' % b, ha='center', va= 'bottom',fontsize=8)
		plt.grid(axis = 'y')
		x_factor=plt.MultipleLocator(1)
		axis=plt.gca()
		axis.xaxis.set_major_locator(x_factor)
		plt.xlim(-0.8,conf['no_models']-0.2)
		plt.savefig('./save/'+conf["Method"]+'_'+str(conf['Local round'])+'l_'+str(conf['Global round'])
				+'g_receive&send_record_'+str(conf["no_models"])+'_Drones.png')
		plt.show()