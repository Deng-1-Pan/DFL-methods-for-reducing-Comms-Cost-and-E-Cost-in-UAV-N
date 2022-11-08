

import torch 
from torchvision import datasets, transforms

def get_dataset(dir, name):
	if name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4), #Add 4 pixel dots at the top, bottom and left
			transforms.RandomHorizontalFlip(), #Horizontal flip
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(dir, train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
		
	
	return train_dataset, eval_dataset