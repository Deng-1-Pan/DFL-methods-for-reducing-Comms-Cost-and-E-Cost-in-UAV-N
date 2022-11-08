
import torch 
from torchvision import models

def get_model(name, pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 