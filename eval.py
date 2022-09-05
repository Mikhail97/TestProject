#Примеры ввода команды в консоль:
#python eval.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/val" –load_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model resnet -pretrained true

#python eval.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/val" –load_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model resnet -pretrained false

#python eval.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/val" –load_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model vgg -pretrained true

#python eval.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/val" –load_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model vgg -pretrained false


import joblib
import sys as sys
from sys import argv
import torch
from torch import nn
import torchvision as tv
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
            self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
            self.downsample = downsample
            self.relu = nn.ReLU()
            self.out_channels = out_channels
        
        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes = 2):
            super(ResNet, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
            self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
            self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
            self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(2048, num_classes)
		
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:
                
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(planes),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
	    
	    
        def forward(self, x):
            x = self.conv1(x)    
            x = self.maxpool(x)       
            x = self.layer0(x)        
            x = self.layer1(x)        
            x = self.layer2(x)         
            x = self.layer3(x) 
            x = self.avgpool(x)               
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x
            
            
def vgg_block(num_convs, input_channels, num_channels):

        block = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        for i in range(num_convs - 1):
            block.add_module("conv{}".format(i),
                            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
                            )
            block.add_module("relu{}".format(i),
                            nn.ReLU()
                            )

        block.add_module("pool", nn.MaxPool2d(2, stride=2))

        return block
        
conv_arch = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))

def vgg(conv_arch):
        net = nn.Sequential()

        for i, (num_convs, input_ch, num_channels) in enumerate(conv_arch):
            #print((num_convs, input_ch, num_channels) )
            net.add_module("block{}".format(i), vgg_block(num_convs, input_ch, num_channels))

        
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),  #6272
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 2))

        net.add_module('classifier', classifier)
        return net

			
def save_weights(model, name=None, path=None):
    joblib.dump(model, os.path.join(path, name))
    return None

def load_weights(path=None):
    return joblib.load(path)
   
def get_mean_and_std(loader):
    mean = std = total_images = 0
    for images, _ in loader:
        images = images.view(images.shape[0], images.shape[1], -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.shape[0]
    mean /= total_images
    std /= total_images
    return mean, std


def evaluate_accuracy(data_iter, net):
    acc_sum, n = torch.Tensor([0]), 0
    net.eval()
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)  
        y_hat = net(X)
        predicted = y_hat.argmax(axis=1)       
        acc_sum += (predicted== y).sum()
        n += y.shape[0]
    return acc_sum.item() / n, acc_sum.item(), n 



if __name__ == "__main__":
    name_script, key_datapath, path_data, key_load_to, path_load_to, key_model, name_model, key_pretrained, val_pretrained = argv

    if len(argv) < 9 or len(argv) >10:
	    print('Недостаточно значений чтобы выполнить скрипт, проверьте количество указанных ключей и значений.\nИх должно быть 9, указано {}'.format(len(argv)))
	    sys.exit()
    if (argv[8].lower() != 'true' and argv[8].lower() != 'false'):
	    print('Значение ключа -pretrained должно быть False либо True, указано {}'.format(argv[8]))
	    sys.exit()
    if (argv[7].lower() != '-pretrained' and argv[7].lower() != '–pretrained'):
	    print('Имя ключа должно быть -pretrained, указано {}'.format(argv[7]))
	    sys.exit()	
    if (argv[6].lower() != 'resnet' and argv[6].lower() != 'vgg'):
	    print('Значение ключа -model должны быть resnet либо vgg, указано {}'.format(argv[6]))
	    sys.exit()
    if (argv[5].lower() != '-model' and argv[5].lower() != '–model'):
	    print('Имя ключа должно быть -model, указано {}'.format(argv[5]))
	    sys.exit()	
    if (argv[3].lower() != '-load_to' and argv[3].lower() != '–load_to'):
	    print('Имя ключа должно быть -load_to, указано {}'.format(argv[3]))
	    sys.exit()
    if (argv[1].lower() != '-datapath' and argv[1].lower() != '–datapath'):
	    print('Имя ключа должно быть -datapath, указано {}'.format(argv[1]))
	    sys.exit()
		
    if name_model.lower() == 'vgg' and val_pretrained.lower() == 'false':
	    if os.path.isfile(path_load_to+'/vgg.pkl'):
	        load_model = load_weights(path_load_to+'/vgg.pkl')
	    else:
	    	print('Файл {} не существует'.format(path_load_to+'/vgg.pkl'))
	    	sys.exit()
    if name_model.lower() == 'vgg' and val_pretrained.lower() == 'true': 
	    if os.path.isfile(path_load_to+'/vgg_Fine_Tuning.pkl'):
	    	load_model = load_weights(path_load_to+'/vgg_Fine_Tuning.pkl')
	    else:
	        print('Файл {} не существует'.format(path_load_to+'/vgg_Fine_Tuning.pkl'))
	        sys.exit()  
    if name_model.lower() == 'resnet' and val_pretrained.lower() == 'false':
	    if os.path.isfile(path_load_to+'/resnet.pkl'):
	        #load_model = ResNet(ResidualBlock, [3, 2, 6, 3], num_classes = 2)
	        #load_model.load_state_dict(torch.load(path_load_to+'/resnet.pt'),strict=False)
	        #load_model.eval()	        
	        load_model = load_weights(path_load_to+'/resnet.pkl')
	    else:
	        print('Файл {} не существует'.format(path_load_to+'/resnet.pkl'))
	        sys.exit()  
    if name_model.lower() == 'resnet' and val_pretrained.lower() == 'true': 
	    if os.path.isfile(path_load_to+'/resnet_Fine_Tuning.pkl'):
	    	#load_model = tv.models.resnet18(num_classes = 2, pretrained=True)
	    	#load_model.load_state_dict(torch.load(path_load_to+'/resnet_Fine_Tuning.pt'),strict=False)
	        load_model = load_weights(path_load_to+'/resnet_Fine_Tuning.pkl')
	    else:
	        print('Файл {} не существует'.format(path_load_to+'/resnet_Fine_Tuning.pkl'))
	        sys.exit()  
	    	
	# Device configuration
    device ='cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
		                                        tv.transforms.Resize((255, 255)),
		                                        transforms.ToTensor()
		                                    ])

    val_data = datasets.ImageFolder(path_data,transform=transform)
    val_loader = DataLoader(val_data,batch_size=32)	  
	
    mean, std = get_mean_and_std(val_loader)


    transform = transforms.Compose([
		                                        tv.transforms.Resize((255, 255)),
		                                        transforms.ToTensor(),
		                                        transforms.Normalize(mean, std)
		                                    ])
		                                                                                
    val_data = datasets.ImageFolder(path_data,transform=transform)
    val_loader = DataLoader(val_data,batch_size=64)                                            

    print('Изображения загрузились в DataLoader')  	
	
    val_acc, val_acc_sum, n = evaluate_accuracy(val_loader, load_model)
    print('Всего изображений {}, правильно классифицированных {}'.format(n,val_acc_sum))
    print('точность %.3f' % (val_acc))  
