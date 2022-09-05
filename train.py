#Примеры ввода команды в консоль:
#python train.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/train" –save_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model ResNet -pretrained True

#python train.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/train" –save_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model ResNet -pretrained False

#python train.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/train" –save_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model vgg -pretrained True

#python train.py –datapath "/home/mikhail/it-academy/модуль 5/hymenoptera_project/input/train" –save_to "/home/mikhail/it-academy/модуль 5/hymenoptera_project/#checkpoints" –model vgg -pretrained False

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
if (argv[3].lower() != '-save_to' and argv[3].lower() != '–save_to'):
	print('Имя ключа должно быть -save_to, указано {}'.format(argv[3]))
	sys.exit()
if (argv[1].lower() != '-datapath' and argv[1].lower() != '–datapath'):
	print('Имя ключа должно быть -datapath, указано {}'.format(argv[1]))
	sys.exit()			
	

name_script, key_datapath, path_data, key_save_to, path_save_to, key_model, name_model, key_pretrained, val_pretrained = argv

def save_weights(model, name=None, path=None):
    joblib.dump(model, os.path.join(path, name))
    return None

def load_weights(path=None):
    return joblib.load(path)

# Device configuration
device ='cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
                                                tv.transforms.Resize((255, 255)),
                                                transforms.ToTensor()
                                            ])

train_data = datasets.ImageFolder(path_data,transform=transform)
train_loader = DataLoader(train_data,batch_size=32)



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

mean, std = get_mean_and_std(train_loader)

transform = transforms.Compose([
                                                tv.transforms.Resize((255, 255)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)
                                            ])

def load_split_train_test(datadir,valid_size = 0.2, train_trainsforms = None, test_trainsforms = None):
    train_trainsforms = train_trainsforms
    test_trainsforms = test_trainsforms

    train_data = datasets.ImageFolder(datadir,transform=train_trainsforms)
    test_data = datasets.ImageFolder(datadir,transform=test_trainsforms)

    num_train = len(train_data)                               # Количество тренировочных наборов
    indices = list(range(num_train))                          # Указатель обучающего набора

    split = int(np.floor(valid_size * num_train))             # Получить 20% данных в качестве набора для проверки
    np.random.shuffle(indices)                                # Перемешать набор данных

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]    # Получить обучающий набор, тестовый набор
    train_sampler = SubsetRandomSampler(train_idx)            # Disrupt обучающий набор, тестовый набор
    test_sampler  = SubsetRandomSampler(test_idx)

    # ============ Загрузчик данных: загрузка обучающего набора, тестового набора ===================
    train_loader = DataLoader(train_data,sampler=train_sampler,batch_size=64)
    test_loader = DataLoader(test_data,sampler=test_sampler,batch_size=64)
    return train_loader,test_loader


train_data_loader, test_data_loader = load_split_train_test(path_data, 0.2, train_trainsforms=transform, test_trainsforms=transform)
#print(train_data_loader.dataset.classes)


#print(print(train_data_loader.dataset[0][1]))
print('Изображения загрузились в DataLoader')
num_classes = 2
num_epochs = 20
learning_rate = 0.0001

if name_model.lower() == 'resnet':
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
		
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes = num_classes).to(device)
    print('Модель ResNet загружена')
elif name_model.lower() == 'vgg':

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

    model = vgg(conv_arch)

    print('Модель VGG загружена')	

# Loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #, momentum=0.9) #, weight_decay = 0.001, momentum = 0.9) 
loss = nn.CrossEntropyLoss(reduction='sum')


def evaluate_accuracy(data_iter, net):
    acc_sum, n = torch.Tensor([0]), 0
    net.eval()
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)  
        y_hat = net(X)
        #print(y_hat)
        predicted = y_hat.argmax(axis=1)       
        acc_sum += (predicted== y).sum()
        n += y.shape[0]
    return acc_sum.item() / n


def train(net, train_iter, test_iter, trainer, num_epochs):
    net.train()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            train_l_sum += l.item()
            predicted = y_hat.argmax(axis=1) 
            train_acc_sum += (predicted == y).sum().item()
            n += y.shape[0]
            print("Step. Train acc: {:.3f}. Train Loss: {:.3f}.             time: {:.3f}".format(
                (predicted == y).sum().item() / y.shape[0], l.item(), time.time() -  start))

        test_acc = evaluate_accuracy(test_iter, net)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))  
                 

train(model, train_data_loader, test_data_loader, optimizer, num_epochs)  
print()   
print('Обучение модели готово.')

if not os.path.isdir(path_save_to):
    os.makedirs(path_save_to)
    
save_weights(model,name=name_model.lower()+".pkl", path=path_save_to)   
print('Веса сохранены в папку {}/{}'.format(path_save_to,name_model.lower()+".pkl"))

if val_pretrained.lower() == 'true':
    transform = transforms.Compose([tv.transforms.RandomHorizontalFlip(),
                                    tv.transforms.RandomVerticalFlip(),
                                    tv.transforms.RandomResizedCrop(
                                        (255, 255), scale=(0.1, 1), ratio=(0.5, 2)),
                                    tv.transforms.Resize((255, 255)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5143, 0.4760, 0.3487],[0.2204, 0.2123, 0.2088])
                                    ])

    train_data_loader, test_data_loader = load_split_train_test(path_data, 0.2, train_trainsforms=transform, test_trainsforms=transform)

    print('Аугментация добавлена')

    if name_model.lower() == 'resnet':
        model = tv.models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)

    else:
        model = tv.models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(
                                            nn.Flatten(),
                                            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),  #6272
                                            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(4096, 2)
                                        )
        model = model.to(device)

    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)

    optimizer = torch.optim.SGD(params_to_update, lr=learning_rate)#, momentum=0.9)

    train(model, train_data_loader, test_data_loader, optimizer, num_epochs)
    print()
    print('Обучение модели с Fine-Tuning готово')
    save_weights(model,name=name_model.lower()+"_Fine_Tuning.pkl", path=path_save_to)   
    print('Веса сохранены в папку {}/{}'.format(path_save_to,name_model.lower()+"_Fine_Tuning.pkl"))
