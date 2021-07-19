import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import models,transforms
from torchsummary import summary
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import copy
import random

trainLoader=None
testLoader=None
numberOfTrainData=0
numberOfTestData=0

class CustomDataset(Dataset):
    def __init__(self,dataPath,labelPath,t):
        self.data=dataPath
        self.labels=pd.read_csv(labelPath)
        self.t=t

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        #print("iiiiii   "+str(index))
        #1 channel
        image=Image.open(self.data+'/'+self.labels.iloc[index,0])
       # print(image.mode)
        #making it 3 channel
        newImage=Image.new('RGB',(image.size[0],image.size[1]))
        newImage.putdata(list(zip(image.getdata(),image.getdata(),image.getdata())))
        #print(newImage.mode)

        newImage=self.t(newImage)
        label=torch.tensor([self.labels.iloc[index,1]]).float()
        #print(label.type())
        return newImage,label


class NeuralNetwork1(nn.Module):
    def __init__(self):
        super(NeuralNetwork1, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,11,4,2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(4,4)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),

            nn.Linear(64*13*13,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096,15),
        )

    def forward(self,data):
        f=self.features(data)
       # print(f.shape)
        c=torch.flatten(f,1)

        c=self.classifier(c)
        return c

class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.classifier=nn.Sequential(
            nn.Dropout(),

            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),

            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),

            nn.ReLU(inplace=True),
            nn.Linear(4096, 15),
        )

    def forward(self,data):
        f=self.features(data)
       # print(f.shape)
        c=torch.flatten(f,1)

        c=self.classifier(c)
        return c

class NeuralNetwork3(nn.Module):
    def __init__(self):
        super(NeuralNetwork3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),

            nn.ReLU(inplace=True),
            nn.Linear(4096, 15),
        )


    def forward(self,data):
        f=self.features(data)
       # print(f.shape)
        c=torch.flatten(f,1)

        c=self.classifier(c)
        return c


def AlexNetNeuralNetwork(fineTuning):
    model=models.alexnet(True)
    if fineTuning==False:
        for p in model.parameters():
            p.requires_grad=False
    newClassifier=nn.Sequential(*model.classifier[:5],nn.BatchNorm1d(4096),model.classifier[5],nn.Linear(4096,15))
    print(model.classifier)
    print(newClassifier)
    model.classifier=newClassifier

    return model


def trainClassifier(network,numberOfEpoches,num):
    criterion=nn.CrossEntropyLoss()
    #optimizer=optim.SGD(network.parameters(),lr=0.000005,momentum=0.9)
    optimizer=optim.Adam(network.parameters(),lr=0.00003,weight_decay=0.008)

    totalTrainLoss = 0
    totalTrainLoss2 = 0

    totalTestLoss = 0
    trainLosses=[]
    testLosses=[]
    trainAcc=[]
    testAcc1=[]
    testAcc5=[]
    #tempNetwork=copy.deepcopy(network)
    network.train()



    for n in range(numberOfEpoches):
        print(n)



        totalTrainLoss=0
        totalTestLoss=0
        c=1
#        print(trainLoader.shape)

        for data in trainLoader:
            if(c%100==0):
                print(c)
            c+=1

            inputs,labels=data[0].to(device),data[1].to(device)
            labels=labels.flatten().long()
            optimizer.zero_grad()
            out=network(inputs)
            loss=criterion(out,labels)
            #print(loss)
            loss.backward()
            optimizer.step()
            totalTrainLoss+=loss.item()
            #print(totalTrainLoss)
        totalTrainLoss=totalTrainLoss/numberOfTrainData
        trainLosses.append(totalTrainLoss)
        # acc=predict(trainLoader,network)
        # print("train Acc  :"+str(acc))
        # trainAcc.append(acc)

        # for data in trainLoader:
        #
        #     inputs,labels=data
        #     labels=labels.flatten().long()
        #     out=network(inputs)
        #     loss=criterion(out,labels)
        #     totalTrainLoss2+=loss.item()
        # totalTrainLoss2 = totalTrainLoss2 / numberOfTestData

        # for data in testLoader:
        #
        #     inputs,labels=data[0].to(device),data[1].to(device)
        #     labels=labels.flatten().long()
        #     out=network(inputs)
        #     loss=criterion(out,labels)
        #     totalTestLoss+=loss.item()
        # totalTestLoss = totalTestLoss / numberOfTestData
        # print("test loss  :" + str(totalTestLoss))

        acc1,totalTestLoss = predict(testLoader, network,1)
        totalTestLoss = totalTestLoss / numberOfTestData
        testLosses.append(totalTestLoss)



        print("train loss  :" + str(totalTrainLoss))
        print("test loss  :" + str(totalTestLoss))
        print("test Acc 1  :" + str(acc1))
        testAcc1.append(acc1)

        acc5,_= predict(testLoader, network, 5)
        print("test Acc 5  :" + str(acc5))
        testAcc5.append(acc5)

       # tempNetwork=copy.deepcopy(network)

    plotEpoch(trainLosses,testLosses,num,'Loss',['train loss','test loss'])
    plotEpoch(testAcc1,testAcc5,num,'Accuracy',['top 1 accuracy','top 5 accuracy'])
    torch.save(network.state_dict(),'./neural_network_'+str(num)+'.pth')

    return network

def plotEpoch(train,test,num,name,legend):
    plt.figure()
    plt.plot(train)
    plt.plot(test)
    plt.legend(legend)
    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.savefig(name+'_'+str(num)+'.jpg')

def predict(dataloader,classifier,num):
    correct=0
    total=0
    totalTestLoss=0
    criterion=nn.CrossEntropyLoss()
    classifier.eval()

    with torch.no_grad():
        for data in dataloader:
            inputs,labels=data[0].to(device),data[1].to(device)
            labels=labels.flatten().long()
            total+=len(labels)

            out=classifier(inputs)
            if(num==1):
                loss = criterion(out, labels)
                totalTestLoss += loss.item()
            _,predicted=torch.topk(out.data,num)
            predicted=predicted.t()
           # print(labels,predicted)

            labels=labels.expand_as(predicted)
           # print(total,labels,(predicted==labels),(predicted==labels).sum().item())
            #_,predicted=torch.max(out.data,1)
            # for ind,l in enumerate(labels):
            #     if(l in predicted[ind]):
            #         correct+=1
            correct+=(predicted==labels).sum().item()
            #print(labels,predicted,(predicted==labels),(predicted==labels).sum(),correct,total)
    return (correct/total),totalTestLoss

#run only once to set images directory and labels csv file to create custom dataset
#if processed data has already been created , set alreadySetUp to True
def setUpCustomData(type,alreadySetUp):
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if(alreadySetUp==False):

       folders = os.listdir('./data/' + type)
       file=open('./processed_data/'+type+'_labels.csv','w',newline='')
       writer=csv.writer(file)



       for l, subDir in enumerate(folders):
           print(subDir)
           ims = os.listdir('./data/' + type + '/' + subDir)
           for im in ims:
               image = cv2.imread('./data/' + type + '/' + subDir + '/' + im,cv2.IMREAD_GRAYSCALE)
               if(type=='Train'):
                  rotationValue=random.randint(-30,30)
                  R=cv2.getRotationMatrix2D((image.shape[1]//2,image.shape[0]//2),rotationValue,1)

                  rotated=cv2.warpAffine(image,R,(image.shape[1],image.shape[0]))

                  flipped=cv2.flip(image,1)

                  cv2.imwrite('./processed_data/' + type + '/' + subDir+'_r_'+im, rotated)

                  cv2.imwrite('./processed_data/' + type + '/' + subDir+'_f_'+im, flipped)
                  writer.writerow([subDir + '_r_' + im, l])

                  writer.writerow([subDir + '_f_' + im, l])

               cv2.imwrite('./processed_data/' + type + '/' + subDir+'_'+im, image)
               writer.writerow([subDir+'_'+im,l])


       file.close()
    dataset=CustomDataset('./processed_data/' + type,'./processed_data/'+type+'_labels.csv',t)
    if(type=='Train'):
        global trainLoader,numberOfTrainData
        numberOfTrainData=len(dataset)
        print(numberOfTrainData)
        trainLoader=DataLoader(dataset,batch_size=12,shuffle=True)
    else:
        global testLoader,numberOfTestData
        numberOfTestData=len(dataset)
        print(numberOfTestData)
        testLoader=DataLoader(dataset,batch_size=12,shuffle=True)

# from google.colab import drive
# drive.mount('/content/gdrive')
def mkdir(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
mkdir('processed_data')
mkdir('processed_data/Train')
mkdir('processed_data/Test')
setUpCustomData('Train',True)
setUpCustomData('Test',True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# nn1=NeuralNetwork1()
# nn1.to(device)
# c1=trainClassifier(nn1,30,1)


# nn2=NeuralNetwork2()
# nn2.to(device)
# c2=trainClassifier(nn2,30,2)
#
# nn3=NeuralNetwork3()
# nn3.to(device)
# c3=trainClassifier(nn3,3,3)

nn4=AlexNetNeuralNetwork(False)
nn4.to(device)
c4=trainClassifier(nn4,2,4)

nn5=AlexNetNeuralNetwork(True)
nn5.to(device)
c5=trainClassifier(nn5,2,5)
