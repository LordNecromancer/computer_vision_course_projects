import cv2
import numpy as np
import os
import random

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,precision_recall_curve,average_precision_score,plot_precision_recall_curve,plot_roc_curve
from sklearn import metrics
from skimage.feature import hog
import matplotlib.pyplot as plt
import pickle

cellSize=8
blockSize=2
kernel='rbf'
C=1
# trainPos=[]
# testPos=[]
# verPos=[]
#
# trainNeg=[]
# testNeg=[]
# verNeg=[]


def normalizeData(src,type):
    all=[]
    folders=os.listdir('./'+src)

    for l,subDir in enumerate(folders):
        print(subDir)
        ims=os.listdir('./'+src+'/'+subDir)
        for im in ims:
            image=cv2.imread('./'+src+'/'+subDir+'/'+im)
           # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if(type=='positive'):
                image=image[55:-55,75:-75]
            else:
                #image=cv2.GaussianBlur(image,(3,3),0)
                image=cv2.resize(image,(100,140))
            cv2.imwrite('./data/96_130/'+type+'/'+im,image)
            all.append('./data/96_130/'+type+'/'+im)
    return all


def allocateData(type,trainNum,testAndVerNum):
    print(1)
    train=[]
    test=[]
    verify=[]
    data = os.listdir('./data/96_130/' + type)
    print(type)

    for i in range(trainNum):
        ind=random.randint(0,len(data)-1)
        train.append('./data/96_130/' + type+'/'+data[ind])

    for i in range(testAndVerNum):
        ind1=random.randint(0,len(data)-1)
        ind2=random.randint(0,len(data)-1)

        test.append('./data/96_130/' + type+'/'+data[ind1])
        verify.append('./data/96_130/' + type+'/'+data[ind2])

    return train,test,verify



def getDescriptors(data,cellSize,blockSize,isImage):

    #params={blockSize:block_size,cellSize:cell_size}
    #hog=cv2.HOGDescriptor()
    descriptors=[]

    if(not isImage):


        for ind in data:
            image = cv2.imread(ind)
            des = hog(image, orientations=9, pixels_per_cell=(cellSize, cellSize),cells_per_block=(blockSize, blockSize),multichannel=True)
            #print(len(des.tolist()),ind)
            descriptors.append(des.tolist())
          # descriptors.append(hog.compute(image).reshape(-1).tolist())

    else:
        des = hog(data, orientations=9, pixels_per_cell=(cellSize, cellSize), cells_per_block=(blockSize, blockSize),multichannel=True)
        descriptors.append(des.tolist())
    return descriptors

def trainClassifier(desPos,desNeg,verifyDes,verifyLabels):
    print(3)

    des=desPos.copy()
    des.extend(desNeg)
    labelPos=[1]*len(desPos)
    labelNeg=[0]*len(desNeg)
    labels=labelPos.copy()
    labels.extend(labelNeg)
    bestKernel=None
    bestC=0
    bestAccuracy=0
    kernels=['rbf','linear']
    CValues=[0.75,1,2]


    for kernel in kernels:
        for e in CValues:
            print(kernel)

            classifier = make_pipeline(StandardScaler(), SVC(kernel=kernel,C=e,gamma='auto'))
            classifier.fit(des,labels)
            predLabels = classifier.predict(verifyDes)
            accuracy = metrics.accuracy_score(verifyLabels, predLabels)
            print(str(e)+" "+kernel )
            print(str(accuracy*100)+" %")
            if(accuracy>bestAccuracy):
                bestAccuracy=accuracy
                bestC=e
                bestKernel=kernel


    return classifier,bestAccuracy,bestKernel,bestC

#help(cv2.HOGDescriptor())
def findBestParameters(trainPos,trainNeg,verPos,verNeg):
    print(4)




    bestCellSize=0
    bestBlockSize=0
    bestAccuracy=0
    bestClassifier=None
    bestC=0
    bestKernel=None

    for cellSize in range(8,24,8):
        print("iteration")
        for blockSize in range(1,3):
            trainPosDes = getDescriptors(trainPos,cellSize,blockSize,False)
            trainNegDes = getDescriptors(trainNeg,cellSize,blockSize,False)
            verifyPosDes = getDescriptors(verPos,cellSize,blockSize,False)
            verifyNegDes = getDescriptors(verNeg,cellSize,blockSize,False)
            verifyDes = verifyPosDes.copy()
            verifyDes.extend(verifyNegDes)
            verifyLabels = [1] * len(verifyPosDes)
            verifyLabels.extend([0] * len(verifyNegDes))

            classifier,accuracy,kernel,C = trainClassifier(trainPosDes, trainNegDes,verifyDes,verifyLabels)
            print("main"+ str(accuracy*100)+' %')

            if(accuracy>bestAccuracy):
                bestAccuracy=accuracy
                bestCellSize=cellSize
                bestBlockSize=blockSize
                bestC=C
                bestKernel=kernel
                bestClassifier=classifier

    return bestClassifier,bestCellSize,bestBlockSize,bestKernel,bestC

def getClassifier(trainPos,trainNeg,cellSize,blockSize,kernel,C):
    desPos = getDescriptors(trainPos, cellSize, blockSize,False)
    desNeg = getDescriptors(trainNeg, cellSize, blockSize,False)
    des = desPos.copy()
    des.extend(desNeg)
    labelPos = [1] * len(desPos)
    labelNeg = [0] * len(desNeg)
    labels = labelPos.copy()
    labels.extend(labelNeg)
    classifier = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma='auto'))
    classifier.fit(des, labels)
    return classifier


def findFaces(im,classifier,cellSize,blockSize,padSize):
    print("YOOOOHOOO BAABBYY")
    resTemp=im.copy()
    imPadded = np.zeros((im.shape[0] + 200, im.shape[1] + 200, 3))
    imPadded[100:-100, 100:-100, :] = im
 #   im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    levels=3
   # ratio=0.66

    finalCords=[]
    # windowW=60
    # windowH=85
    tempIm=imPadded.copy()

    #for l in range(levels):
    for ratio in range(1,6):
        windowW=int(ratio*0.35*96)
        windowH=int(ratio*0.35*130)
        cord = []

        for j in range(0,tempIm.shape[0]-windowH,10):
            for i in range(0,tempIm.shape[1]-windowW,10):
                window=tempIm[j:j+windowH,i:i+windowW]
                window=cv2.resize(window,(96,130))
                des=getDescriptors(window,cellSize,blockSize,True)
                label=classifier.predict(des)
                if(label==1):
                 #   cord.append([i,j,windowW*((1 / ratio) ** l),windowH*((1 / ratio) ** l)])
                    cord.append([i-padSize,j-padSize,windowW,windowH])
                    # realI1=int(i*(1/ratio)**l)
                    # realJ1=int(j*(1/ratio)**l)
                    # realI2 = int(i * (1 / ratio) ** l+windowW*(1 / ratio) ** l)
                    # realJ2 = int(j * (1 / ratio) ** l+windowH*(1 / ratio) ** l)
                    #
                   # cv2.rectangle(resTemp,(realI1,realJ1),(realI2,realJ2),(255,0,0),2)
                    cv2.rectangle(resTemp, (i, j), (i+windowW, j+windowH), (255, 0, 0), 2)

        finalCords.extend(nms(im,cord,ratio*0.35,False))
        print("22222222   ")
        print(finalCords)
        # tempIm = cv2.GaussianBlur(im.copy(), (1, 1), 0)
        # tempIm = cv2.resize(tempIm, (int(im.shape[1] * (ratio ** l)), int(im.shape[0] * (ratio ** l))))

    cv2.imwrite('resT.jpg', resTemp)

    return finalCords


def FaceDetector(im):



# main classifier to find faces
    precalculatedClassifier=pickle.load(open('classifier.sav','rb'))

# temporary classifier to find faces
    #precalculatedClassifier=pickle.load(open('tempClassifier.sav','rb'))

    cord=findFaces(im,precalculatedClassifier,cellSize,blockSize,100)
    faces=nms(im,cord,-1,True)
    return faces



#non-maximum supression

def nms(im,cordination,thresh,shouldDraw):
    res = im.copy()
    finalCord=[]
    print("thresh    "+str(thresh))
    if (thresh == -1):
        thresh = 95
    else:
        thresh = 100 * thresh

    while(len(cordination)>0):
        q=[]
        curr=cordination.pop()
        q.append(curr)
        curMaxSize=curr[2]
        curMax=curr




        for elem in cordination.copy():
            dist = np.sqrt((curr[0]-elem[0])**2+(curr[1]-elem[1])**2)
          #  print("hhhhh",len(cordination),dist,curr,elem)
          #  print(cordination)
            if(dist<thresh):
                cordination.remove(elem)
                q.append(elem)
                if(elem[2]>curMaxSize):
                    curMaxSize=elem[2]
                    curMax=elem

       # avg=np.uint32(np.mean(q,axis=0))
        avg=np.int32(curMax)
        finalCord.append(avg.tolist())
        print(thresh,int(thresh),str(int(thresh)))
        cv2.rectangle(res,(avg[0],avg[1]),(avg[0]+avg[2],avg[1]+avg[3]),(255,0,0),2)

    #cv2.imwrite('temp_'+str(int(thresh))+'.jpg',res)
    if(shouldDraw):
        return res

    print("111111111    ")
    print(finalCord)
    return finalCord




#only run once


os.mkdir('data')
os.mkdir('data/positive')
os.mkdir('data/negative')
normalizeData('lfw','positive')
normalizeData('256_ObjectCategories','negative')

#general part
trainPos,testPos,verPos=allocateData('positive',500,100)
trainNeg,testNeg,verNeg=allocateData('negative',3000,100)
#


#part 1

classifier,cellSize,blockSize,kernel,C=findBestParameters(trainPos,trainNeg,verPos,verNeg)

#saving for later use (so that I won't have to train the model every time)
pickle.dump(classifier,open('classifier.sav','wb'))

print(cellSize,blockSize,kernel,C)
testPosDes=getDescriptors(testPos,cellSize,blockSize,False)
testNegDes=getDescriptors(testNeg,cellSize,blockSize,False)
testDes = testPosDes.copy()
testDes.extend(testNegDes)
testLabels = [1] * len(testPosDes)
testLabels.extend([0] * len(testNegDes))

predLabels = classifier.predict(testDes)
score=classifier.decision_function(testDes)
#fpr,tpr,_=roc_curve(testLabels,score)
#precision,recall,_=precision_recall_curve(testLabels,score)
AP_score=average_precision_score(testLabels,score)

# plt.plot(fpr,tpr,color="navy")
# plt.xlim([0.0, 1.05])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
ROC_curve=plot_roc_curve(classifier,testDes,testLabels)
plt.savefig("res1.jpg")

AP_curve=plot_precision_recall_curve(classifier,testDes,testLabels)
plt.savefig("res2.jpg")
print("AP score :"+ str(AP_score))



accuracy=metrics.accuracy_score(testLabels, predLabels)
print(  str(100 * accuracy) + ' %')


# Part 2

im3=cv2.imread('Esteghlal.jpg')
im4=cv2.imread('Persepolis.jpg')
im5=cv2.imread('Melli.jpg')

#a temporary classifier to find faces
# precalculatedClassifier=getClassifier(trainPos,trainNeg,cellSize,blockSize,kernel,C)
# pickle.dump(precalculatedClassifier,open('tempClassifier.sav','wb'))



res3=FaceDetector(im3)
res4=FaceDetector(im4)
res5=FaceDetector(im5)



#
cv2.imwrite('res3.jpg',res3)
cv2.imwrite('res4.jpg',res4)
cv2.imwrite('res5.jpg',res5)





