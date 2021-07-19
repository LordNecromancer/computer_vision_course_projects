import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

names=['Bedroom','Coast','Forest','Highway','Industrial','Inside_City','Kitchen','Livingroom',
       'Mountain','Office','Open_Country','Store','Street','Suburb','Tall_Building']
labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]



def readData(dir,d):
    feature=[]
    label=[]
    for l,subDir in enumerate(names):
        ims=os.listdir('./data/'+dir+'/'+subDir)
        for im in ims:
            image=cv2.imread('./data/'+dir+'/'+subDir+'/'+im)
            image=cv2.GaussianBlur(image,(25,25),0)
            image=cv2.resize(image,(d,d)).reshape(-1).tolist()
            feature.append(image)
            label.append(l)
    return feature,label
def classifyByKNN(k,d):
    trainFeature,trainLabel=readData('train',d)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainFeature,trainLabel)
    testFeature,testLabel=readData('test',d)
    predTestLabels=knn.predict(testFeature)
    print('knn with k='+str(k)+'  d='+str(d)+'   :'+str(100*metrics.accuracy_score(testLabel,predTestLabels))+ ' %')


def getFeatures(dir):
    sift = cv2.SIFT_create()
    allDes=[]
    labels=[]
    featuresByImage={}

    for l, subDir in enumerate(names):
        ims = os.listdir('./data/' + dir + '/' + subDir)
        for im in ims:
            image = cv2.imread('./data/' + dir + '/' + subDir + '/' + im)
            kps, des = sift.detectAndCompute(image, None)
            featuresByImage[dir + '/' + subDir + '/' + im]=des
            allDes.extend(des)
            labels.append(l)
    return allDes,featuresByImage,labels

def getVisualWords(k,des):
    kmeans=KMeans(k)
    return kmeans,kmeans.fit(des).cluster_centers_


def getHistogramFeatures(kmeans,visualWords,featuresByImage):
    histograms=[]
    for key,features in featuresByImage.items():
        histogram=[0]*len(visualWords)
        for feature in features:
            center=kmeans.predict([feature])[0]

            histogram[center]+=1
        histograms.append(histogram)
    return histograms
def classifyByBOW(nv):
    trainDes,trainFeaturesByImage,trainLabels=getFeatures('train')
    kmeans,visualWords=getVisualWords(nv,trainDes)
    trainHistograms=getHistogramFeatures(kmeans,visualWords,trainFeaturesByImage)
    #ks=[1,5,10,15,20,25,30]
    ks=[25]
    testDes, testFeaturesByImage, testLabels = getFeatures('test')

    testHistograms = getHistogramFeatures(kmeans, visualWords, testFeaturesByImage)
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(trainHistograms, trainLabels)
        predTestLabels = knn.predict(testHistograms)
        print("bow   with "+str(nv)+'   visual words and K :  '+str(k)+"   = "+str(100*metrics.accuracy_score(testLabels,predTestLabels))+' %')


def classifySVM(nv):
    trainDes, trainFeaturesByImage, trainLabels = getFeatures('train')
    kmeans, visualWords = getVisualWords(nv, trainDes)
    trainHistograms = getHistogramFeatures(kmeans, visualWords, trainFeaturesByImage)

    testDes,testFeaturesByImage,testLabels=getFeatures('test')
    testHistograms=getHistogramFeatures(kmeans,visualWords,testFeaturesByImage)
    classifier=make_pipeline(StandardScaler(),SVC(gamma='auto'))
    classifier.fit(trainHistograms,trainLabels)
    predTestLabels = classifier.predict(testHistograms)
    plot_confusion_matrix(classifier,testHistograms,testLabels)
    plt.savefig('res09.jpg')
    print("svm   with " + str(nv) + "   = " + str( 100*metrics.accuracy_score(testLabels, predTestLabels))+' %')


classifyByKNN(1,14)
# classifyByKNN(1,16)
# classifyByKNN(1,18)
#
# classifyByKNN(3,14)
# classifyByKNN(3,16)
# classifyByKNN(3,18)
#
# classifyByKNN(10,14)
# classifyByKNN(10,16)
# classifyByKNN(10,18)
#
#
#classifyByBOW(5)
classifyByBOW(50)
# classifyByBOW(75)
#classifyByBOW(100)

#classifySVM(5)
#classifySVM(50)
#classifySVM(75)
classifySVM(100)