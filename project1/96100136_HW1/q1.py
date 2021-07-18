import cv2 
import numpy as np
import math
import random
i1=cv2.imread("im01.jpg")
i2=cv2.imread("im02.jpg")
# i1=cv2.resize(i1,(i1.shape[1]//2,i1.shape[0]//2))
# i2=cv2.resize(i2,(i2.shape[1]//2,i2.shape[0]//2))


greyI1= cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
greyI2= cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

k=0.05
threshold=2800000000
corrsepondThreshold=1.14

def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def nonMaximalSupression(image,windowSize):
    image=image.copy()
    # mask=image.copy()
    # mask[mask>60]=255
    # mask[mask<=70]=0
    # num,labels=cv2.connectedComponents(mask,connectivity=8)
    # print(num)
    # #print(labels[labels==1])
    # res=np.zeros(image.shape)
    # for i in range(1,num):
    #     #print(image[labels==i])
    #     m=np.max(image[labels==i])
    #     #print(m)
    #     y,x=np.where((labels==i)& (image==m))
    #     #print("booooooooooooooo")
    #     #print(ind)
    #     res[y[0],x[0]]=255

    # print("ggggggggg")
    # print(labels)
    # image=res
    for i in range(windowSize//2,image.shape[1]-windowSize//2,windowSize//2):
        for j in range(windowSize//2,image.shape[0]-windowSize//2,windowSize//2):
            window=image[j-windowSize//2:j+windowSize//2,i-windowSize//2:i+windowSize//2]
            #w=im[j-windowSize//2:j+windowSize//2+1,i-windowSize//2:i+windowSize//2+1]
            maxInd=np.unravel_index(np.argmax(window, axis=None), window.shape)


            if(window[maxInd]>0) :
                window[:,:] = 0
                #im[j-windowSize//2:j+windowSize//2+1,i-windowSize//2:i+windowSize//2+1]=imm[j-windowSize//2:j+windowSize//2+1,i-windowSize//2:i+windowSize//2+1].copy()

                window[maxInd]=255
               # cv2.circle(w,maxInd,1,(0,0,255,2))
            else:
                window[:,:]=0
                #im[j-windowSize//2:j+windowSize//2+1,i-windowSize//2:i+windowSize//2+1]=imm[j-windowSize//2:j+windowSize//2+1,i-windowSize//2:i+windowSize//2+1].copy()

    return  image


def write(name,file):
    cv2.imwrite(name,file)


def createFeatureVector(image,mask,n):
    features={}
    indices=np.nonzero(mask)
    #print(len(indices[0]))
    image=image.copy()

    for t in range(len(indices[0])):
        v=[]

        i=indices[1][t]
        j=indices[0][t]
        window=image[j-n//2:j+n//2+1,i-n//2:i+n//2+1]
       # print(j,i,j-n//2,j+n//2,i-n//2,i+n//2)
        v=window.reshape(-1,3)

        features[(j,i)]=v
    return features

def getDistance(p1,p2):
   # print(p1,p2,p2.reshape(-1))
    return np.sum((p1.reshape(-1)-p2.reshape(-1))**2)

def generateDistanceMatrix(f1,f2):
    dists=np.empty((len(f1),len(f2)),np.float16)
    for ind1,p1 in enumerate(f1.values()):
        for ind2,p2 in enumerate(f2.values()):
            dist=getDistance(p1,p2)
            # if(dist<1000):
            #     print(dist)
            #print(dist,ind1,ind2)
            dists[ind1,ind2]=dist

    return dists



def findCorrospondingPoints(distanceMatrix):
    corr1={}
    corr2 = {}
    corrFinal={}
    dists=distanceMatrix.copy()
    #print(dists.shape)
    for r in range(dists.shape[0]):
        d1=np.min(dists[r,:])
        dists[r,:][np.where(d1==dists[r,:])]=math.inf
        d2=np.min(dists[r,:])
        #print("1   :   " + str(d2 / d1))
        if(d2/d1>corrsepondThreshold):
           # print(np.where(d1==distanceMatrix[r,:])[0][0] )
            if(len(corr1.values())>0 and np.where(d1==distanceMatrix[r,:])[0][0] in corr1.values() ):
                corr2[np.where(d1==distanceMatrix[r,:])[0][0]]=False
                corr1[r]=False

            else:
                corr1[r]=np.where(d1==distanceMatrix[r,:])[0][0]
    dists=distanceMatrix.copy()

    for c in range(dists.shape[1]):
        d1=np.min(dists[:,c])
        dists[:,c][np.where(d1==dists[:,c])[0][0]]=math.inf
        d2=np.min(dists[:,c])
        #print("2   :   "+str(d2/d1))
        if(d2/d1>corrsepondThreshold):
            if (len(corr2.values())>0 and np.where(d1 == distanceMatrix[:,c])[0][0] in corr2.values()):
                corr1[np.where(d1 == distanceMatrix[:,c])[0][0]] = False
                corr2[c] = False
            else:
                corr2[c]=np.where(d1==distanceMatrix[:,c])[0][0]
                if(np.where(d1==distanceMatrix[:,c])[0][0] in corr1 and corr1[np.where(d1==distanceMatrix[:,c])[0][0]] is not False  and corr1[np.where(d1==distanceMatrix[:,c])[0][0]]==c):
                    corrFinal[np.where(d1==distanceMatrix[:,c])[0][0]]=c


    #print(corr1,corr2)
    return corrFinal

def createFinalResult(im1,im2,corr,m1,m2):
    res=np.hstack((im1,im2))
    indices1=np.nonzero(m1)
    indices2=np.nonzero(m2)

    for cor in corr:
        i1=indices1[1][cor]
        i2=indices2[1][corr[cor]]
        j1=indices1[0][cor]
        j2=indices2[0][corr[cor]]
        p1=(i1,j1)
        p2=(i2+im1.shape[1],j2)
        randomB=random.randint(0,256)
        randomG=random.randint(0,256)
        randomR=random.randint(0,256)
        cv2.line(res,p1,p2,(randomB,randomG,randomR),2)
        cv2.circle(im1,(i1,j1),5,(randomB,randomG,randomR),2)
        cv2.circle(im2,(i2,j2),5,(randomB,randomG,randomR),2)

    return res

i1x=cv2.Sobel(i1,cv2.CV_64F,1,0,ksize = -1)
i1y=cv2.Sobel(i1,cv2.CV_64F,0,1,ksize = -1)

i2x=cv2.Sobel(i2,cv2.CV_64F,1,0,ksize = -1)
i2y=cv2.Sobel(i2,cv2.CV_64F,0,1,ksize = -1)

i1xp2=i1x*i1x
i1yp2=i1y*i1y
i1xy=i1x*i1y

i2xp2=i2x*i2x
i2yp2=i2y*i2y
i2xy=i2x*i2y

gradientMag1=np.sqrt(i1xp2+i1yp2)
gradientMag2=np.sqrt(i2xp2+i2yp2)

w=9
sigma=1.8
s1p2x=cv2.GaussianBlur(i1xp2,(w,w),sigma)
s1p2y=cv2.GaussianBlur(i1yp2,(w,w),sigma)
s1xy=cv2.GaussianBlur(i1xy,(w,w),sigma)

s2p2x=cv2.GaussianBlur(i2xp2,(w,w),sigma)
s2p2y=cv2.GaussianBlur(i2yp2,(w,w),sigma)
s2xy=cv2.GaussianBlur(i2xy,(w,w),sigma)

det1=s1p2x*s1p2y-s1xy*s1xy
trace1=s1p2x+s1p2y

det2=s2p2x*s2p2y-s2xy*s2xy
trace2=s2p2x+s2p2y

r1=det1-k*trace1**2
r2=det2-k*trace2**2



thresholdR1=normalize(r1.copy())
thresholdR2=normalize(r2.copy())




thresholdR1[r1<threshold]=0
thresholdR2[r2<threshold]=0
# thresholdR1[r1>=threshold]=255
# thresholdR2[r2>=threshold]=255000000
write('res05_thresh.jpg',thresholdR1)
write('res06_thresh.jpg',thresholdR2)

thresholdR1Grey=(thresholdR1[:,:,0]+thresholdR1[:,:,1]+thresholdR1[:,:,2])//3
thresholdR2Grey=(thresholdR2[:,:,0]+thresholdR2[:,:,1]+thresholdR2[:,:,2])//3




thresholdR1=nonMaximalSupression(thresholdR1Grey,20)
thresholdR2=nonMaximalSupression(thresholdR2Grey,20)

temp1=i1.copy()
temp2=i2.copy()
ind=np.where(thresholdR1>0)
for num in range(len(ind[0])):
    cv2.circle(temp1,(ind[1][num],ind[0][num]),3,(0,0,255),-1)
write('res07_harris.jpg',temp1)
ind=np.where(thresholdR2>0)
for num in range(len(ind[0])):
    cv2.circle(temp2,(ind[1][num],ind[0][num]),3,(0,0,255),-1)
write('res08_harris.jpg',temp2)

#write('i1.jpg',thresholdR1[1])
#write('i2.jpg',thresholdR2[1])

features1=createFeatureVector(i1,thresholdR1,9)
features2=createFeatureVector(i2,thresholdR2,9)

print(len(features1))

distMatrix=generateDistanceMatrix(features1,features2)
corrospondence=findCorrospondingPoints(distMatrix)
res=createFinalResult(i1,i2,corrospondence,thresholdR1,thresholdR2)
write('res09_corres.jpg',i1)
write('res10_corres.jpg',i2)

write('res11.jpg',res)
#print(corrospondence)








cv2.imwrite('res01_grad.jpg',normalize(gradientMag1))
cv2.imwrite('res02_grad.jpg',normalize(gradientMag2))
cv2.imwrite('res03_score.jpg',normalize(r1))
cv2.imwrite('res04_score.jpg',normalize(r2))
