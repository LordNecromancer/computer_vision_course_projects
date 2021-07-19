
import cv2
import numpy as np
import random


src=cv2.imread('01.JPG')
dst=cv2.imread('02.JPG')
sift = cv2.SIFT_create()

dstKp, dstDes = sift.detectAndCompute(dst, None)
srcKp, srcDes = sift.detectAndCompute(src, None)

match = cv2.BFMatcher().knnMatch(srcDes, dstDes, k=2)
#FLANN_INDEX_KDTREE = 1
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5),dict(checks=50))
# match = flann.knnMatch(srcDes,dstDes,k=2)
print(len(match))
matches = []

srcKpArray = []
dstKpArray = []

listlessMatches = []

for p1, p2 in match:
    point1 = srcKp[p1.queryIdx].pt
    point2 = dstKp[p1.trainIdx].pt

    if p1.distance < 0.75* p2.distance :
        matches.append([p1])
        listlessMatches.append(p1)

        srcKpArray.append(point1)
        dstKpArray.append(point2)
print(len(matches))
srcKpArray=np.int32(srcKpArray)
dstKpArray=np.int32(dstKpArray)
F, mask=cv2.findFundamentalMat(srcKpArray,dstKpArray,cv2.FM_RANSAC)
print("$$$$$$")
print(F)
srcInliers=srcKpArray[mask.ravel()==1]
dstInliers=dstKpArray[mask.ravel()==1]

srcOutliers=srcKpArray[mask.ravel()==0]
dstOutliers=dstKpArray[mask.ravel()==0]

temp1=src.copy()
temp2=dst.copy()

for i in range(len(srcKpArray)):
    s=srcKpArray[i]
    d=dstKpArray[i]

    if(s in srcInliers):
        cv2.circle(temp1,(srcKpArray[i][0],srcKpArray[i][1]),6,(0,255,0),-1)
        cv2.circle(temp2,(dstKpArray[i][0],dstKpArray[i][1]),6,(0,255,0),-1)
    else:
        cv2.circle(temp1, (srcKpArray[i][0], srcKpArray[i][1]), 6, (0, 0, 255), -1)
        cv2.circle(temp2, (dstKpArray[i][0], dstKpArray[i][1]), 6, (0, 0, 255), -1)

temp2=cv2.resize(temp2,(temp1.shape[1],temp1.shape[0]))
concat=np.hstack((temp1,temp2))
cv2.imwrite('res05.jpg',concat)

u,s,vt=np.linalg.svd(F)
#e2=np.linalg.solve(np.transpose(F),np.array([[0],[0],[0]],dtype=np.float64))
print(u,s,vt)
e1=vt[2]
e2=u[:,2]
e1=e1/e1[2]
e2=e2/e2[2]

print(e1,e2)

ep1=np.full((6000,27000,3),255,dtype=np.uint8)
cv2.circle(ep1,(200,200),25,(255,0,0),-1)
ep1[int(-e1[1])+200:int(-e1[1])+200+src.shape[0],int(-e1[0])+200:int(-e1[0])+200+src.shape[1],:]=src

cv2.imwrite('res06.jpg',ep1)

ep2=np.full((6000,27000,3),255,dtype=np.uint8)
cv2.circle(ep2,(200+int(e2[0]),200),25,(255,0,0),-1)
ep2[int(-e2[1])+200:int(-e2[1])+200+dst.shape[0],200:200+dst.shape[1],:]=dst

cv2.imwrite('res07.jpg',ep2)

#srcKp10=srcInliers[:10]
#dstKp10=dstInliers[:10]

temp1=src.copy()
temp2=dst.copy()

randomInd=[]
randomColor=[]
while (len(randomInd)<10):
    ind=random.randint(0,len(srcInliers)-1)
    if(ind not in randomInd):
        b=random.randint(0,256)
        g=random.randint(0,256)
        r=random.randint(0,256)
        randomInd.append(ind)
        randomColor.append((b,g,r))


for k,i in enumerate(randomInd):
    skp=np.array([[srcInliers[i][0]],[srcInliers[i][1]],[1]])
    dkp=np.array([[dstInliers[i][0]],[dstInliers[i][1]],[1]])

    l2=np.matmul(F,skp)
    l1=np.matmul(np.transpose(F),dkp)

    m1=-l1[0]/l1[1]
    b1=-l1[2]/l1[1]

    m2 = -l2[0] / l2[1]
    b2 = -l2[2] / l2[1]

    cv2.circle(temp1,(srcInliers[i][0],srcInliers[i][1]),10,(0,0,255),-1)
    cv2.circle(temp2,(dstInliers[i][0],dstInliers[i][1]),10,(0,0,255),-1)


    for x in range(0,src.shape[1]-1):
        y=int(m1*x+b1)
        temp1[y-1:y+1,x,:]=randomColor[k]


    for x in range(0, dst.shape[1] - 1):
        y = int(m2 * x + b2)
        temp2[ y - 1:y + 1,x, :] = randomColor[k]

temp2=cv2.resize(temp2,(temp1.shape[1],temp1.shape[0]))
concat2=np.hstack((temp1,temp2))
cv2.imwrite('res08.jpg',concat2)

