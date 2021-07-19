import numpy as np
import cv2
import random


class Ransac:

    def __init__(self,im1,im2,kp1,kp2,cor,threshold):
        self.im1=im1
        self.im2=im2
        self.kp1=kp1
        self.kp2=kp2
        self.cor=cor
        self.threshold=threshold
        self.n=np.inf
        self.p=0.99


    def run(self):

        counter=0
        h=None
        inliers=None
        inliersCount=0
        while(counter<self.n):

            currentH=self.calculateRandomHomography()
            counter=counter+1
            currentInliers=self.getInliers(currentH)
            if(len(currentInliers)>inliersCount):
                h=currentH
                inliers=currentInliers
                inliersCount=len(currentInliers)

            self.n=self.updateN(len(currentInliers))

        finalH=self.calculateHomograpgyFromAllPoints(inliers)
        print(counter)
        return finalH
        #return self.warp(inliers,H)


    def computeNormalizationMatrix(self,points):



        sumX=0
        sumY=0
        sumXprim=0
        sumYprim=0
        for p in points:
            sumX+=(self.kp1[p.queryIdx].pt)[0]
            sumY+=(self.kp1[p.queryIdx].pt)[1]
            sumXprim += (self.kp2[p.trainIdx].pt)[0]
            sumYprim += (self.kp2[p.trainIdx].pt)[1]
        meanX=sumX/len(points)
        meanY=sumY/len(points)

        meanXprim = sumXprim / len(points)
        meanYprim = sumYprim / len(points)

        denom1=0
        denom2=0
        for p in points:
            denom1+=np.sqrt(((self.kp1[p.queryIdx].pt)[0]-meanX)**2+((self.kp1[p.queryIdx].pt)[1]-meanY)**2)
            denom2+=np.sqrt(((self.kp2[p.trainIdx].pt)[0]-meanXprim)**2+((self.kp2[p.trainIdx].pt)[1]-meanYprim)**2)

        s1=len(points)*np.sqrt(2)/denom1
        s2 = len(points) * np.sqrt(2) / denom2

        T=np.array([[s1,0,-s1*meanX],[0,s1,-s1*meanY],[0,0,1]])
        Tprim = np.array([[s2, 0, -s2 * meanXprim], [0, s2, -s2 * meanYprim], [0, 0, 1]])

        return T,Tprim




    def calculateRandomHomography(self):


        p1 = random.choice(self.cor)
        p2 = random.choice(self.cor)
        p3 = random.choice(self.cor)
        p4 = random.choice(self.cor)


        points=[p1,p2,p3,p4]
        #T,Tprim = self.computeNormalizationMatrix(points)


        A=np.empty((8,9),dtype=np.float32)

        for ind,p in enumerate(points):
            x=(self.kp1[p.queryIdx].pt)[0]
            y = (self.kp1[p.queryIdx].pt)[1]
            # n1=np.matmul(T,np.array([[x],[y],[1]]))
            # x=n1[0]
            # y=n1[1]

            xprim = (self.kp2[p.trainIdx].pt)[0]
            yprim = (self.kp2[p.trainIdx].pt)[1]
            # n2 = np.matmul(Tprim, np.array([[xprim], [yprim], [1]]))
            # xprim=n2[0]
            # yprim=n2[1]


            A[2*ind,:]=np.array([[0,0,0,-x,-y,-1,x*yprim,y*yprim,yprim]])
            A[2*ind+1, :] = np.array([[ x, y, 1,0,0,0, -x * xprim, -y * xprim, -xprim]])

            # A[2 * ind, :] = np.array([[-x, -y, -1,0, 0, 0,  x * yprim, y * yprim, yprim]])
            # A[2 * ind + 1, :] = np.array([[ 0, 0, 0,-x, -y, -1, x * xprim, y * xprim, xprim]])

        u,s,vt=np.linalg.svd(A)
        h=vt[-1,:].reshape(3,3)

        #h=np.matmul(np.matmul(np.linalg.inv(Tprim),h),T)
        return h


    def getInliers(self,h):

        inliers=[]

        for c in self.cor:
            x = (self.kp1[c.queryIdx].pt)[0]
            y = (self.kp1[c.queryIdx].pt)[1]

            truex = (self.kp2[c.trainIdx].pt)[0]
            truey = (self.kp2[c.trainIdx].pt)[1]
            hcordination=np.matmul(h,np.array([[x],[y],[1]]))
            hx=hcordination[0]/hcordination[2]
            hy=hcordination[1]/hcordination[2]

            hInvCordination=np.matmul(np.linalg.inv(h),np.array([[truex],[truey],[1]]))

            hInvx = hInvCordination[0] / hInvCordination[2]
            hInvy = hInvCordination[1] / hInvCordination[2]

            #print(hcordination,hx,truex,hy,truey)

            dist=np.sqrt((truex-hx)**2+(truey-hy)**2)+np.sqrt((x-hInvx)**2+(y-hInvy)**2)
            if(dist<self.threshold):
                inliers.append(c)
        return inliers

    def updateN(self,count):

        if(count==0):
            return np.inf

        return (np.log(1-self.p))/np.log(1-(count/len(self.cor))**4)

    def calculateHomograpgyFromAllPoints(self,inliers):

        A=np.empty((len(inliers)*2,9),dtype=np.float32)

        for ind,p in enumerate(inliers):
            x=(self.kp1[p.queryIdx].pt)[0]
            y = (self.kp1[p.queryIdx].pt)[1]
            xprim = (self.kp2[p.trainIdx].pt)[0]
            yprim = (self.kp2[p.trainIdx].pt)[1]

            A[2*ind,:]=np.array([[0,0,0,-x,-y,-1,x*yprim,y*yprim,yprim]])
            A[2*ind+1, :] = np.array([[ x, y, 1,0,0,0, -x * xprim, -y * xprim, -xprim]])

            # A[2 * ind, :] = np.array([[-x, -y, -1,0, 0, 0,  x * yprim, y * yprim, yprim]])
            # A[2 * ind + 1, :] = np.array([[ 0, 0, 0,-x, -y, -1, x * xprim, y * xprim, xprim]])

        u,s,vt=np.linalg.svd(A)
        h=vt[-1,:].reshape(3,3)
        return h



im1=cv2.imread("im03.jpg")
im2=cv2.imread("im04.jpg")
sift=cv2.SIFT_create()
kp1=sift.detect(im1,None)
kp2=sift.detect(im2,None)

des1=sift.compute(im1,kp1)
des2=sift.compute(im2,kp2)
match=cv2.BFMatcher().knnMatch(des1[1],des2[1],k=2)
matches=[]



listlessMatches=[]

for p1,p2 in match:
    point1 = kp1[p1.queryIdx].pt
    point2 = kp2[p1.trainIdx].pt

    if p1.distance<0.7*p2.distance:
        matches.append(p1)



ransac=Ransac(im1,im2,kp1,kp2,matches,50)
h=ransac.run()
print(h)

cv2.imwrite('res20.jpg',cv2.warpPerspective(im2,np.linalg.inv(h),(5000,1800)))








