import cv2
import numpy as np




class Blend:

    def __init__(self,d,n):
        self.d=d
        self.n=n
        self.leftGaussianPyr = []
        self.leftLaplacianPyr = []
        self.rightGaussianPyr = []
        self.rightLaplacianPyr = []
        self.t=0




    def blend(self,left,prev, right):
        self.leftGaussianPyr = []
        self.leftLaplacianPyr = []
        self.rightGaussianPyr = []
        self.rightLaplacianPyr = []
        self.t=self.t+1

        ll = left.shape[0] % (2 ** self.n)
        rr = left.shape[1] % (2 ** self.n)
        left = cv2.resize(left, (left.shape[1] - rr, left.shape[0] - ll))
        right = cv2.resize(right, (left.shape[1] , left.shape[0] ))
        prev = cv2.resize(prev, (left.shape[1] , left.shape[0] ))




        self.createPyramids(left, self.n, right)
        res = self.collapsePyramids(self.d,prev,right)
        return res


    def getMask(self,left,right):
        maskL=np.zeros((left.shape[0],left.shape[1]),dtype=np.float32)
        maskR=np.zeros((left.shape[0],left.shape[1]),dtype=np.float32)
        mask1=np.zeros((left.shape[0],left.shape[1],3),dtype=np.float32)
        mask2=np.zeros((left.shape[0],left.shape[1],3),dtype=np.float32)
        mask=np.zeros((left.shape[0],left.shape[1],3),dtype=np.float32)




        l=left[:, :, 0] + left[:, :, 1] + left[:, :, 2]
        r=right[:, :, 0] + right[:, :, 1] + right[:, :, 2]

        maskL[l>0]=1
        maskR[r>0]=1
        contoursL,hierL=cv2.findContours(np.uint8(maskL),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursR,hierR=cv2.findContours(np.uint8(maskR),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # maskL[:,:]=0
        # maskR[:,:]=0
        # mask2[:,:,0]=maskL
        # mask2[:,:,1]=maskL
        # mask2[:,:,2]=maskL
        mask2=cv2.drawContours(mask2,contoursL,0,(1,1,1),2)
        # mask1[:,:,0]=maskR
        # mask1[:,:,1]=maskR
        # mask1[:,:,2]=maskR
        mask1=cv2.drawContours(mask1,contoursR,0,(1,1,1),2)

        sum=mask1+mask2
        mp=sum.copy()
        sum[sum!=2]=0
        sum[sum==2]=1
        mpp=sum.copy()
        pt1,pt2=self.getTwoPoints(sum)
        if(pt1[0]!=pt2[0]):
            m=(pt2[1]-pt1[1])/(pt2[0]-pt1[0])
            print(m)
            x0=pt1[0]
            y0=pt1[1]

            point1=[(0-y0+m*x0)//m,0]
            point2=[(sum.shape[0]-y0+m*x0)//m,sum.shape[0]]
        else:
            point1 = [pt1[0], 0]
            point2 = [pt2[0], sum.shape[0]]

        #cv2.line(mask1,point1,point2,(1,1,1),3)
        print(point1,point2)


        #mask[:,:,:]=0
        if(point1[1]<point2[1]):
            points=np.array([[0,0],point1,point2,[0,sum.shape[0]]],dtype=np.int32)
        else:
            points=np.array([[0,0],point2,point1,[0,sum.shape[0]]],dtype=np.int32)

        mask=cv2.fillConvexPoly(mask,points,(1,1,1))


       #  maskL[l>0]=150
       #  maskR[r>0]=50
       #  sum=maskL+maskR
       #  cv2.imwrite('s.jpg',sum)
       #  sumIntersection=sum.copy()
       #
       #
       #  sum[sum==150]=1
       #  sum[sum==200]=1
       #  sum[sum==50]=0
       # # sum=cv2.morphologyEx(sum, cv2.MORPH_CLOSE, np.ones((13,13),np.uint8))
       #  #sum=cv2.erode(sum,np.ones((25,25),np.uint8),iterations = 1)
       #
       #
       #
       #  sumIntersection[sumIntersection==150]=0
       #  sumIntersection[sumIntersection==50]=0
       #  sumIntersection[sumIntersection==200]=1
       #
       #  sumIntersection = cv2.erode(sumIntersection, np.ones((5,5),np.uint8), iterations=1)
       #  contours,hier=cv2.findContours(np.uint8(sumIntersection),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
       #  sumIntersection[:,:]=0
       #  print(np.dtype(sumIntersection))
       #  sumIntersection=np.astype(np.float32)
       #  #sumIntersection=cv2.drawContours(sumIntersection,)
       #
       #  #sumIntersection = cv2.morphologyEx(sumIntersection, cv2.MORPH_GRADIENT, np.ones((2,2),np.uint8))
       #
       #
       #  mask2[:,:,0]=sumIntersection
       #  mask2[:,:,1]=sumIntersection
       #  mask2[:,:,2]=sumIntersection
       #  mask2=cv2.drawContours(mask2,contours,0,(1,1,1),2)
       #  sum[sumIntersection>0]=0
       #  mask1[:, :, 0] = sum
       #  mask1[:, :, 1] = sum
       #  mask1[:, :, 2] = sum










        mm=mask.copy()
        mm[mm==1]=255
        mp[mp>0]=255
        mpp[mpp>0]=255







        cv2.imwrite('masL.jpg',maskL)
        cv2.imwrite('masR.jpg',maskR)
        cv2.imwrite('mp.jpg',mp)
        cv2.imwrite('mpp.jpg',mpp)



        cv2.imwrite('mm.jpg',mm)





        return mask



    def getTwoPoints(self,mask):
        indices=(mask>0).nonzero()
        points=np.vstack((indices[1],indices[0]))
        points=np.transpose(points).reshape((-1,2))
        isFound,labels,centers=cv2.kmeans(np.float32(points),2,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),20,cv2.KMEANS_RANDOM_CENTERS)
        print("center",centers)

        pt1=np.float32(centers[0])
        pt2=np.float32(centers[1])
        print(pt1)


        return pt1,pt2





    def collapsePyramids(self,d,prev,right):
        lt = self.leftGaussianPyr.pop()
        rt = self.rightGaussianPyr.pop()
        maskMain=np.float32(self.getMask(prev.copy(),right.copy()))
        #res=self.merge(prev,right,maskMain)
        maskTemp=cv2.resize(maskMain,(lt.shape[1],lt.shape[0]))

        mask = cv2.GaussianBlur(maskTemp, (d, d),0)
        print(mask.dtype)
        #mask1[mask2>0]=0
        res = lt*mask  + rt*(1-mask)
        m=mask.copy()
        m=m*250
        #m[(mask<1)& (mask > 0)] = 30
        cv2.imwrite('two/lft'+str(self.t)+'_'+str(1)+'.jpg',lt)
        cv2.imwrite('two/right'+str(self.t)+'_'+str(1)+'.jpg',rt)
        cv2.imwrite('two/mask'+str(self.t)+'_'+str(1)+'.jpg',m)
        cv2.imwrite('two/res'+str(self.t)+'_'+str(1)+'.jpg',res)
        cv2.imwrite('two/ltm'+str(self.t)+'_'+str(1)+'.jpg',lt*mask)



        #res[mask2>0]=lt[mask2>0] * (1-mask2[mask2>0])  + rt[mask2>0]  *mask2[mask2>0]




        #self.leftLaplacianPyr.pop()
        #self.rightLaplacianPyr.pop()
        c=2

        while (len(self.leftLaplacianPyr) > 0):
            lt = self.leftLaplacianPyr.pop()
            rt = self.rightLaplacianPyr.pop()
            maskTemp=cv2.resize(maskMain, (lt.shape[1] , lt.shape[0]))


            #mask = self.getMask(self.leftGaussianPyr.pop(),self.rightGaussianPyr.pop())

            mask = cv2.GaussianBlur(maskTemp, (d, d), 0)


            res = res+(lt * mask + rt * (1 - mask))
            m = mask.copy()
            m=m*250
            cv2.imwrite('two/lft'+str(self.t)+'_' + str(c) + '.jpg', lt)
            cv2.imwrite('two/right'+str(self.t)+'_' + str(c) + '.jpg', rt)
            cv2.imwrite('two/mask'+str(self.t)+'_' + str(c) + '.jpg', m)
            cv2.imwrite('two/res'+str(self.t)+'_' + str(c) + '.jpg', res)
            cv2.imwrite('two/ltm' + str(self.t) + '_' + str(c) + '.jpg', lt * mask)

            c+=1
            #resTemp[mask2 > 0] = lt[mask2 > 0] * (1 - mask2[mask2 > 0]) + rt[mask2 > 0] * mask2[mask2 > 0]




            if(len(self.leftLaplacianPyr) > 0):

                res = cv2.resize(res, (res.shape[1] * 2, res.shape[0] * 2))


        return res

    def createPyramids(self,left, n, right):
        right = np.float32(right)
        left = np.float32(left)

        tl = cv2.GaussianBlur(left.copy(), (self.d, self.d), 0)
        tr = cv2.GaussianBlur(right.copy(), (self.d, self.d), 0)
        print(tr.dtype)
        self.leftGaussianPyr.append(tl)
        self.rightGaussianPyr.append(tr)
        self.leftLaplacianPyr.append(left - tl)
        self.rightLaplacianPyr.append(right - tr)
        left = tl
        right = tr
        for i in range(n-1):

            # creating gaussian and laplacian pyramid for left
            tempL = cv2.resize(left, (left.shape[1] // 2, left.shape[0] // 2))

            tempGaussianL = cv2.GaussianBlur(tempL.copy(), (self.d, self.d), 0)
            tempLaplacianL = tempL - tempGaussianL
            left = tempGaussianL

            self.leftGaussianPyr.append(tempGaussianL)
            self.leftLaplacianPyr.append(tempLaplacianL)

            # creating gaussian and laplacian pyramid for right

            tempR = cv2.resize(right, (right.shape[1] // 2, right.shape[0] // 2))

            tempGaussianR = cv2.GaussianBlur(tempR.copy(), (self.d, self.d), 0)
            tempLaplacianR = tempR - tempGaussianR
            right = tempGaussianR

            self.rightGaussianPyr.append(tempGaussianR)
            self.rightLaplacianPyr.append(tempLaplacianR)


def extractAndSaveFrames(video):
    for i in range(1,901):
        frameExists, frame = video.read()
        cv2.imwrite('frames/frame'+str(i)+'.jpg',frame)
        #frames.append(frame)
        # grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # greys.append(grey)


def padImage(im):
    return np.pad(im, ((750, 850), (2500, 2500), (0, 0)), mode='constant', constant_values=0)


def calculateHomography(src,dst):
    sift = cv2.SIFT_create()

    dstKp, dstDes = sift.detectAndCompute(dst, None)

    srcKp, srcDes = sift.detectAndCompute(src, None)

    match = cv2.BFMatcher().knnMatch(srcDes, dstDes, k=2)
    print(len(match))
    matches = []

    srcKpArray = []
    dstKpArray = []

    listlessMatches = []

    for p1, p2 in match:
        point1 = srcKp[p1.queryIdx].pt
        point2 = dstKp[p1.trainIdx].pt

        if p1.distance < 0.7* p2.distance:
            matches.append([p1])
            listlessMatches.append(p1)

            srcKpArray.append(point1)
            dstKpArray.append(point2)


    h, mask = cv2.findHomography(np.float32(srcKpArray).reshape(-1, 1, 2), np.float32(dstKpArray).reshape(-1, 1, 2), cv2.RANSAC, 5.0, maxIters=150)



    return h,mask

def writeMatrixToFIle(file, h):
    hh=h.copy()
    hh=hh.reshape(-1)
    string=""
    for homog in hh :
        string=string + " , "+ str(homog)
    file.write((string+"\n"))


def generateWarpedFrames(start,end, resizedNextKeyH, resizedNextKey,file):
    s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    for i in range(start, end):
        im = cv2.imread('frames/frame' + str(i) + '.jpg')
        im = padImage(im)
        imResized = resize(im, 2, 2)
        hTemp, maskTemp = calculateHomography(imResized, resizedNextKey)
        h = np.matmul(hTemp, resizedNextKeyH)
        h = np.matmul(np.matmul(s, h), np.linalg.inv(s))
        res = cv2.warpPerspective(im, h, dsize=(im.shape[1], im.shape[0]))

        writeMatrixToFIle(file, h)
        cv2.imwrite('task3/frame'+str(i)+'.jpg',res)



def createStripes(l):

    for i in range(1,901):
        print(i)
        im=cv2.imread('task3/frame' + str(i) + '.jpg')
        for k in range((im.shape[1]//l)-1):
            cv2.imwrite('stripes/stripe' + str(i) + '_'+str(k+1)+'.jpg', im[:,k*l:(k+1)*l,:])


def resize(img,w,h):
    return cv2.resize(img,(img.shape[1]//w,img.shape[0]//h))
def doTaskOne():
    im270 = cv2.imread('frames/frame270.jpg')
    im450 = cv2.imread('frames/frame450.jpg')

    im270 = padImage(im270)
    im450 = padImage(im450)

    h, mask = calculateHomography(im270,im450)

    result = im450.copy()
    warped270 = cv2.warpPerspective(im270, h, dsize=(im270.shape[1], im270.shape[0]))
    warped=warped270.copy()
    warped[result != (0, 0, 0)] = result[result != (0, 0, 0)]

    rect450=cv2.rectangle(im450,(3000,1250),(3250,1500),(0,0,255),3)
    rect270=cv2.rectangle(warped270,(3000,1250),(3250,1500),(0,0,255),3)
    rect270=cv2.warpPerspective(warped270, np.linalg.inv(h), dsize=(im270.shape[1], im270.shape[0]))

    cv2.imwrite('res01-450-rect.jpg', rect450)
    cv2.imwrite('res02-270-rect.jpg', rect270)


    cv2.imwrite('res03-270-450-panorama.jpg', warped)



def doTaskTwo():
    im90 = cv2.imread('frames/frame90.jpg')
    im270 = cv2.imread('frames/frame270.jpg')
    im450 = cv2.imread('frames/frame450.jpg')
    im630 = cv2.imread('frames/frame630.jpg')
    im810 = cv2.imread('frames/frame810.jpg')

    print("yu")
    im90 = padImage(im90)
    #im90=resize(im90,2,2)

    print("yuc")

    im270 = padImage(im270)
    #im270=resize(im270,2,2)
    im450 = padImage(im450)
   # im450=resize(im450,2,2)

    im630 = padImage(im630)
    #im630=resize(im630,2,2)

    im810 = padImage(im810)
   # im810=resize(im810,2,2)

    #
    h90, mask90 = calculateHomography(im90,im270)
    h270, mask270 = calculateHomography(im270,im450)
    h630, mask630 = calculateHomography(im630,im450)
    h810, mask810 = calculateHomography(im810,im630)
    #
    h90=np.matmul(h90,h270)
    h810=np.matmul(h810,h630)
    #
    warped90=cv2.warpPerspective(im90, h90, dsize=(im270.shape[1], im270.shape[0]))
    warped270=cv2.warpPerspective(im270, h270, dsize=(im270.shape[1], im270.shape[0]))
    warped630=cv2.warpPerspective(im630, h630, dsize=(im270.shape[1], im270.shape[0]))
    warped810=cv2.warpPerspective(im810, h810, dsize=(im270.shape[1], im270.shape[0]))

    b=Blend(19,4)

    result=b.blend(warped90,warped90,warped270)
    result=b.blend(result,warped270,im450)
    result=b.blend(result,im450,warped630)
    result=b.blend(result,warped630,warped810)

    cv2.imwrite('res04-key-frames-panorama.jpg', result)




def doTaskThree():
    im90 = cv2.imread('frames/frame90.jpg')
    im270 = cv2.imread('frames/frame270.jpg')
    im450 = cv2.imread('frames/frame450.jpg')
    im630 = cv2.imread('frames/frame630.jpg')
    im810 = cv2.imread('frames/frame810.jpg')

    im90 = padImage(im90)
    im90r=resize(im90,2,2)


    im270 = padImage(im270)
    im270r=resize(im270,2,2)
    im450 = padImage(im450)
    im450r=resize(im450,2,2)

    im630 = padImage(im630)
    im630r=resize(im630,2,2)

    im810 = padImage(im810)
    im810r=resize(im810,2,2)
    
    resizedH90, mask90 = calculateHomography(im90r,im270r)
    resizedH270, mask270 = calculateHomography(im270r,im450r)
    resizedH630, mask630 = calculateHomography(im630r,im450r)
    resizedH810, mask810 = calculateHomography(im810r,im630r)
    #
    resizedH90=np.matmul(resizedH90,resizedH270)
    resizedH810=np.matmul(resizedH810,resizedH630)
    
    frames=[]
    I=np.array([[1,0,0],[0,1,0],[0,0,1]])
    s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    file = open("homography.txt", "a")



    generateWarpedFrames(1,90, resizedH90, im90r,file)
    h90=np.matmul(np.matmul(s,resizedH90),np.linalg.inv(s))
    warped90=cv2.warpPerspective(im90, h90, dsize=(im270.shape[1], im270.shape[0]))
    cv2.imwrite('task3/frame' + str(90) + '.jpg', warped90)

    #frames.append(warped90)
    writeMatrixToFIle(file,h90)


    generateWarpedFrames(91,270, resizedH270, im270r,file)
    h270 = np.matmul(np.matmul(s, resizedH270), np.linalg.inv(s))
    warped270 = cv2.warpPerspective(im270, h270, dsize=(im270.shape[1], im270.shape[0]))
    cv2.imwrite('task3/frame' + str(270) + '.jpg', warped270)
    writeMatrixToFIle(file,h270)

    #
    generateWarpedFrames(271,450, I, im450r,file)
    cv2.imwrite('task3/frame' + str(450) + '.jpg', im450)
    # generateWarpedFrames(frames,301,450, I, I, im450r,file)
    # frames.append(im450)


    generateWarpedFrames(451, 630, I, im450r,file)
    h630 = np.matmul(np.matmul(s, resizedH630), np.linalg.inv(s))
    warped630 = cv2.warpPerspective(im630, h630, dsize=(im270.shape[1], im270.shape[0]))
    cv2.imwrite('task3/frame' + str(630) + '.jpg', warped630)
    writeMatrixToFIle(file,h630)


    generateWarpedFrames( 631, 810, resizedH630, im630r,file)
    h810 = np.matmul(np.matmul(s, resizedH810), np.linalg.inv(s))
    warped810 = cv2.warpPerspective(im810, h810, dsize=(im270.shape[1], im270.shape[0]))
    cv2.imwrite('task3/frame' + str(810) + '.jpg', warped810)
    writeMatrixToFIle(file,h810)


    generateWarpedFrames( 811, 901, resizedH810, im810r,file)
    # h900 = np.matmul(np.matmul(s, resizedH900), np.linalg.inv(s))
    # warped810 = cv2.warpPerspective(im900, h900, dsize=(im270.shape[1], im270.shape[0]))
    # frames.append(warped810)






    file.close()

    video = cv2.VideoWriter("res05-refrence-plane.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (im270.shape[1], im270.shape[0]))

    for i in range(1,901):
        print(i)
        im=cv2.imread('task3/frame' + str(i) + '.jpg')

        video.write(im)
    video.release()


def doTaskFour(l):
    tempIm=cv2.imread('task3/frame' + str(1) + '.jpg')

    result=np.zeros(tempIm.shape,dtype=np.uint8)
    #values=[]

    for k in range((tempIm.shape[1] // l) - 1):
        print(k)
        currentStripes=np.zeros((900,tempIm.shape[0],l,3))
        for i in range(1,901):
            stripe=cv2.imread('stripes/stripe' + str(i) + '_'+str(k+1)+'.jpg')
            currentStripes[i-1,:,:,:]=stripe
        currentRes=np.apply_along_axis(lambda c :np.median(c[np.nonzero(c)]),0,currentStripes)
        currentRes[np.isnan(currentRes)]=0
        cv2.imwrite('stripes/res'  + str(k + 1) + '.jpg',currentRes)
        result[:,k*l:(k+1)*l,:]=currentRes


    cv2.imwrite('res06-background-panorama.jpg',result)

def doTaskFive():
    im=cv2.imread('res06-background-panorama.jpg')
    file=open('homography.txt','r')
    i=1
    im450=im[750:-850,2500:-2500,:]
    cv2.imwrite('background video frames/frame' + str(450) + '.jpg', im450)

    while True:
        h=file.readline()
        if not h:
            break
        h=np.array([[np.float64(v.strip()) for v in h.replace(',',' ').strip().split('  ')]])
        h=h.reshape((3,3))
        new=cv2.warpPerspective(im,np.linalg.inv(h),(im.shape[1],im.shape[0]))
        new=new[750:-850,2500:-2500,:]
        if (i == 450):
            i += 1
        cv2.imwrite('background video frames/frame'+str(i)+'.jpg',new)
        i=i+1

    file.close()
    im270=cv2.imread('background video frames/frame' + str(270) + '.jpg')

    video = cv2.VideoWriter("res07-background-video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (im270.shape[1], im270.shape[0]))

    for i in range(1,901):
        print(i)
        im = cv2.imread('background video frames/frame' + str(i) + '.jpg')

        video.write(im)
    video.release()


def doTaskSix():
    threshold=30000

    for i in range(1,901):
        print(i)

        bc= np.uint8(cv2.imread('background video frames/frame' + str(i) + '.jpg'))
        main=np.uint8(cv2.imread('frames/frame'+str(i)+'.jpg'))
        # dist=np.zeros((main.shape),np.float64)
        # g=cv2.GaussianBlur(bc,(25,25),0)
        # l=bc-g
        # l=cv2.cvtColor(l,cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("l.jpg",l)
        #
        #
        # l=cv2.morphologyEx(l,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        # cv2.imwrite("l.jpg",l)
        #
        # mask=dist.copy()
        # l[l < 200
        # ] = 0
        # l[l > 0] = 1
        # mask[:,:,0]=l
        # mask[:,:,1]=l
        # mask[:,:,2]=l

        h,mask=calculateHomography(bc,main)
        bc=np.float64(cv2.warpPerspective(bc, h, dsize=(main.shape[1], main.shape[0])))
        main=np.float64(main)
        res=main.copy()
        dist=np.sum((main-bc)**2,axis=2,dtype=np.float64)

        #b,g,r=cv2.split(res)
        res[dist>threshold]=(0,0,255)
        #res=cv2.merge((b,g,r))
        cv2.imwrite('forground video frames/frame'+str(i)+'.jpg',res)

    im1= cv2.imread('forground video frames/frame' + str(1) + '.jpg')

    video = cv2.VideoWriter("res08-forground-video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25,
                            (im1.shape[1], im1.shape[0]))

    for i in range(1,901):
        print(i)
        im = cv2.imread('forground video frames/frame' + str(i) + '.jpg')

        video.write(im)
    video.release()



def doTaskSeven():
    im = cv2.imread('res06-background-panorama.jpg')
    file = open('homography.txt', 'r')
    i = 1
    im450 = im[750:-850, 2000:-2000, :]
    cv2.imwrite('task7/frame' + str(450) + '.jpg', im450)

    while True:
        h = file.readline()
        if not h:
            break
        h = np.array([[np.float64(v.strip()) for v in h.replace(',', ' ').strip().split('  ')]])
        h = h.reshape((3, 3))
        new = cv2.warpPerspective(im, np.linalg.inv(h), (im.shape[1], im.shape[0]))
        new = new[750:-850, 2000:-2000, :]
        if(i==450):
            i+=1
        cv2.imwrite('task7/frame' + str(i) + '.jpg', new)
        i = i + 1

    file.close()
    im270 = cv2.imread('task7/frame' + str(270) + '.jpg')

    video = cv2.VideoWriter("res09-background-video-wider.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25,
                            (im270.shape[1], im270.shape[0]))

    for i in range(1, 901):
        im = cv2.imread('task7/frame' + str(i) + '.jpg')
        if(im[0,0,0]==0 and im[0,0,1]==0 and im[0,0,2]==0 ):
            print(i)

            continue
        if (im[-1, -1, 0] == 0 and im[-1, -1, 1] == 0 and im[-1, -1, 2] == 0):
            print(i)

            continue
        video.write(im)
    video.release()




def doTaskEight():

    file=open('homography.txt','r')
    i=1
    homographies=np.zeros((900,9),np.float64)


    while True:
        if (i == 450):
            homographies[i, :] = np.array([1,0,0,0,1,0,0,0,1])

            i += 1

        h=file.readline()
        if not h:
            break


        h=np.array([[np.float64(v.strip()) for v in h.replace(',',' ').strip().split('  ')]])
        print(homographies.shape,h.shape)
        homographies[i-1,:]=h
        #h=h.reshape((3,3))


        i=i+1

    file.close()
    homographies=cv2.blur(homographies,(1,33))
    for i in range(1,901):
        im=cv2.imread('task3/frame' + str(i) + '.jpg')
        h=homographies[i-1,:]
        h=h.reshape((3,3))
        new = cv2.warpPerspective(im, np.linalg.inv(h), (im.shape[1], im.shape[0]))
        new = new[800:-900, 2600:-2600, :]
        cv2.imwrite('task8/frame' + str(i) + '.jpg',new)

    im270=cv2.imread('task8/frame' + str(270) + '.jpg')

    video = cv2.VideoWriter("res10-video-shakeless.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (im270.shape[1], im270.shape[0]))

    for i in range(1,901):
        print(i)
        im = cv2.imread('task8/frame' + str(i) + '.jpg')

        video.write(im)
    video.release()





#un-comment if it is the first run

# video=cv2.VideoCapture('video.mp4')
# extractAndSaveFrames(video)

doTaskOne()
doTaskTwo()
doTaskThree()
createStripes(40)
doTaskFour(40)
doTaskFive()
doTaskSix()
doTaskSeven()
doTaskEight()
# im270=cv2.imread('task3/frame' + str(270) + '.jpg')
#
# video = cv2.VideoWriter("temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (im270.shape[1], im270.shape[0]))
#
# for i in range(899):
#     print(i)
#     im = cv2.imread('task3/frame' + str(i + 1) + '.jpg')
#
#     video.write(im)
# video.release()


