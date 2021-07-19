import cv2
import numpy as np


def getObjectGrid(w,h,dimension):
    grid=np.empty((w*h,3),dtype=np.float32)
    c=0
    for j in range(h):
        for i in range(w):
            grid[c]=(i*dimension,j*dimension,0)
            c+=1
    return grid


def getParameters(images,greys):
    worldObjPoints=[]
    imagePoints=[]
    objPoints=getObjectGrid(9,6,22)
    #print(objPoints)
    for i in range(len(images)):

        foundPattern,corners=cv2.findChessboardCorners(greys[i],patternSize=(9,6))
       # print(corners)
        if foundPattern==True:

            #refinedCorners=cv2.cornerSubPix(greys[i],corners,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,25,0.002))
            worldObjPoints.append(objPoints)
            imagePoints.append(corners)

            # cv2.drawChessboardCorners(images[i],(9,6),corners,foundPattern)
            # cv2.imwrite(str(i)+'.jpg',images[i])
            # cv2.drawChessboardCorners(images[i], (9, 6), refinedCorners, foundPattern)
            # cv2.imwrite(str(i) + 'eee.jpg', images[i])

    patternFound,cameraMatrix,distortion,rotation,translation=cv2.calibrateCamera(worldObjPoints,imagePoints,(greys[0].shape[1],greys[0].shape[0]),None,None)
    print(cameraMatrix)



    patternFound,cameraMatrix,distortion,rotation,translation=cv2.calibrateCamera(worldObjPoints,imagePoints,(greys[0].shape[1],greys[0].shape[0]),None,None,flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
    return cameraMatrix

n1=0.07







images=[]
greys=[]

for i in range(1,21):
    if(i<10):
        image=cv2.imread('im0'+str(i)+'.jpg')
    else:
        image=cv2.imread('im'+str(i)+'.jpg')

    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    images.append(image)
    greys.append(grey)

camMatrix1=getParameters(images[0:10],greys[0:10])
camMatrix2=getParameters(images[5:15],greys[5:15])
camMatrix3=getParameters(images[10:20],greys[10:20])
camMatrix4=getParameters(images[0:20],greys[0:20])

print("part 2")
print(camMatrix1)
print(camMatrix2)
print(camMatrix3)
print(camMatrix4)