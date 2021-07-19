import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import fsolve
from scipy import ndimage

imMain=cv2.imread('vns.jpg')
im=cv2.imread('vnsc2.jpg')

edges=cv2.Canny(im,200,200)
vx=0
vy=0
vz=0
h=0


def getParalellLines(theta,margin,thresh):
    return cv2.HoughLines(edges,1,np.pi/180,thresh,min_theta=theta-margin,max_theta=theta+margin)

def standardForm(polarLines):

    if polarLines is not None:
        for i in range(0, len(polarLines)):
            rho = polarLines[i][0][0]
            theta = polarLines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(im, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imwrite('t.jpg',im)
    lines=np.zeros((polarLines.shape[0],3),dtype=np.float64)
    print(polarLines.shape[0],polarLines[0].shape[0])
    print(lines.shape)

    for ind,line in enumerate(polarLines):
        rho=line[0][0]
        theta=line[0][1]
        lines[ind,:]=(np.cos(theta),np.sin(theta),-rho)
    return lines

def getVanishingPoint(lines):
    u,s,vt=np.linalg.svd(lines)
    line=vt[2,:]

    x=line[0]/line[2]
    y=line[1]/line[2]
    return [x,y]
def findVanishingPoints():
    linesX=getParalellLines(np.pi/2+0.21,0.1,170)
    linesY=getParalellLines(np.pi/2-0.04,0.05,230)
    linesZ=getParalellLines(np.pi-0.02,0.05,500)
    linesX=standardForm(linesX)
    linesY=standardForm(linesY)
    linesZ=standardForm(linesZ)

    vx=getVanishingPoint(linesX)
    vy=getVanishingPoint(linesY)
    vz=getVanishingPoint(linesZ)
    dx = imMain.shape[1] - im.shape[1]
    vy[0] = vy[0] + dx
    vx[0] = vx[0] + dx
    vz[0] = vz[0]+  dx

    hvx=[vx[0],vx[1],1]
    hvy=[vy[0],vy[1],1]

    h=np.cross(hvx,hvy)
    t=np.sqrt(h[0]**2+h[1]**2)
    normalizedH=h/t
    print("&&&&&&&&")
    print(normalizedH)

    m=-normalizedH[0]/normalizedH[1]
    b=-normalizedH[2]/normalizedH[1]
    x1=0
    y1=int(b)
    x2=imMain.shape[1]
    y2=int(m*x2+b)
    l=cv2.line(imMain.copy(),(x1,y1),(x2,y2),(0,0,255),3)
    cv2.imwrite('res01.jpg',l)

    x=[-27000,10000]
    y=[m*x[0]+b,m*x[1]+b]
    plt.axis('equal')

    ax2=plt.gcf()
    ax2.set_size_inches(5.5,18)


    plt.xlim(x)
    plt.ylim([5000,-190000])
    ax=plt.gca()
    plt.plot(x,y)

    plt.plot([vx[0],vy[0]],[vx[1],vy[1]],'o')
    plt.plot([vz[0]],[vz[1]],'o')


    border=patches.Rectangle((0,0),imMain.shape[1],imMain.shape[0],linewidth=1,edgecolor='r',fill=None)
    ax.add_patch(border)
    plt.tight_layout()


    plt.savefig('res02.jpg')
    plt.close()



    print(vx)
    print(vy)
    print(vz)
    return vx,vy,vz,normalizedH

def func(x):
    eq1=-(x[0]**2)-(x[1]**2)-(x[2]**2)+(vx[0]+vy[0])*x[1]+(vx[1]+vy[1])*x[2]-vx[0]*vy[0]-vx[1]*vy[1]
    eq2=-(x[0]**2)-(x[1]**2)-(x[2]**2)+(vx[0]+vz[0])*x[1]+(vx[1]+vz[1])*x[2]-vx[0]*vz[0]-vx[1]*vz[1]
    eq3=-(x[0]**2)-(x[1]**2)-(x[2]**2)+(vy[0]+vz[0])*x[1]+(vy[1]+vz[1])*x[2]-vy[0]*vz[0]-vy[1]*vz[1]


    return [eq1,eq2,eq3]

def findAngle(v1,v2):
    print('*****')


    dot = np.dot(v1 ,v2)
    theta = math.acos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return theta

def getCameraParameters():
    res=fsolve(func,[100,100,100])
    plt.figure()
    plt.plot([res[1]],[res[2]], 'o')
    plt.title(str(res[0]))
    plt.imshow(cv2.cvtColor(imMain.copy(),cv2.COLOR_BGR2RGB))

    plt.savefig('res03.jpg')
    print(res)

    K=np.array([[res[0],0,res[1]],[0,res[0],res[2]],[0,0,1]])
    z=np.matmul(np.linalg.inv(K),np.array([[vz[0]],[vz[1]],[1]]))
    zMain=np.array([[0],[0],[1]])
    # dot=np.sqrt(np.sum(z*zMain))
    # print(dot)
    # theta=np.arccos(dot/(np.sqrt(np.sum(z**2))*np.sqrt(np.sum(zMain**2))))
    # print(np.sum(zMain**2))
    # print(theta)

    theta=findAngle(np.array([vx[0]-vy[0],vx[1]-vy[1],0]),[1,0,0])
    print(555555,theta)

    #theta=-math.atan((vx[1]-vy[1])/(vx[0]-vy[0]))
    rz=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    r=cv2.warpPerspective(imMain.copy(),rz,(imMain.shape[1],imMain.shape[0]))
    cv2.imwrite('r.jpg',r)


    #K=np.matmul(K,rz)
    nvz=np.matmul(rz,np.array([[vz[0]], [vz[1]], [1]]))
    nvz=[nvz[0]/nvz[2],nvz[1]/nvz[2],1]
    z = np.matmul(np.linalg.inv(K), np.array([[nvz[0]], [nvz[1]], [1]]))
    z=[z[0][0][0],z[1][0][0],z[2][0][0]]
    print(z)
    zMain = [0,0,1]
    theta=findAngle(z,zMain)-np.pi/2
    print(666666,theta)
    rx = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    # print(rx)
    #r = cv2.warpPerspective(imMain.copy(), rx, (imMain.shape[1], imMain.shape[0]))

    print(theta)
    rr=ndimage.rotate(r,-theta)
    homography=np.matmul(rx,rz)
    print("ttt")
    print(homography)
    cv2.imwrite('res04.jpg',rr)




vx,vy,vz,h=findVanishingPoints()
print(h)
getCameraParameters()