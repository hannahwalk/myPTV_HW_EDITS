#!/usr/bin/env python
# coding: utf-8



import pims
import matplotlib.pyplot as plt
import numpy as np
import math 
import json
#import pandas as pd
#import time


@pims.pipeline
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray 

@pims.pipeline
def cropL(img):
    x_min = 130
    x_max = 530
    y_min = 300
    y_max = 700 
    return img[y_min:y_max,x_min:x_max]

@pims.pipeline
def cropR(img):
    x_min = 870
    x_max = 1270
    y_min = 300
    y_max = 700 
    return img[y_min:y_max,x_min:x_max]

@pims.pipeline
def subtractBackground(img,fLavg):
    return img - fLavg

@pims.pipeline
def zeroPad(img,r,c):
    imgNew = np.zeros((r,c))

    
    imgNew[0:np.shape(img)[0],0:np.shape(img)[1]] = img
    
    return imgNew

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def outlierFilter(data,nSD):
    '''Remove any outliers that are beyond median+\-n*SD'''
    sigma = nSD*np.std(data)
    mu = np.mean(data)
    data[(data > (mu+sigma)) & (data < (mu-sigma))] = np.nan
    return data

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def pivStereo(dxR,dyR,dxL,dyL):
        '''Perform stereo PIV'''
        #stereo calculation
        #second run to save all results in txt file
        theta = np.array([ -5.75852669,  72.35917543,  -7.94223563]) #angles from calibration
        angle = theta*np.pi/180 #convert to radians
        
        #first find rotation matrix of each camera
        R_left = eulerAnglesToRotationMatrix(angle/2.0)
        R_right = eulerAnglesToRotationMatrix(-angle/2.0)

        #stack rotation matrices
        rMat = np.concatenate((R_left[[0,1],:],R_right[[0,1],:]),axis=0)
        pMat = np.vstack((dxL,dyL,dxR,dyR))
        
        velocity = np.linalg.lstsq(rMat,pMat)[0]
        
        return velocity

def coarsePIVfunc(frames,side,save = 0,fnum = '0'):
    dxArr = []
    dyArr = []
    fnum = []
    dxArr_mn = []
    dyArr_mn = []
    fignum = 0
    cnd = False
    batchcount = 0
    corrP = []

    for i in np.arange(len(frames)-1):
        A = frames[i]
        B = frames[i+1]
        A = A[-390:,-390:]
        B = B[-390:,-390:]
        #A = A[-256:,-256:]
        #B = B[-256:,-256:]
        
        # Subtract off a 'functional' background mean - 2D 'detrending'
        A = np.round(A-np.mean(A))
        B = np.round(B-np.mean(B))

        A[A<0] = 0
        B[B<0] = 0

        # Cross correlation
        Af = np.fft.fft2(A)
        Bf = np.fft.fft2(B)

        FXc = np.multiply(np.conj(Af),(Bf))
        FX = np.real(np.fft.ifft2(FXc))

        # reorganize from inverse fourier transform
        NF = np.fft.fftshift(FX)

        # find max in correlation matrix
        #mx = np.amax(np.amax(FX))
        mx = np.amax(np.amax(FX))
        mxpos = np.argwhere(NF == mx)
        
        corrP.append(mx)
        
        xmx = mxpos[0][0]
        ymx = mxpos[0][1]

        if cnd == True:
            plt.figure(fignum + 1)
            plt.imshow(NF)
            plt.plot(ymx,xmx,marker = 'o',mfc='none', color='r')

        dy = (ymx-1-len(A)/2)
        dx = (xmx-1-len(A)/2)

        dxArr.append(dy)
        dyArr.append(dx)
        fnum.append(i)
        
        fignum = fignum + 3
        gap = 9
        if batchcount - gap == 0:
            dxArr_mn.append(np.median(dxArr[-(gap-1):]))
            dyArr_mn.append(np.median(dyArr[-(gap-1):]))
            batchcount = 0
        else:
            batchcount += 1

    plotting = 0
    if plotting:
        plt.figure(9)
        plt.scatter(fnum,dxArr,2)
        plt.title("dx")
        plt.ylim((-100,100))
        
        plt.figure(10)
        plt.scatter(fnum,dyArr,2)
        plt.title("dy")
        plt.ylim((-40,40))
        
        # Plot results
        plt.figure(11)
        plt.plot(dxArr_mn)
        plt.title("dx")
    
        plt.figure(12)
        plt.plot(dyArr_mn)
        plt.title("dy")
        
        plt.figure(13)
        plt.plot(corrP)
    

    if save:
        fpath = '/Users/hannahwalker/Desktop/viv/videos/test_frames1/frame_test/'
        if fnum == '0':
            fname = fnum + '_coarsePIV'
        else:
            fname = 'test_coarsePIV'
            
        out_filedx = open(fpath+fname+'dx'+side+'.txt', "w")
        json.dump(dxArr_mn,out_filedx)
        out_filedx.close()

        out_filedy = open(fpath+fname+'dy'+side+'.txt', "w")
        json.dump(dyArr_mn,out_filedy)
        out_filedy.close()
    
    return dxArr, dyArr, dxArr_mn, dyArr_mn, corrP


# Main code begins here
def runCoarsePIV():
   
    save = 0
    fpathR = '/Users/hannahwalker/Desktop/MyPTV-master/Experiment/Images_camR/'
    framesR = pims.ImageSequence(fpathR+'*.bmp')
    dxR, dyR, dxRMN, dyRMN, corrR = coarsePIVfunc(framesR,'R',save)
    # Quality Control
    dxRMN = outlierFilter(np.array(dxRMN),1)
    dyRMN = outlierFilter(np.array(dyRMN),1)
    print("done with R")
    
    fpathL = '/Users/hannahwalker/Desktop/MyPTV-master/Experiment/Images_camL/'
    framesL = pims.ImageSequence(fpathL+'*.bmp')

    dxL, dyL, dxLMN, dyLMN, corrL = coarsePIVfunc(framesL,'L',save)
    dxLMN = outlierFilter(np.array(dxLMN),1)
    dyLMN = outlierFilter(np.array(dyLMN),1)
    print("done with L")

    
    dxRMN = moving_average(dxRMN, 10)
    dyRMN = moving_average(dyRMN, 10)
    dxLMN = moving_average(dxLMN, 10)
    dyLMN = moving_average(dyLMN, 10)

    Fr = 65
    timeV = np.arange(0,np.shape(dxRMN)[0])/Fr*10

    scale = 3.6
    velocity = scale*pivStereo(dxRMN,dyRMN,dxLMN,dyLMN)/10
    velLong  = scale*pivStereo(dxR,dyR,dxL,dyL)/10
    
    timeLong =  np.arange(0,np.shape(velLong[0])[0])/Fr

    
    VfiltX = np.interp(timeLong,timeV,velocity[0])
    VfiltY = np.interp(timeLong,timeV,velocity[1])
    VfiltZ = np.interp(timeLong,timeV,velocity[2])
    
    plotting = 0
    if plotting:
        plt.figure(103)
        plt.plot(timeV,velocity[0])
        plt.plot(timeV,velocity[1])
        plt.plot(timeV,velocity[2])
        plt.xlabel('time [s]')
        plt.ylabel('velocity [cm/s]')
        plt.title('Smoothed PIV Velocity (moving Average of 10 frame batch vector)')
        
        plt.figure(104)
        plt.plot(timeLong,VfiltX)
        plt.plot(timeLong,VfiltY)
        plt.plot(timeLong,VfiltZ)
        plt.xlabel('time [s]')
        plt.ylabel('velocity [cm/s]')
        plt.title('Smoothed PIV Velocity ')
        
        plt.figure(105)
        plt.plot(timeLong,np.sqrt( np.square(VfiltX) + np.square(VfiltY) + np.square(VfiltZ) ))

    vLongSmooth = np.vstack([VfiltX,VfiltY,VfiltZ])
    
    return timeLong, vLongSmooth
        
if __name__ == '__main__':
    runCoarsePIV()