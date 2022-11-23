# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	
Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.
Purpose: Main file of Gravitational Search Algorithm(GSA) 
            for minimizing of the Objective Function
Code compatible:
 -- Python: 2.* or 3.*
"""

import random
import numpy
import math
from solution import solution
import time
import massCalculation
import gConstant
import gField
import moveGSA
import cv2

def increment(i, precision):
    if i not in precision.keys():
        precision[int(i)] = 1
    else:
        precision[int(i)] += 1

def getRandomRect(n, leftW, rightW, upH, downH):  
    points = []
    for i in range(n):
        xc = random.randint(leftW, rightW)
        yc = random.randint(upH, downH)
        points.append((xc, yc))
        #image = cv2.rectangle(image, (s1, e1), (s2, e2), color, thickness)     
    return numpy.array(points)
def getStartEnd(xc, yc, W, H):
    s1 = xc - W//2
    e1 = yc - H//2
    s2 = xc + W//2
    e2 = yc + H//2
    return [(s1, e1), (s2, e2)]
mn = -1

        
def GSA(objf,lb,ub,dim,PopSize,iters,data):
    #data = [image_list,groundTruth,frame,leftW,rightW,upH,downH,W,H,thickness,color]
    image_list = data[0]
    groundTruth = data[1]
    frame = data[2]
    leftW = data[3]
    rightW = data[4]
    upH = data[5]
    downH = data[6]
    W = data[7]
    H = data[8]
    thickness = data[9]
    color = data[10]
    k = data[11]
    precisionRate = data[12]
    # GSA parameters
    ElitistCheck =1
    Rpower = 1 

    index = 0
    s=solution()
    timerStart = time.time() 
    s.startTime= time.strftime("%Y-%m-%d-%H-%M-%S")
    pos=getRandomRect(PopSize, groundTruth[0][0], groundTruth[0][0] + W, groundTruth[0][1], groundTruth[0][1] + H)
    for image in image_list:
        """ Initializations """
        vel=numpy.random.rand(PopSize,dim)
        fit = numpy.zeros(PopSize)
        M = numpy.zeros(PopSize)
        gBestScore=float("inf")
        
        
        convergence_curve=numpy.zeros(iters) 
        
        for l in range(0,iters):
            for i in range(0,PopSize):
                l1 = [None] * dim
                l1 = pos[i]
                dr = 2
                if(l1[0] < leftW): 
                    l1[0] = leftW + dr
                elif(l1[0] > rightW):
                    l1[0] = rightW - dr
                if(l1[1] < upH):
                    l1[1] = upH + dr
                elif(l1[1] > downH):
                    l1[1] = downH - dr
                pos[i] = l1
                #Calculate objective function for each particle
                StartEnd = getStartEnd(l1[0], l1[1], W, H)
                start_point = StartEnd[0]
                end_point = StartEnd[1]
                s1 = start_point[0]
                e1 = start_point[1]
                s2 = end_point[0]
                e2 = end_point[1]

                img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]

                fitness=[]
                fitness=objf(img)
                fit[i]=fitness
        
                    
                if(gBestScore>fitness):
                    gBestScore=fitness
                    gBest=l1       
            
            """ Calculating Mass """
            M = massCalculation.massCalculation(fit,PopSize,M)

            """ Calculating Gravitational Constant """        
            G = gConstant.gConstant(l,iters)        
            
            """ Calculating Gfield """        
            acc = gField.gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower)
            
            """ Calculating Position """        
            pos, vel = moveGSA.move(PopSize,dim,pos,vel,acc)
            
            convergence_curve[l]=gBestScore
        
            #if (l%1==0):
            #    print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(gBestScore)]);
        StartEnd = getStartEnd(pos[0][0], pos[0][1], W, H)
        start_point = StartEnd[0]
        end_point = StartEnd[1]
        prevX = start_point[0]
        prevY = start_point[1]
        flagX = True
        flagY = True
        img = numpy.copy(image)
        xCenter = 0
        yCenter = 0
        for i in pos:
            xCenter += i[0]
            yCenter += i[1]
            StartEnd = getStartEnd(i[0], i[1], W, H)
            start_point = StartEnd[0]
            end_point = StartEnd[1]
            s1 = start_point[0]
            e1 = start_point[1]
            s2 = end_point[0]
            e2 = end_point[1]
            img = cv2.rectangle(img, (s1, e1), (s2, e2), color, thickness)  
            if(prevX != s1):
                flagX = False
            if(prevY != e1):
                flagY = False
        xCenter /= len(pos)
        yCenter /= len(pos)
        if (index % k == 0):
            s.xCenter.append(xCenter)
            s.yCenter.append(yCenter)
        if(flagX == True or flagY == True):
            pos=getRandomRect(PopSize, leftW, rightW, upH, downH)
        img = cv2.rectangle(img, (groundTruth[index][0], groundTruth[index][1]), (groundTruth[index][0] + W, groundTruth[index][1] + H), (0,255,0), thickness)
        precision = numpy.sqrt((xCenter - groundTruth[index][0] - W/2)**2 + (yCenter - groundTruth[index][1] - H/2)**2)
        increment(precision, precisionRate)
        cv2.imshow("window", img)
        cv2.waitKey(1)
        index += 1
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.Algorithm="GSA"
    s.objectivefunc=objf.__name__
    return s, precisionRate