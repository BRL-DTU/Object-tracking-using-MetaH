# -*- coding: utf-8 -*-
"""
Created on Thirsday March 21  2019
@author: Ali Asghar Heidari, Hossam Faris
% _____________________________________________________
% Main paper:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems, 
% DOI: https://doi.org/10.1016/j.future.2019.02.028
% _____________________________________________________
"""
import random
import numpy
import math
from solution import solution
import time
import cv2
def getStartEnd(xc, yc, W, H):
    s1 = xc - W//2
    e1 = yc - H//2
    s2 = xc + W//2
    e2 = yc + H//2
    return [(s1, e1), (s2, e2)]

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


def HHO(objf,lb,ub,dim,SearchAgents_no,Max_iter,data):

    #dim=30
    #SearchAgents_no=50
    #lb=-100
    #ub=100
    #Max_iter=500
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
    
    # initialize the location and Energy of the rabbit
    
    
    #if not isinstance(lb, list):
    #    lb = [lb for _ in range(dim)]
    #    ub = [ub for _ in range(dim)]
    lb = [leftW, upH]
    ub = [rightW, downH]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)
         
    #Initialize the locations of Harris' hawks
    
    
    
    ############################
    s=solution()
    

    #print("HHO is now tackling  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    index = 0
    for image in image_list:
        Rabbit_Location=numpy.zeros(dim)
        Rabbit_Energy=float("inf")  #change this to -inf for maximization problems
        X = getRandomRect(SearchAgents_no, groundTruth[0][0], groundTruth[0][0] + W, groundTruth[0][1], groundTruth[0][1] + H)
    
        #Initialize convergence
        convergence_curve=numpy.zeros(Max_iter)
        
        t=0  # Loop counter
        
        # Main loop
        while t<Max_iter:
            for i in range(0,SearchAgents_no):
                
                # Check boundries
                        
                X[i,:]=numpy.clip(X[i,:], lb, ub)
                
                # fitness of locations
                StartEnd = getStartEnd(X[i][0], X[i][1], W, H)
                start_point = StartEnd[0]
                end_point = StartEnd[1]
                s1 = numpy.int_(start_point[0])
                e1 = numpy.int_(start_point[1])
                s2 = numpy.int_(end_point[0])
                e2 = numpy.int_(end_point[1])

                img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]

                fitness=objf(img)
                
                # Update the location of Rabbit
                if fitness<Rabbit_Energy: # Change this to > for maximization problem
                    Rabbit_Energy=fitness 
                    Rabbit_Location=X[i,:].copy() 
                
            E1=2*(1-(t/Max_iter)) # factor to show the decreaing energy of rabbit    
            
            # Update the location of Harris' hawks 
            for i in range(0,SearchAgents_no):

                E0=2*random.random()-1  # -1<E0<1
                Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper

                # -------- Exploration phase Eq. (1) in paper -------------------

                if abs(Escaping_Energy)>=1:
                    #print("if abs(Escaping_Energy)>=1:")
                    #Harris' hawks perch randomly based on 2 strategy:
                    q = random.random()
                    rand_Hawk_index = math.floor(SearchAgents_no*random.random())
                    X_rand = X[rand_Hawk_index, :]
                    if q<0.5:
                        # perch based on other family members
                        X[i,:]=X_rand-random.random()*abs(X_rand-2*random.random()*X[i,:])

                    elif q>=0.5:
                        #perch on a random tall tree (random site inside group's home range)
                        X[i,:]=(Rabbit_Location - X.mean(0))-random.random()*((ub-lb)*random.random()+lb)

                # -------- Exploitation phase -------------------
                elif abs(Escaping_Energy)<1:
                    #print("if abs(Escaping_Energy)<1:")
                    #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                    #phase 1: ----- surprise pounce (seven kills) ----------
                    #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                    r=random.random() # probablity of each event
                    
                    if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                        X[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X[i,:])

                    if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                        Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                        X[i,:]=(Rabbit_Location-X[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                    
                    #phase 2: --------performing team rapid dives (leapfrog movements)----------

                    if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                        #print("if r<0.5 and abs(Escaping_Energy)>=0.5:")
                        #rabbit try to escape by many zigzag deceptive motions
                        Jump_strength=2*(1-random.random())
                        X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                        X1 = numpy.clip(X1, lb, ub)
                        StartEnd = getStartEnd(X1[0], X1[1], W, H)
                        start_point = StartEnd[0]
                        end_point = StartEnd[1]
                        s1 = numpy.int_(start_point[0])
                        e1 = numpy.int_(start_point[1])
                        s2 = numpy.int_(end_point[0])
                        e2 = numpy.int_(end_point[1])

                        img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]
                        if objf(img)< fitness: # improved move?
                            X[i,:] = X1.copy()
                        else: # hawks perform levy-based short rapid dives around the rabbit
                            X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                            X2 = numpy.clip(X2, lb, ub)
                            StartEnd = getStartEnd(X2[0], X2[1], W, H)
                            start_point = StartEnd[0]
                            end_point = StartEnd[1]
                            s1 = numpy.int_(start_point[0])
                            e1 = numpy.int_(start_point[1])
                            s2 = numpy.int_(end_point[0])
                            e2 = numpy.int_(end_point[1])

                            img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]
                            if objf(img)< fitness:
                                X[i,:] = X2.copy()
                    if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                        #print("if r<0.5 and abs(Escaping_Energy)<0.5:")
                        Jump_strength=2*(1-random.random())
                        X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))
                        X1 = numpy.clip(X1, lb, ub)
                        StartEnd = getStartEnd(X1[0], X1[1], W, H)
                        start_point = StartEnd[0]
                        end_point = StartEnd[1]
                        s1 = numpy.int_(start_point[0])
                        e1 = numpy.int_(start_point[1])
                        s2 = numpy.int_(end_point[0])
                        e2 = numpy.int_(end_point[1])

                        img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]
                        if objf(img)< fitness: # improved move?
                            X[i,:] = X1.copy()
                        else: # Perform levy-based short rapid dives around the rabbit
                            X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                            X2 = numpy.clip(X2, lb, ub)
                            StartEnd = getStartEnd(X2[0], X2[1], W, H)
                            start_point = StartEnd[0]
                            end_point = StartEnd[1]
                            s1 = numpy.int_(start_point[0])
                            e1 = numpy.int_(start_point[1])
                            s2 = numpy.int_(end_point[0])
                            e2 = numpy.int_(end_point[1])

                            img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]
                            if objf(img)< fitness:
                                X[i,:] = X2.copy()
                    
            convergence_curve[t]=Rabbit_Energy
            #if (t%1==0):
                #print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Rabbit_Energy)])
            t=t+1
        
        StartEnd = getStartEnd(X[0][0], X[0][1], W, H)
        start_point = StartEnd[0]
        end_point = StartEnd[1]
        prevX = start_point[0]
        prevY = start_point[1]
        flagX = True
        flagY = True
        img = numpy.copy(image)
        xCenter = 0
        yCenter = 0
        for i in X:
            xCenter += i[0]
            yCenter += i[1]
            StartEnd = getStartEnd(i[0], i[1], W, H)
            start_point = StartEnd[0]
            end_point = StartEnd[1]
            s1 = numpy.int_(start_point[0])
            e1 = numpy.int_(start_point[1])
            s2 = numpy.int_(end_point[0])
            e2 = numpy.int_(end_point[1])
            img = cv2.rectangle(img, (s1, e1), (s2, e2), color, thickness)  
            if(prevX != s1):
                flagX = False
            if(prevY != e1):
                flagY = False
        xCenter /= len(X)
        yCenter /= len(X)
        if (index % k == 0):
            s.xCenter.append(xCenter)
            s.yCenter.append(yCenter)
        #if(flagX == True or flagY == True):
        #    X = getRandomRect(SearchAgents_no, groundTruth[0][0], groundTruth[0][0] + W, groundTruth[0][1], groundTruth[0][1] + H)
        img = cv2.rectangle(img, (groundTruth[index][0], groundTruth[index][1]), (groundTruth[index][0] + W, groundTruth[index][1] + H), (0,255,0), thickness)
        precision = numpy.sqrt((xCenter - groundTruth[index][0] - W/2)**2 + (yCenter - groundTruth[index][1] - H/2)**2)
        increment(precision, precisionRate)
        cv2.imshow("window", img)
        cv2.waitKey(1)
        index += 1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.Algorithm="HHO"   
    s.objectivefunc=objf.__name__
    s.best =Rabbit_Energy 
    s.bestIndividual = Rabbit_Location
    
    
    
    return s, precisionRate

def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step
