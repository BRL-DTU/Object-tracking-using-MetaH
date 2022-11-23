import random

def move(PopSize,dim,pos,vel,acc,pBest,gBest,C):
    for i in range(0,PopSize):
        for j in range (0,dim):
            r1=random.random()
            r2=random.random()
            r3=random.random()
            C1 = 1
            C2 = 2
            vel[i,j]=r1*vel[i,j]+acc[i,j]+(1-C)*(C1*r2*(pBest[i][j] - pos[i][j]) + C2*r3*(gBest[j] - pos[i][j]))
            pos[i][j]=pos[i][j]+vel[i,j]
    
    return pos, vel
