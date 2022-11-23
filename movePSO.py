import random

def move(PopSize,dim,pos,vel,pBest,gBest):
    for i in range(0,PopSize):
        for j in range (0,dim):
            r1=random.random()
            r2=random.random()
            C1 = 1
            C2 = 2
            vel[i,j]=(C1*r1*(pBest[i][j] - pos[i,j]) + C2*r2*(gBest[j] - pos[i,j]))
            pos[i,j]=pos[i,j]+vel[i,j]
    
    return pos, vel
