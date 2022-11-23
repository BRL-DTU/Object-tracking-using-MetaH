import Hybrid_GSA_PSO as gsa_pso
import GSA as gsa
import PSO as pso
import DE as de
import HHO as hho
import benchmarks
import csv
import numpy
import time
import matplotlib.pyplot as plt
from dataset import image_list,groundTruth,frame,leftW,rightW,upH,downH,W,H,thickness,color,K

def selector(algo,func_details,popSize,Iter,data):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
    

    if(algo==0):
        x=gsa_pso.GSA_PSO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,data)
    elif(algo==1):
        x=gsa.GSA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,data)    
    elif(algo==2):
        x=pso.PSO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,data)
    elif(algo==3):
        F = 0.3
        cr = 0.5
        x=de.differential_evolution(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,F,cr,data)
    elif(algo==4):
        #objf,lb,ub,dim,SearchAgents_no,Max_iter,data
        x=hho.HHO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,data)
    return x
    
    
# Select optimizers
Hybrid_GSA_PSO=True
GSA=True
PSO=True
DE=True
HHO=True

# Select benchmark function
F1=True
F2=True

Algorithm=[Hybrid_GSA_PSO, GSA, PSO, DE, HHO]
AlgoColor=['b','r','k','g','y']
objectivefunc=[F1, F2] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
Runs=1

# Select general parameters for all optimizers (population size, number of iterations)
PopSize = 15
iterations = 12


#Export results ?
Export=False


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
atLeastOneIteration=False



#cv2.imshow("frame", frame)
#cv2.imshow("window", image)
#cv2.waitKey(0)
mn = -1

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,iterations):
	CnvgHeader.append("Iter"+str(l+1))

#global leftW
#global rightW
#global upH
#global downH
#global frame
#global W
#global H
#global thickness
#global success

figure, axis = plt.subplots(1, len(objectivefunc))
figure1, axis1 = plt.subplots(1, len(objectivefunc))
X = numpy.arange(0, len(groundTruth), K)
Cx = []
Cy = []
index= 0
for truth in groundTruth:
    if(index%K == 0):
        Cx.append(truth[0])
        Cy.append(truth[1])
    index += 1
Cx = numpy.array(Cx)
Cy = numpy.array(Cy)

precisionColor = ['b', 'r']
precisionLabel = ['F1', 'F2']
precision = [[{}, {}], [{}, {}], [{}, {}], [{}, {}], [{}, {}]]
precisionMaxX = 0

for i in range (0, len(Algorithm)):
    for j in range (0, len(objectivefunc)):
        if((Algorithm[i]==True) and (objectivefunc[j]==True)): # start experiment if an Algorithm and an objective function is selected
            for k in range (0,Runs):
                func_details=benchmarks.getFunctionDetails(j)
                data = [image_list,groundTruth,frame,leftW,rightW,upH,downH,W,H,thickness,color,K,precision[i][j]]
                x, precision[i][j]=selector(i,func_details,PopSize,iterations,data)
                precision[i][j] = {element: precision[i][j][element] for element in sorted(precision[i][j].keys())}
                precisionMaxX = max(precisionMaxX, list(precision[i][j].keys())[-1])
                if(Export==True):
                    with open(ExportToFile, 'a') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (atLeastOneIteration==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime"],CnvgHeader])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.Algorithm,x.objectivefunc,x.startTime,x.endTime,x.executionTime],x.convergence])
                        writer.writerow(a)
                    out.close()
                atLeastOneIteration=True # at least one experiment
                x.xCenter = numpy.array(x.xCenter)
                x.yCenter = numpy.array(x.yCenter)
                error = numpy.sqrt((x.xCenter - Cx)**2 + (x.yCenter - Cy)**2)
                axis[j].plot(X, error, color=AlgoColor[i], label=x.Algorithm, linewidth=0.5)
                axis[j].set_title("CLE: " + x.objectivefunc)
                axis[j].legend()

precisionY = []
precisionX = [i for i in range(precisionMaxX + 1)]

for i in range(len(objectivefunc)):
    prec = []
    for n in range(len(Algorithm)):
        if((Algorithm[n]==True) and (objectivefunc[i]==True)):
            val = 0
            precY = []
            for j in range(precisionMaxX + 1):
                if j not in precision[n][i].keys():
                    precY.append(val)
                else:
                    val += precision[n][i][j]
                    precY.append(val)
            prec.append(numpy.array(precY))
            axis1[i].plot(precisionX, prec[n]/len(precision[n][i]), color=AlgoColor[n])
            axis1[i].set_title("Precision rate")
            axis1[i].legend()
        precisionY.append(prec)

#print(precision)
#print(precisionMaxX)
if (atLeastOneIteration==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
plt.show()
        
        
