# differential evolution search of the two-dimensional sphere objective function
import numpy
import random
import time
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from solution import solution
import cv2
NUM_BIN = 16
BAND = 255 /NUM_BIN
def to_b_num(img):
    assert isinstance(img, numpy.ndarray)
    b = (img //BAND).astype(int)
    return numpy.minimum(b, 15)

def create_kernel(r, c):
    """
    create kernel with epancechnikov profile
    
    @param r, c: size
    """ 
    rr = numpy.arange(r) / (r-1) *2 -1
    cc = numpy.arange(c) / (c-1) *2 -1
    C, R = numpy.meshgrid(cc, rr)
    X2 = C**2 + R**2
    
    kernel = numpy.maximum(1-X2, 0)
    return kernel / numpy.sum(kernel)

def create_target_model(target_img):
    B = to_b_num(target_img)
    
    kernel = create_kernel(*target_img.shape[:2])
    
    M = numpy.empty((NUM_BIN, 3))
    for b in range(NUM_BIN):
        for ch in range(3):
            M[b, ch] = numpy.sum(kernel[B[:,:,ch]==b])
            
    return M

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

'''
# define objective function
histogramA = create_target_model(frame)
B = histogramA[:,0]
G = histogramA[:,1]
R = histogramA[:,2]
normR = R / numpy.sum(R)
normG = G / numpy.sum(G)
normB = B / numpy.sum(B)
sqrtNormR = numpy.sqrt(normR)
sqrtNormG = numpy.sqrt(normG)
sqrtNormB = numpy.sqrt(normB)
def obj(imgB):
    histogramB = create_target_model(imgB)
    _B= histogramB[:,0]
    _G= histogramB[:,1]
    _R= histogramB[:,2]
    norm_R = _R / numpy.sum(_R)
    norm_G = _G / numpy.sum(_G)
    norm_B = _B / numpy.sum(_B)
    sqrtNorm_R = numpy.sqrt(norm_R)
    sqrtNorm_G = numpy.sqrt(norm_G)
    sqrtNorm_B = numpy.sqrt(norm_B)
    x = numpy.dot(sqrtNormR, sqrtNorm_R)
    y = numpy.dot(sqrtNormG, sqrtNorm_G)
    z = numpy.dot(sqrtNormB, sqrtNorm_B)
    return 1 - ((x + y + z)/3)
'''


'''grayA = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
def obj(imgB):
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    return -score'''


# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def differential_evolution(objf, lb, ub, dim, pop_size, iter, F, cr, data):
    # initialise population of candidate solutions randomly within the specified bounds
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
    bounds = asarray([(leftW, rightW), (upH, downH)])
    index= 0
    pop = getRandomRect(pop_size, groundTruth[0][0], groundTruth[0][0] + W, groundTruth[0][1], groundTruth[0][1] + H)
    s=solution()
    timerStart = time.time() 
    s.startTime= time.strftime("%Y-%m-%d-%H-%M-%S")
    for image in image_list:
        
        # evaluate initial population of candidate solutions
        boxes = []
        for l1 in pop:
            StartEnd = getStartEnd(l1[0], l1[1], W, H)
            start_point = StartEnd[0]
            end_point = StartEnd[1]
            s1 = numpy.int_(start_point[0])
            e1 = numpy.int_(start_point[1])
            s2 = numpy.int_(end_point[0])
            e2 = numpy.int_(end_point[1])

            img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]
            boxes.append(img)
        obj_all = [objf(ind) for ind in boxes]
        # find the best performing vector of initial population
        best_vector = pop[argmin(obj_all)]
        best_obj = min(obj_all)
        prev_obj = best_obj
        # initialise list to store the objective function value at each iteration
        obj_iter = list()
        # run iterations of the algorithm
        for i in range(iter):
            # iterate over all candidate solutions
            for j in range(pop_size):
                # choose three candidates, a, b and c, that are not the current one
                candidates = [candidate for candidate in range(pop_size) if candidate != j]
                a, b, c = pop[choice(candidates, 3, replace=False)]
                # perform mutation
                mutated = mutation([a, b, c], F)
                # check that lower and upper bounds are retained after mutation
                mutated = check_bounds(mutated, bounds)
                # perform crossover
                trial = crossover(mutated, pop[j], len(bounds), cr)
                # compute objective function value for target vector
                l1 = pop[j]

                dr = 2
                if(l1[0] < leftW): 
                    l1[0] = leftW + dr
                elif(l1[0] > rightW):
                    l1[0] = rightW - dr
                if(l1[1] < upH):
                    l1[1] = upH + dr
                elif(l1[1] > downH):
                    l1[1] = downH - dr
                pop[j] = l1
                StartEnd = getStartEnd(l1[0], l1[1], W, H)
                start_point = StartEnd[0]
                end_point = StartEnd[1]
                s1 = numpy.int_(start_point[0])
                e1 = numpy.int_(start_point[1])
                s2 = numpy.int_(end_point[0])
                e2 = numpy.int_(end_point[1])
                img = image[e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]
                obj_target = objf(img)
                # compute objective function value for trial vector
                l1 = trial

                dr = 2
                if(l1[0] < leftW): 
                    l1[0] = leftW + dr
                elif(l1[0] > rightW):
                    l1[0] = rightW - dr
                if(l1[1] < upH):
                    l1[1] = upH + dr
                elif(l1[1] > downH):
                    l1[1] = downH - dr
                trial = l1
                StartEnd = getStartEnd(l1[0], l1[1], W, H)
                start_point = StartEnd[0]
                end_point = StartEnd[1]
                s1 = numpy.int_(start_point[0])
                e1 = numpy.int_(start_point[1])
                s2 = numpy.int_(end_point[0])
                e2 = numpy.int_(end_point[1])
                img = image_list[0][e1 + thickness:e2 - thickness, s1 + thickness:s2 - thickness]
                obj_trial = objf(img)
                # perform selection
                if obj_trial < obj_target:
                    # replace the target vector with the trial vector
                    pop[j] = trial
                    # store the new objective function value
                    obj_all[j] = obj_trial
            # find the best performing vector at each iteration
            best_obj = min(obj_all)
            # store the lowest objective function value
            if best_obj < prev_obj:
                best_vector = pop[argmin(obj_all)]
                prev_obj = best_obj
                obj_iter.append(best_obj)
                # report progress at each iteration
                #print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
        
        StartEnd = getStartEnd(pop[0][0], pop[0][1], W, H)
        start_point = StartEnd[0]
        end_point = StartEnd[1]
        prevX = start_point[0]
        prevY = start_point[1]
        flagX = True
        flagY = True
        img = numpy.copy(image)
        xCenter = 0
        yCenter = 0
        for i in pop:
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
        xCenter /= len(pop)
        yCenter /= len(pop)
        if (index % k == 0):
            s.xCenter.append(xCenter)
            s.yCenter.append(yCenter)
        if(flagX == True or flagY == True):
            pop=getRandomRect(pop_size, leftW, rightW, upH, downH)
        img = cv2.rectangle(img, (groundTruth[index][0], groundTruth[index][1]), (groundTruth[index][0] + W, groundTruth[index][1] + H), (0,255,0), thickness)
        precision = numpy.sqrt((xCenter - groundTruth[index][0] - W/2)**2 + (yCenter - groundTruth[index][1] - H/2)**2)
        increment(precision, precisionRate)
        cv2.imshow("window", img)
        cv2.waitKey(1)
        index += 1
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.Algorithm="DE"
    s.objectivefunc=objf.__name__
    return s, precisionRate
    #return [best_vector, best_obj, obj_iter]


# define population size
#pop_size = 5
# define lower and upper bounds for every dimension
# define number of iterations
#iter = 20
# define scale factor for mutation
#F = 0.5
# define crossover rate for recombination
#cr = 0.7

# perform differential evolution
#solution = 
#differential_evolution(pop_size, iter, F, cr)
#print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))

'''
# line plot of best objective function values
pyplot.plot(solution[2], '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
'''