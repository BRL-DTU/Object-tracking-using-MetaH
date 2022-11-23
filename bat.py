import numpy
import random
import math
import cv2
import glob

PopSize = 15
iterations = 3

NUM_BIN = 8
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

Dataset = 'BlurFace'
image_list = []
for filename in glob.glob(r'../' + Dataset + r'/img/*.jpg'): 
    img=cv2.imread(filename)
    image_list.append(img)
groundTruthDir = r"../"+Dataset+"/groundtruth_rect.txt" 

groundTruth = []
with open(groundTruthDir, 'r') as f:
    line = f.readline()
    while(line):
        line = line[:-1]
        if(Dataset == 'Basketball' or Dataset == 'DragonBaby'):
            line = list(map(int, line.split(',')))
        else:
            line = list(map(int, line.split('\t')))
        groundTruth.append(line)
        line = f.readline()
image = image_list[0]
W = groundTruth[0][2]
H = groundTruth[0][3]
start_point = (groundTruth[0][0], groundTruth[0][1])
end_point = (groundTruth[0][0] + W, groundTruth[0][1] + H)
color = (255, 0, 0)
thickness = 2
image = cv2.rectangle(image, start_point, end_point, color, thickness)
frame = image[start_point[1] + thickness:end_point[1] - thickness, start_point[0] + thickness:end_point[0] - thickness]
leftW = W//2
rightW = (image.shape[1]) - W//2
upH = H//2
downH = (image.shape[0]) - H//2

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


pop_size = 5
iterations = 10
upper = 10
lower = -10
dim = 0


class Bat():
    def __init__(self, pop_size, dim, iterations, data):
        self.pop_size = pop_size
        self.iterations = iterations

        self.X = numpy.zeros((pop_size, dim))  # Position
        self.Xs = numpy.zeros((pop_size, dim))  # Solutions
        self.V = numpy.zeros((pop_size, dim))  # Velocity
        self.f = numpy.random.randn(pop_size, 1)  # Frequency
        self.A = numpy.random.rand(pop_size, 1)  # Loudness
        self.r = numpy.random.rand(pop_size, 1)  # Pulse Width
        self.dim = dim

        self.Fmin = 0
        self.LB = [0]*self.dim
        self.UB = [0]*self.dim
        
        self.Fitness = [0]*self.pop_size
        self.pbest = [0]*self.pop_size
        #self.lower = lower   # X domain
        #self.upper = upper   # X domain
        self.gbest = 0
        self.image_list = data[0]
        self.groundTruth = data[1]
        self.frame = data[2]
        self.leftW = data[3]
        self.rightW = data[4]
        self.upH = data[5]
        self.downH = data[6]
        self.W = data[7]
        self.H = data[8]
        self.thickness = data[9]
        self.color = data[10]

        # Hyperparameters
        self.beta = numpy.random.rand()  # random
        self.alpha = 0.9
        self.gamma = 0.2

        self.LB[0] = self.leftW
        self.UB[0] = self.rightW
        self.LB[1] = self.upH
        self.UB[1] = self.downH

    def best_bat(self):
        i = 0
        j = 0
        for i in range(self.pop_size):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.dim):
            self.pbest[i] = self.Xs[j][i]
        self.Fmin = self.Fitness[j]

    def bat_position(self):
        '''
        for i in range(self.dim):
            self.LB[i] = self.lower
            self.UB[i] = self.upper
        '''

        

        for i in range(self.pop_size):
            self.f[i] = 0
            for j in range(self.dim):
                self.V[i][j] = 0.0
                self.Xs[i][j] = self.LB[j] + \
                    (self.UB[j] - self.LB[j]) * numpy.random.uniform(0, 1)
            StartEnd = getStartEnd(self.Xs[i][0], self.Xs[i][1], self.W, self.H)
            start_point = StartEnd[0]
            end_point = StartEnd[1]
            s1 = numpy.int_(start_point[0])
            e1 = numpy.int_(start_point[1])
            s2 = numpy.int_(end_point[0])
            e2 = numpy.int_(end_point[1])
            img = self.image_list[0][e1 + self.thickness:e2 - self.thickness, s1 + self.thickness:s2 - self.thickness]
            self.Fitness[i] = self.fitness(self.frame, img)
        self.best_bat()

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val
    '''
    def fitness(self, dim, sol):
        val = 0.0
        for i in range(dim):
            d = sol[i] * sol[i]
            val = val + d
        return val
    '''
    # ROSENBROCK FUNCTION

    def fitness(self, imgA, imgB):
        histogramA = create_target_model(imgA)
        histogramB = create_target_model(imgB)
        B = histogramA[:,0]
        G = histogramA[:,1]
        R = histogramA[:,2]
        normR = R / numpy.sum(R)
        normG = G / numpy.sum(G)
        normB = B / numpy.sum(B)
        sqrtNormR = numpy.sqrt(normR)
        sqrtNormG = numpy.sqrt(normG)
        sqrtNormB = numpy.sqrt(normB)

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
        return 1 - (numpy.sqrt((x**2 + y**2 + z**2)))

    def run(self):
        
        index = 0
        #self.X = numpy.zeros((pop_size, dim))  # Position
        #self.Xs = numpy.zeros((pop_size, dim))  # Solutions
        
        for image in self.image_list:
            t = 0
            self.bat_position()
            #self.V = numpy.zeros((pop_size, dim))  # Velocity
            #self.f = numpy.random.randn(pop_size, 1)  # Frequency
            #self.A = numpy.random.rand(pop_size, 1)  # Loudness
            #self.r = numpy.random.rand(pop_size, 1)  # Pulse Width
            a, b = tuple(numpy.random.randint(0, 10, 2))
            fmax = max(a, b)
            fmin = min(a, b)
            for i in range(self.pop_size):
                self.f[i] = fmin + (fmax - fmin)*self.beta
                for j in range(self.dim):
                    self.V[i][j] = self.V[i][j] + \
                        (self.Xs[i][j] - self.pbest[i])*self.f[i]
                    self.X[i][j] = self.Xs[i][j] + self.V[i][j]
                    self.X[i][j] = self.simplebounds(
                        self.X[i][j], self.LB[j], self.UB[j])
            while (t < self.iterations):

                '''
                Initialising Frequency Max and Min
                '''
                a, b = tuple(numpy.random.randint(0, 10, 2))
                fmax = max(a, b)
                fmin = min(a, b)

                '''for i in range(self.pop_size):
                    self.f[i] = fmin + (fmax - fmin)*self.beta
                    for j in range(self.dim):
                        self.V[i][j] = self.V[i][j] + \
                            (self.Xs[i][j] - self.pbest[i])*self.f[i]
                        self.X[i][j] = self.Xs[i][j] + self.V[i][j]
                        self.X[i][j] = self.simplebounds(
                            self.X[i][j], self.LB[j], self.UB[j])'''

                for i in range(self.pop_size):
                    #index = 0
                    
                    if (numpy.random.rand() > self.r[i]):
                        # freq = np.where(self.f == self.f.max())
                        '''
                        freq = self.f.max()
                        for i in self.pop_size:
                            if self.f[i] == freq:
                                index = i

                        # New Local Position
                        self.X[index] = self.X[index] + \
                            numpy.random.unique(-1, 1, 1) * self.A[index]
                        '''

                        for j in range(self.dim):
                            self.X[i][j] = self.pbest[j] + \
                                (numpy.random.uniform(-1, 1, 1)) * \
                                self.A[j]  # LOOK CLOSELY
                            self.X[i][j] = self.simplebounds(
                                self.X[i][j], self.LB[j], self.UB[j])
                        StartEnd = getStartEnd(self.X[i][0], self.X[i][1], self.W, self.H)
                        start_point = StartEnd[0]
                        end_point = StartEnd[1]
                        s1 = numpy.int_(start_point[0])
                        e1 = numpy.int_(start_point[1])
                        s2 = numpy.int_(end_point[0])
                        e2 = numpy.int_(end_point[1])

                        img = image[e1 + self.thickness:e2 - self.thickness, s1 + self.thickness:s2 - self.thickness]
                        Fnew = self.fitness(self.frame, img)

                        if (numpy.random.rand() < self.A[i]) and (Fnew <= self.Fitness[i]):
                            self.r[i] = self.r[i] * \
                                (1 - numpy.exp(-self.gamma*t))   # R increases
                            self.A[i] = self.A[i]*self.alpha   # Loudness decreases

                            for j in range(self.dim):
                                self.Xs[i][j] = self.X[i][j]
                            self.Fitness[i] = Fnew
                        '''
                        else:
                        freq = self.f.max()
                        for i in self.pop_size:
                            if self.f[i] == freq:
                                index = i
                        self.pbest[i] = self.X[index]

                        '''
                        if Fnew <= self.Fmin:
                            for j in range(self.dim):
                                self.pbest[j] = self.X[i][j]
                            self.Fmin = Fnew

                t += 1
                #print(self.Fmin)
                # if t == self.iterations - 2:
                #     print(self.X, self.Xs)
            StartEnd = getStartEnd(self.X[0][0], self.X[0][1], W, H)
            start_point = StartEnd[0]
            end_point = StartEnd[1]
            prevX = start_point[0]
            prevY = start_point[1]
            flagX = True
            flagY = True
            for i in self.X:
                StartEnd = getStartEnd(i[0], i[1], W, H)
                start_point = StartEnd[0]
                end_point = StartEnd[1]
                s1 = numpy.int_(start_point[0])
                e1 = numpy.int_(start_point[1])
                s2 = numpy.int_(end_point[0])
                e2 = numpy.int_(end_point[1])
                image = cv2.rectangle(image, (s1, e1), (s2, e2), self.color, self.thickness)  
                if(prevX != s1):
                    flagX = False
                if(prevY != e1):
                    flagY = False
            #if(flagX == True or flagY == True):
            #    self.X=getRandomRect(PopSize, leftW, rightW, upH, downH)
            image = cv2.rectangle(image, (self.groundTruth[index][0], self.groundTruth[index][1]), (self.groundTruth[index][0] + self.W, self.groundTruth[index][1] + self.H), (0,255,0), self.thickness)  
            cv2.imshow("window", image)
            cv2.waitKey(1)
            index += 1


# Variables
pop_size = pop_size
dim = 2
data = [image_list,groundTruth,frame,leftW,rightW,upH,downH,W,H,thickness,color]

# Function Call
bat = Bat(pop_size, dim, iterations, data)
bat.run()
