import numpy
import math
import cv2
import numpy
from dataset import frame
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

HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
H = []
S = []
V = []
color = ('h','s','v')
for channel,col in enumerate(color):
    histr = cv2.calcHist([HSV_frame],[channel],None,[256],[0,256])
    n = len(histr)
    if(channel == 0):
        for i in histr:
            H.append(i[0])
    elif(channel == 1):
        for i in histr:
            S.append(i[0])
    else:
        for i in histr:
            V.append(i[0])
H = numpy.array(H)
S = numpy.array(S)
V = numpy.array(V)
normH = H / numpy.sum(H)
normS = S / numpy.sum(S)
normV = V / numpy.sum(V)
sqrtNormH= numpy.sqrt(normH)
sqrtNormS = numpy.sqrt(normS)
sqrtNormV = numpy.sqrt(normV)


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
def F1(imgB):
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

def F2(imgB):
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2HSV)
    _R = []
    _G = []
    _B = []
    color = ('b','g','r')
    for channel,col in enumerate(color):
        histr = cv2.calcHist([imgB],[channel],None,[256],[0,256])
        n = len(histr)
        if(channel == 0):
            for i in histr:
                _B.append(i[0])
        elif(channel == 1):
            for i in histr:
                _G.append(i[0])
        else:
            for i in histr:
                _R.append(i[0])
    _R = numpy.array(_R)
    _G = numpy.array(_G)
    _B = numpy.array(_B)
    norm_R = _R / numpy.sum(_R)
    norm_G = _G / numpy.sum(_G)
    norm_B = _B / numpy.sum(_B)
    sqrtNorm_R = numpy.sqrt(norm_R)
    sqrtNorm_G = numpy.sqrt(norm_G)
    sqrtNorm_B = numpy.sqrt(norm_B)
    
    x = numpy.dot(sqrtNormH, sqrtNorm_R)
    y = numpy.dot(sqrtNormS, sqrtNorm_G)
    z = numpy.dot(sqrtNormV, sqrtNorm_B)
    return 1 - ((x + y + z)/3)


def getFunctionDetails(a):
  # [name, lb, ub, dim]
  param = {  0: ["F1",-100,100,2],
             1: ["F2",-100,100,2],
            }
  return param.get(a, "nothing")



