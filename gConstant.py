import numpy

def gConstant(l,alfa):
    G0 = 100
    Gimd = (1/(l + 1)) ** alfa
    G = G0*Gimd
    return G
