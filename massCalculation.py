import numpy 

def massCalculation(fit,PopSize,M):
    Fmax = max(fit)
    Fmin = min(fit)
    Fsum = sum(fit)
    Fmean = Fsum/len(fit)
    if Fmax == Fmin:
        M = numpy.ones(PopSize)
    else:
        best = Fmin
        worst = Fmax
        for p in range(0,PopSize):
           M[p] = (fit[p]-worst)/(best-worst)
            
    Msum=sum(M)
    for q in range(0,PopSize):
        M[q] = M[q]/Msum
            
    return M
