import numpy as np

def precision(m):
    return np.diag(m) / np.sum(m, axis=1)
        
def recall(m):
    return np.diag(m) / np.sum(m, axis=0)

def F1(P,R):
    #F = np.zeros_like(P)
    #for i in range(len(
    return 2 * P * R / (P + R)
    
def accuracy(m):
    return np.sum(np.diag(m))/np.sum(np.sum(m))
        
