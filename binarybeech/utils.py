 import numpy as np

    def _precision(m):
        return np.diag(m) / np.sum(m, axis=1)
        
    def _recall(m):
        return np.diag(m) / np.sum(m, axis=0)

    def _F1(P,R):
        #F = np.zeros_like(P)
        #for i in range(len(
        return 2 * P * R / (P + R)
    
    def _accuracy(m):
        return np.sum(np.diag(m))/np.sum(np.sum(m))
        
