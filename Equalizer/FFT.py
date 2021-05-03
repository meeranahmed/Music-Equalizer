import scipy
from scipy.fftpack import rfft, rfftfreq
from matplotlib import pyplot as plt
import numpy as np

def FFT(x, y):
        
        n=len(x)
        # T=1/y
        yF = rfft(x)
        xF = rfftfreq(n ,1/y)

        print(yF.max())
        # # plt.plot(xF,np.abs(yF))
        # # plt.show()
        plot2=plt.figure(2)
        plt.xlim(0,1000)
        plt.plot(xF,np.abs(yF))
        plt.show()

