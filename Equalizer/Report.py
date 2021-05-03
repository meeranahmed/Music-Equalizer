import matplotlib.pyplot as plot
from scipy import signal
import numpy as np

fig1=plot.figure(1,figsize=(8.27,11.7)) #set the size of the PD
x=[1,2,3,4,5,6,7]
y=[1,4,8,9,2,1,5]
plot.subplot(321)
plot.plot(x,y)
fs=10e3
plot.specgram(y,Fs=fs)
plot.subplot(322)
plot.colorbar()
plot.show()
fig1.savefig('Report.pdf' , dpi=1000)

    