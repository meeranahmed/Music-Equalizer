#!/usr/bin/env python
# coding: utf-8




import numpy as np
import wavio





sr = 44100   # 44.1khz sampling rate
T  = 3      # wave duration in seconds
F  = [420 , 460 , 500 , 540 , 580 , 620 , 660 , 700 , 740, 780] # array of frequencies




t = np.linspace(0, T, T * sr) # create time array with T * sr samples
s = np.zeros(T * sr)          # initialize s array




for f in F:
    s += np.sin(2 * np.pi * f * t)  # add sin wave with this frequency





wavio.write("sine.wav", s, sr, sampwidth=3)



