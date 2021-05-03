from FFT import FFT
from PyQt5 import QtCore, QtGui, QtWidgets
import librosa
from librosa import display
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QGroupBox, QPushButton, QVBoxLayout, QSlider, QLabel, QComboBox, QApplication, QFileDialog
from mainwindow import Ui_MainWindow
import pyqtgraph as pg
from scipy import signal, fft
from scipy.fftpack import rfft, rfftfreq
from matplotlib import pyplot as plot

from playsound import playsound
from scipy.io.wavfile import write

import scipy.io.wavfile as wavf
import numpy as np
import sys
import winsound


class Functions (Ui_MainWindow):
    def __init__(self, window):
        self.setupUi(window)
        self.plotWidgets = [self.sig1, self.sig2, self.fourier2, self.fourier1, self.spectro1, self.spectro2]
        self.Scales = [self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7, self.s8, self.s9, self.s10]
        self.frequency = [400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800]
        self.samples = []
        self.sampling_rate = 0
        self.duration = 0
        self.Time = []
        self.plotsData = []  # [x, y], All data of the curve
        self.plotsObjects = []  # Each object of a plot
        self.modifiedAmplitudes = []
        self.s=250
        self.actionImport.triggered.connect(self.Importbutton1)
        self.menuPrint_2.triggered.connect(self.save_pdf)
        self.actionNew.triggered.connect(self.newWindow)
        self.zoom_in.clicked.connect(lambda: self.zoom(x=0, y=0.5))
        self.zoom_out.clicked.connect(lambda: self.zoom(x=0, y=2))
        self.up.clicked.connect(lambda: self.scroll(x=0, y=0.1))
        self.down.clicked.connect(lambda: self.scroll(x=0, y=-0.1))
        self.right.clicked.connect(lambda: self.scroll(x=0.2, y=0))
        self.left.clicked.connect(lambda: self.scroll(x=-0.2, y=0))
        self.play1.clicked.connect(lambda: self.play_sound(1))
        self.play2.clicked.connect(lambda: self.play_sound(2))
        self.default_speed.clicked.connect(self.PTimer)
        self.fast_speed.clicked.connect(lambda: self.speed(x=-100))
        self.speed_slow.clicked.connect(lambda: self.speed(x=100))
        self.pause1.clicked.connect(self.pause_sound)
        self.pause2.clicked.connect(self.stop_signal)
        self.color.currentTextChanged.connect(self.default_spectrogram)
        self.min_spectro_slider.valueChanged.connect(self.default_spectrogram)
        self.max_spectro_slider.valueChanged.connect(self.default_spectrogram)
        for i in range(2):
            self.plotWidgets[i].setYRange(0, 0.5, padding=0)
            self.plotWidgets[i].setXRange(0, 0.05, padding=0)

        for i in range(2, 4):
            self.plotWidgets[i].setXRange(0, 1000, padding=0)

    def PTimer(self):
            for i in range(2):
                self.timer = QtCore.QTimer()
                self.timer.setInterval(self.s)    
                self.timer.timeout.connect(self.PTimer)
                self.timer.start()
                xrange , yrange = self.plotWidgets[i].viewRange()
                ScaleValue = (xrange[1] - xrange[0])/50
                self.plotWidgets[i].setXRange(xrange[0]+ScaleValue, xrange[1]+ScaleValue, padding=0)
    
    def stop_signal(self):
        self.timer.stop()


    def speed(self,x):
        if -100<=x<=400:
            self.s += x
            self.PTimer()
        else:
            pass

    def newWindow(self):
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Functions(self.MainWindow)
        self.MainWindow.show()
        self.ui.initPlot()

    def initPlot(self):
        self.sig1.setBackground('#000')
        self.sig1.addLegend()
        for i in range(10):
            self.connect_sliders(i)

    def Importbutton1(self):
        filename = QFileDialog.getOpenFileName(None, 'choose signal', "*.wav;;")
        global file_path
        if filename[0]:
            file_path = filename[0]
        self.openFile(file_path)

    def openFile(self, file_path, Ampltuidesset=[], Timeset=[]):

        #self.Time = []
        global Ampltuides
        self.samples, self.sampling_rate = librosa.load(file_path, sr=None, mono=True, offset=0.0, duration=None)
        self.duration = len(self.samples) / self.sampling_rate
        for i in range(len(self.samples)):
            x = i/self.sampling_rate
            self.Time.append(x)

        self.Ampltuides = np.array(self.samples)
        n = np.min(self.Ampltuides)
        m = np.max(self.Ampltuides)
        self.modifiedAmplitudes = np.array(self.samples)
        freq=np.arange(1.0,(self.sampling_rate/2)+1)

        dec= 20*(np.log10(freq)*(-1))

        self.min_spectro_slider.setMinimum(min(dec))
        self.max_spectro_slider.setMinimum(min(dec))
        self.min_spectro_slider.setMaximum(max(dec)+1)
        self.max_spectro_slider.setMaximum(max(dec)+1)
        print(min(dec))
        print(max(dec))
        Timeset.append(self.Time)
        Ampltuidesset.append(self.Ampltuides)
        self.addPlot(Timeset[-1], Ampltuidesset[-1], color=(0, 0, 255), name="Original s(t)")  # First plot
        self.addPlot(Timeset[-1], Ampltuidesset[-1], color=(255, 255, 0), name="Modulated s(t)")  # Second plot
        global x_fourier
        x_fourier, y_fourier = self.getFFT(Ampltuidesset[-1])

        bandwidthLeftIdx = np.where(np.abs(y_fourier * self.duration / self.sampling_rate) > 0.005)[0].min()
        bandwidthRightIdx = np.where(np.abs(y_fourier * self.duration / self.sampling_rate) > 0.005)[0].max()

        self.bandwidthLeft = x_fourier[bandwidthLeftIdx]
        self.bandwidthRight = x_fourier[bandwidthRightIdx]

        self.frequency[0] = 0

        bandwidthRange = self.bandwidthRight - self.bandwidthLeft
        for i in range(1, 11):
            self.frequency[i] = int(i / 10 * bandwidthRange + self.bandwidthLeft)

        self.frequency[10] = x_fourier[-1] / self.duration

        for i in range(2):
            self.plotWidgets[i].plotItem.getViewBox().setLimits(xMin=0, xMax=x, yMin=n, yMax=m)

        self.addPlot(x_fourier, np.abs(y_fourier / self.sampling_rate * self.duration), color=(255, 255, 0), name="fourier_after")  # Third plot
        self.addPlot(x_fourier, np.abs(y_fourier / self.sampling_rate * self.duration), color=(0, 0, 255), name="fourier_before")  # Third plot
        self.default_spectrogram()

    def new(self, Ampltuides=[], Time=[]):
        self.Ampltuidesset = []
        self.Ampltuidesset.append(Ampltuides)
        self.Timeset = []
        self.Timeset.append(Time)
        self.openFile(self.Ampltuidesset, self.Timeset)

    def connect_sliders(self, index):
        self.Scales[index].valueChanged.connect(
            lambda: self.changeValue(index))

    def addPlot(self, x, y, color, name):

        plotIdx = len(self.plotsData)

        thisPlotData = [np.array(x), np.array(y)]

        self.plotsData.append(thisPlotData)
        pen = pg.mkPen(color, width=2)

        plotObj = self.plotWidgets[plotIdx].plot(
            thisPlotData[0], thisPlotData[1], name, pen=pen)

        self.plotsObjects.append(plotObj)
        plotObj.setData(thisPlotData[0], thisPlotData[1])

    def updatePlotData(self, plotIdx):
        self.plotsObjects[plotIdx].setData(
            self.plotsData[plotIdx][0], self.plotsData[plotIdx][1])

    def default_spectrogram(self):
        global cmap, min_slider, max_slider
        min_slider= self.min_spectro_slider.value()
        max_slider= self.max_spectro_slider.value()
        print(min_slider)
        print(max_slider)
        global fs
        fs=self.sampling_rate
        palette = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

        for i in range(len(palette)):
            if self.color.currentIndex() == i:
                cmap = palette[i]
        f = plot.figure()
        f.set_figwidth(8)
        f.set_figheight(6)
        plot.specgram(self.Ampltuides, Fs=fs, cmap=cmap) 
        cb1=plot.colorbar()
        plot.savefig('spectroo1.png', bbox_inches='tight')
        self.spectro1.setPixmap(QtGui.QPixmap('spectroo1.png'))
        self.spectro2.setPixmap(QtGui.QPixmap('spectroo1.png'))
        plot.specgram(y, Fs=fs, cmap=cmap, scale='dB', vmin=min_slider, vmax=max_slider)
        cb1.remove()
        plot.colorbar()
        plot.savefig('spectroo2.png', bbox_inches='tight')
        self.spectro2.setPixmap(QtGui.QPixmap('spectroo2.png'))
        self.pdf()
        fs = 44100
        out_f = 'out.wav'
        wavf.write(out_f, fs, y)
    def changeValue(self, slider_num):
        global y
        y = self.samples
        x_fourier, y_fourier = self.getFFT(y)

        for i, s in enumerate(self.Scales):
            val = s.value()

            l = int(self.frequency[i] * self.duration)
            r = int(self.frequency[i + 1] * self.duration)

            y_fourier[l:r] = y_fourier[l:r] * val

        y_mag_phase = self.multplication(y_fourier)
        y = self.getIFFT(y_mag_phase)

        self.plotsData[1][1] = y

        self.updatePlotData(1)
        self.plotsData[2][0] = x_fourier
        self.plotsData[2][1] = np.abs(y_fourier / self.sampling_rate * self.duration)
        self.default_spectrogram()
        self.updatePlotData(2)
    fig = plot.figure(figsize=(8.27, 11.7))  # set the size of the PDF

    def pdf(self):
        plot.subplot(221)
        plot.plot(self.Time, self.Ampltuides)
        plot.title("Original Signal")
        plot.xlim(0, 0.4)
        plot.subplot(222)
        plot.specgram(self.Ampltuides, Fs=fs, cmap=cmap)  # plot spectrogram of the signal
        plot.title("Original Spectrogram")
        plot.colorbar()
        plot.subplot(223)
        plot.plot(self.Time, y)
        plot.title("Modefied signal")
        plot.xlim(0, 0.4)
        plot.subplot(224)
        plot.specgram(y, Fs=fs, cmap=cmap, scale='dB', vmin=min_slider, vmax=max_slider)
        plot.title("Modefied spectrogram")
        plot.colorbar()
        plot.tight_layout()

    def save_pdf(self):
        plot.savefig('report.pdf', dpi=1000)

    def getFFT(self, y):
        x_fourier = fft.rfftfreq(len(self.samples), 1/self.sampling_rate)  # generate frequency vector
        y_fourier = fft.rfft(y)  # take real fft of signal

        return x_fourier, y_fourier

    def multplication(self, y_fourier):
        mag = np.abs(y_fourier)
        phase = np.angle(y_fourier)
        y_mag_phase = np.multiply(mag, np.exp(1j*phase))
        return y_mag_phase

    def getIFFT(self, y_mag_phase):
        y = fft.irfft(y_mag_phase)
        return y

    def play_sound(self, flag):
        if flag == 1:
            winsound.PlaySound(file_path, winsound.SND_ASYNC)
        if flag == 2:
            winsound.PlaySound('out.wav', winsound.SND_ASYNC)

    def pause_sound(self):
        winsound.PlaySound(None, winsound.SND_PURGE)

    def zoom(self, x, y):

        for i in range(2):
            self.plotWidgets[i].plotItem.getViewBox().scaleBy((x, y))

    def scroll(self, x, y):
        for i in range(2):
            self.plotWidgets[i].plotItem.getViewBox().translateBy(x=x, y=y)


app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

ui = Functions(MainWindow)
ui.initPlot()
MainWindow.show()
app.exec_()
