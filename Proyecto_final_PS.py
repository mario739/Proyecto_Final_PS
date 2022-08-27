import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc
from matplotlib.animation import FuncAnimation
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
#--------------------------------------
fig        = plt.figure()
fig.suptitle('Procesamiento de senales ECG', fontsize=16)

#Frecuencia de muestreo de las senales ECG y numero de muestras de la senal
fs          = 360
N           = 10000

#DATOS DEL FILTRO ELIMINA BANDA DE 60Hz
firData,=np.load("stop_band_60.npy").astype(float)
#firData,=np.load("average_10_stage2.npy").astype(float)
#firData,=np.load("average_30_stage4.npy").astype(float)
firData=np.insert(firData,0,firData[-1]) #ojo que pydfa me guarda 1 dato menos...
M          = len(firData)
nData         = np.arange(0,N+M-1,1)
firExtendedData=np.concatenate((firData,np.zeros(N-1)))
print(len(firExtendedData))

def x(n):
    ecg = electrocardiogram()[52000:62000]
    return ecg

tData=nData/fs
fData=nData*(fs/(N+M-1))-fs/2
xData=np.zeros(N+M-1)
xData[:N]+=x(tData[:N])

#SENAL ORIGINAL 
signalAxe = fig.add_subplot(3,3,1)
signalAxe.set_title("Senal original",rotation=0,fontsize=10,va="center")
signalLn, = plt.plot(tData,xData,'b-o',label="x",linewidth=3,alpha=0.6)
signalAxe.legend()
signalAxe.grid(True)
signalAxe.set_xlim(0,(N+M-2)/fs)
signalAxe.set_ylim(np.min(xData)-0.2,np.max(xData)+0.2)
convZoneLn = signalAxe.fill_between([0,0],10,-10,facecolor="yellow",alpha=0.5)


#DATOS DEL FILTRO PASA BANDA
data_pass_band,=np.load("band_pass_100_360hz.npy").astype(float)
data_pass_band=np.insert(data_pass_band,0,data_pass_band[-1]) #ojo que pydfa me guarda 1 dato menos...
M2       = len(data_pass_band)
nData_data_pass_band        = np.arange(0,N+M2-1,1)
firExtendedData_data_pass_band=np.concatenate((data_pass_band,np.zeros(N-1)))

tData_data_pass_band=nData_data_pass_band/fs
fData_data_pass_band=nData_data_pass_band*(fs/(N+M2-1))-fs/2
xData_data_pass_band=np.zeros(N+M2-1)
xData_data_pass_band[:N]+=x(tData_data_pass_band[:N])



# --------------------------------------

# IMPLENTACION DEL FILTRO PASA BANDA 0.5Hz A 100Hz
XAxe  = fig.add_subplot(3,3,2)
XAxe.set_title("Filtro pasa banda 0.5Hz a 100Hz",rotation=0,fontsize=10,va="center")
XData = np.fft.fft(xData_data_pass_band)

circularXData=np.fft.fftshift(XData)
XLn,  = plt.plot(fData_data_pass_band,np.abs(circularXData),'b-',label="X",linewidth=3,alpha=0.5)
XAxe.legend()
XAxe.grid(True)
XAxe.set_xlim(-fs/2,fs/2-fs/N)

HData=np.fft.fft(firExtendedData_data_pass_band)
circularHData=np.fft.fftshift(HData)
HAxe  = fig.add_subplot(3,3,5)
HLn,  = plt.plot(fData_data_pass_band,np.abs(circularHData),'g-',label="H",linewidth=3,alpha=0.5)
HAxe.legend()
HAxe.grid(True)
HAxe.set_xlim(-fs/2,fs/2-fs/N)


YAxe  = fig.add_subplot(3,3,8)
YData=XData*HData
circularYData=np.fft.fftshift(YData)
YLn,    = plt.plot(fData_data_pass_band,np.abs(circularYData),'r-',label = "Y",linewidth=3,alpha=0.8)
YAxe.legend()
YAxe.grid(True)
YAxe.set_ylim(np.min(np.abs(circularXData)),np.max(np.abs(circularXData)))
YAxe.set_xlim(-fs/2,fs/2-fs/N)



# IMPLEMENTACION DEL FILTRO ELIMINA BANDA DE 60Hz
XAxe  = fig.add_subplot(3,3,3)
XAxe.set_title("Filtro elimina banda 60Hz",rotation=0,fontsize=10,va="center")
XData = np.fft.fft(xData)

circularXData=np.fft.fftshift(XData)
XLn,  = plt.plot(fData,np.abs(circularXData),'b-',label="X",linewidth=3,alpha=0.5)
XAxe.legend()
XAxe.grid(True)
XAxe.set_xlim(-fs/2,fs/2-fs/N)

HData=np.fft.fft(firExtendedData)
circularHData=np.fft.fftshift(HData)
HAxe  = fig.add_subplot(3,3,6)
HLn,  = plt.plot(fData,np.abs(circularHData),'g-',label="H",linewidth=3,alpha=0.5)
HAxe.legend()
HAxe.grid(True)
HAxe.set_xlim(-fs/2,fs/2-fs/N)

YAxe  = fig.add_subplot(3,3,9)
YData=XData*HData
circularYData=np.fft.fftshift(YData)
YLn,    = plt.plot(fData,np.abs(circularYData),'r-',label = "Y",linewidth=3,alpha=0.8)
YAxe.legend()
YAxe.grid(True)
YAxe.set_ylim(np.min(np.abs(circularXData)),np.max(np.abs(circularXData)))
YAxe.set_xlim(-fs/2,fs/2-fs/N)


#SENAL 
convData=np.fft.ifft(YData)
#DETECCION DE LA FRECUENCIA CARDIACA 

#Funcion para carcular los picos de la senal
peaks, _ = find_peaks(convData, distance=150)

#Se calcula la diferencia entre cada punto y se saca el promedio
diff=np.diff(peaks)
prom=np.sum(diff)/len(diff)

#formula para sacar la Fc aprox
fc=(1/((1/fs)*prom))*60

#Funciones para plotear la grfica de l fc
found_peaks= fig.add_subplot(3,3,4)
found_peaks.set_title("Frecuencia Cardiaca Aprox= {0:.1f}bpm".format(fc),rotation=0,fontsize=10,va="center")
found_peak,=plt.plot(convData)
found_peak,=plt.plot(peaks, convData[peaks], "x")

plt.get_current_fig_manager().window.showMaximized()
plt.show()
