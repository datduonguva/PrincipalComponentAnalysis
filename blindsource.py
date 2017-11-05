import numpy as np
from pca import *
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA as PCAA
from sklearn.decomposition import FastICA

X = np.loadtxt('mixture_of_signal.txt')

ica = FastICA(n_components=4)
signal_ica = ica.fit_transform(X)
mixing_matrix = ica.mixing_

pca = PCA(n_components=4)
signal_pca = pca.fit_transform(X)

pcaa = PCAA(n_components=4)
signal_pcaa = pcaa.fit_transform(X)

models = [X, signal_ica, signal_pca, signal_pcaa]

colors = ['blue', 'red', 'black', 'green']

plt.figure()
plt.title('Input signal')
for i in range(len(X[0])):
	plt.plot(X[:, i], color = colors[i])


plt.figure()
plt.title('ICA')
plt.subplots_adjust(wspace =1, hspace=1)
for i in range(4):
	plt.subplot(4, 1, i+1)
	plt.title('Signal '+str(i+1))
	plt.plot(signal_ica[:, i], color=colors[i])


plt.figure()
plt.title('PCA')
plt.subplots_adjust(wspace =1, hspace=1)
for i in range(4):
	plt.subplot(4, 1, i+1)
	plt.title('Signal '+str(i+1))
	plt.plot(signal_pca[:, i], color=colors[i])


plt.figure()
plt.title('PCAA')
plt.subplots_adjust(wspace =1, hspace=1)
for i in range(4):
	plt.subplot(4, 1, i+1)
	plt.title('Signal '+str(i+1))
	plt.plot(signal_pcaa[:, i], color=colors[i])
plt.show()

