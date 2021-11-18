# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:59:09 2021

@author: Eow
"""


from imageio import imread
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

f=imread('noisyland.tif').astype(float)

filtroruido = np.ones((7,30))/(7*30)
pb1 = ndimage.convolve(f, filtroruido, mode='constant', cval=0)
pa1 = f-pb1

#criar um filtro com tamanho de bandas mais pequeno que o ruido
filtroruido2 = np.ones((1,30))/(1*30)

pb2 = ndimage.convolve(f, filtroruido2, mode='constant', cval=0)
pa2 = f-pb2

final = pb1+pa2
#Correção do Ruído vertical (tendo já sido corrigidas o ruído horizontal)

filtroruido3 = np.ones((30,5))/(5*30)
#Filtro que contem o ruído vertical dentro dele
pb3=ndimage.convolve(final,filtroruido3, mode='constant', cval=0)
pa3=final-pb3

filtroruido4=np.ones((30,1))/(1*30)
#Filtro que está contido dentro do ruído vertical
pb4=ndimage.convolve(final, filtroruido4, mode='constant', cval=0.0) 
#Aplicação do filtro passa-baixa ao ruído
pa4= final-pb4
#Cálculo do respetivo filtro passa-alta

ff= pb3+pa4  #Correção do ruído vertical

pb5=ndimage.convolve(ff,filtroruido3, mode='constant', cval=0)
pb6=ndimage.convolve(ff,filtroruido4,mode='constant',cval=0)
pa6=ff-pb6

fff=pb5+pa6


ffff=ndimage.median_filter(fff,footprint=filtroruido4)
k=0.8
u=fff+k*(fff-ffff)


plt.figure()
plt.subplot(121); plt.imshow(fff, 'gray',vmin=0,vmax=255); plt.title('original'); plt.axis('off')
plt.subplot(122); plt.imshow(u, 'gray', vmin=0,vmax=255); plt.title('final'); plt.axis('off')
# plt.subplot(223); plt.imshow(ff, 'gray'); plt.title('final'); plt.axis('off')