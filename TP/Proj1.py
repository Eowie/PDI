# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:10:59 2021

@author: Eow
"""

from imageio import imread
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

f=imread('noisyland.tif').astype(float)

filtro_pb = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]],
                  dtype='float')/9

#aplicar um filtro de passa baixa e aplicar de seguida um filtro passa alta sobre o resultado
f_pb = ndimage.convolve(f,filtro_pb,mode='constant', cval=0)

filtro_pa = np.array([[-1,-1,-1],
                      [-1, 8,-1],
                      [-1,-1,-1]])/9

f_pa = ndimage.convolve(f, filtro_pa, mode='constant', cval=0)

#soma devera ser igual 'a imagem original
somaf=f_pb+f_pa                    
                        

#criar um filtro com tamanho de bandas que contenham o ruido
filtroruido = np.ones((301,51))/(51*301)
pb1 = ndimage.convolve(f, filtroruido, mode='constant', cval=0)
pa1 = f-pb1

#criar um filtro com tamanho de bandas mais pequeno que o ruido
#se mudar-mos o tamanho dos filtros podemos tentar encontrar uma melhor solucao
filtroruido2 = np.ones((301,3))/(3*301)

pb2 = ndimage.convolve(f, filtroruido2, mode='constant', cval=0)
pa2 = f-pb2

#ver o efeito de cada um dos filtros
plt.figure()
plt.subplot(221); plt.imshow(f, 'gray'); plt.title('original')
plt.subplot(222); plt.imshow(pb1, 'gray'); plt.title('Passa Baixa')
plt.subplot(243); plt.imshow(pa1, 'gray', vmin=0, vmax=255); plt.title('Passa Alta')


final = pb1+pa2

plt.subplot(121); plt.imshow(f, 'gray'); plt.title('original')
plt.subplot(122); plt.imshow(final, 'gray'); plt.title('final'); plt.axis('off')

