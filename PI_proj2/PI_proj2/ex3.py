# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:31:43 2019

@author: joao

Nome: João Miguel Pinto Ferreira
nº aluno: 50214
curso: Engenharia Geoespacial
disciplina: Processamento Digital de Imagem
Projeto 2, exercicio 3
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from imageio import imread
from skimage.morphology import disk, rectangle, \
binary_dilation, binary_opening, binary_closing


def reconstrucao_bin(mask,marker):
    se=rectangle(3,3)
    ant=np.zeros(mask.shape)
    R=deepcopy(marker)
    while np.array_equal(R,ant)==False:
        ant=R
        R=np.logical_and(binary_dilation(R,se),mask)
    return R

# Abre a imagem
img = imread('ik02.tif')
#separa as bandas
R=img[:,:,0]
G=img[:,:,1]
B=img[:,:,2]

#operações com bandas para obter um histograma que permita extrair as casas
f=abs((R-G+(G-B))+45)

#histograma de f
h,r=np.histogram(f, bins=256, range=(0,256))


# limiarizaçã de histograma pelo método da	máxima	distância
xmax=np.where(h==np.max(h)); xmax=xmax[0][0]
ymax=h[xmax]
x1=xmax
y1=ymax
x2=255
y2=h[x2]
a1=(y2-y1)/(x2-x1)
b1=y1-a1*x1
a2=-a1
b2=np.zeros((256)); x=np.copy(b2); y=np.copy(b2);d=np.copy(b2)
for i in range (xmax,255):
    b2[i]=h[i]-a2*i
    x[i]=(b1-b2[i])/(a2-a1)
    y[i]=a2*(b1-b2[i])/(a2-a1)+b2[i]
    d[i]=np.sqrt((x[i]-i)**2+(y[i]-h[i])**2)
t7=np.where(d==np.max(d)); t7=t7[0][0]

#criar imagem binária através da máxima distância
binaria=f>t7
#filtro é um disco pois funciona melhor neste caso
filtro=disk(2)
#operação de abertura
abertura=binary_opening(binaria,filtro)
# reconstrução binária
rec=reconstrucao_bin(binaria,abertura)
# operação de fecho
fecho=binary_closing(rec,filtro)

# multiplicação de cada uma das bandas pela imagem binária
r_f=R*fecho
g_f=G*fecho
b_f=B*fecho

#stack das bandas para criação de imagem RGB
rgb=np.dstack((r_f,g_f,b_f)) 


plt.subplot(121),plt.imshow(img,'gray');plt.axis('off')
plt.title('original')

plt.subplot(122),plt.imshow(rgb);plt.axis('off')
plt.title('casas')

#histograma da imagem f
plt.figure(figsize=(12,2))
plt.subplot(131);plt.plot(h,'g')
plt.title('Histograma')
