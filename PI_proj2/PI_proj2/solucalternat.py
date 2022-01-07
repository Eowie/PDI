# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:18:01 2021

@author: silam
"""

from skimage.morphology import disk, rectangle, erosion, dilation, opening, closing, local_minima, local_maxima, watershed, binary_erosion, binary_dilation, binary_opening, binary_closing, reconstruction, skeletonize
from imageio import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

Ik0=imread("ik02.tif")
img2_r=Ik0[:,:,0]
img2_g=Ik0[:,:,1]
img2_b=Ik0[:,:,2]

#Média das 3 bandas
Ik=np.uint(img2_r/3 + img2_g/3 + img2_b/3)
Ik=ndimage.filters.gaussian_filter(Ik, 1)

#Função da reconstrucao geodesica binaria
def reconstrucao_bin (mask,marker):
    se= rectangle(3,3)
    ant=np.zeros(mask.shape)
    R=np.copy(marker)
    while np.array_equal(R,ant) == False:
        ant = R
        R=np.logical_and(binary_dilation(R,se),mask)
    return R

g,r = np.histogram(Ik, bins=256, range=(0,256))


dif=img2_r.astype(float)-(img2_b.astype(float))
dif[dif<0]=255
Ik=dif

#Threshold dos valores da imagem da média das bandas para isolar as estradas
t2 = 50
t1 = 70
X = Ik >= t1
Y = Ik <= t2

#Aplicação da reconstrução à imagem binária anterior para obter a imagem binária desejada
M = np.logical_not(np.logical_or(X,Y))
Z = reconstrucao_bin(M, np.logical_and(binary_dilation(Y, rectangle(3, 3)), M))
TH3 = np.logical_or(Y, Z) #Imagem binária desejada

#Filtragem da imagem binária resultante do passo anterior
dil = binary_dilation(TH3, rectangle(15,15)) #Dilatação
fecho = binary_erosion(dil,rectangle(15,15)) #Erosão da dilatação que resulta no fecho da img


def reconstrucao_dual(mask,marker):
    a=1
    ee=rectangle(3,3)
    while a!=0:
        E=erosion(marker,ee)
        R=np.maximum(mask.astype(float),E.astype(float))
        a=np.count_nonzero(marker!=R)
        marker=deepcopy(R)
    return R

#Medicao da Distancia
d= ndimage.distance_transform_edt(fecho==1)

#Reconstrucao dual da distancia
d1=d+1
rd_d= reconstrucao_dual(d,d1)

#minimo regional
min_reg = rd_d - d

ee=rectangle(3,3)
#watershed para cálculo da linha do rio
marker,n= ndimage.label(min_reg)
W = watershed(fecho, marker, mask = np.ones(fecho.shape))
Iint = W- erosion(W, ee) #Linha do rio

#linha do rio binaria
Iint_bin=Iint.astype(bool)

a=reconstrucao_bin(TH3,Iint_bin)


#Multiplicação da imagem do passo anterior pela imagem original para obter o output desejado
estradas_final = np.copy(Ik0) #Cópia da imagem original para obter as cores no output
for i in range(Ik0.shape[2]):
    estradas_final[:,:,i] = estradas_final[:,:,i] * a








Ik=estradas_final[:,:,0]






t2a = 230
t1a = 200
X = Ik <= t1a
Y = Ik >= t2a

#Aplicação da reconstrução à imagem binária anterior para obter a imagem binária desejada
M = np.logical_not(np.logical_or(X,Y))
Z = reconstrucao_bin(M, np.logical_and(binary_dilation(Y, rectangle(3, 3)), M))
TH3 = np.logical_or(Y, Z) #Imagem binária desejada

#Filtragem da imagem binária resultante do passo anterior
dil = binary_dilation(TH3, rectangle(15,15)) #Dilatação
fecho = binary_erosion(dil,rectangle(15,15)) #Erosão da dilatação que resulta no fecho da img


def reconstrucao_dual(mask,marker):
    a=1
    ee=rectangle(3,3)
    while a!=0:
        E=erosion(marker,ee)
        R=np.maximum(mask.astype(float),E.astype(float))
        a=np.count_nonzero(marker!=R)
        marker=deepcopy(R)
    return R

#Medicao da Distancia
d= ndimage.distance_transform_edt(fecho==1)

#Reconstrucao dual da distancia
d1=d+1
rd_d= reconstrucao_dual(d,d1)

#minimo regional
min_reg = rd_d - d

ee=rectangle(3,3)
#watershed para cálculo da linha do rio
marker,n= ndimage.label(min_reg)
W = watershed(fecho, marker, mask = np.ones(fecho.shape))
Iint = W- erosion(W, ee) #Linha do rio

#linha do rio binaria
Iint_bin=Iint.astype(bool)

a1=reconstrucao_bin(TH3,Iint_bin)



estradas_finala = np.copy(Ik0) #Cópia da imagem original para obter as cores no output
for i in range(Ik0.shape[2]):
    estradas_finala[:,:,i] = estradas_finala[:,:,i] * a1
    
    
    
    
    






plt.figure()
plt.subplot(231); plt.imshow(Ik0, 'gray'); plt.title('Original Cinzento'); plt.axis('off')
plt.subplot(232); plt.imshow(dil,'gray'); plt.title('fecho'); plt.axis('off')
plt.subplot(233); plt.imshow(fecho, 'gray'); plt.title('sk1'); plt.axis('off')
plt.subplot(234); plt.imshow(Iint, 'gray'); plt.title('Iint_bin'); plt.axis('off')
plt.subplot(235); plt.imshow(a, 'gray'); plt.title('test'); plt.axis('off')
plt.subplot(236); plt.imshow(estradas_finala, 'gray'); plt.title('a'); plt.axis('off')