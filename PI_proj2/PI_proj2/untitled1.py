# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:44:29 2021

@author: silam
"""
from skimage.morphology import disk, rectangle, erosion, dilation, opening, closing, local_minima, local_maxima, watershed, binary_erosion, binary_dilation, binary_opening, binary_closing, reconstruction, skeletonize
from imageio import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
plt.close('all')

# ex 2

#Importação e divisão da imagem RGB nas 3 bandas
Ik0=imread("ik02.tif")
img2_r=Ik0[:,:,0]
img2_g=Ik0[:,:,1]
img2_b=Ik0[:,:,2]

#Média das 3 bandas
Ik2=np.uint(img2_r/3 + img2_g/3 + img2_b/3)

Ik=ndimage.filters.gaussian_filter(Ik2, 1)

plt.imshow(Ik,'gray')

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


#Threshold dos valores da imagem da média das bandas para isolar as estradas
t2 = 230
t1 = 190
X = Ik <= t1
Y = Ik >= t2

#Aplicação da reconstrução à imagem binária anterior para obter a imagem binária desejada
M = np.logical_not(np.logical_or(X,Y))
Z = reconstrucao_bin(M, np.logical_and(binary_dilation(Y, rectangle(3, 3)), M))
TH3 = np.logical_or(Y, Z) #Imagem binária desejada

#Filtragem da imagem binária resultante do passo anterior
dil = binary_dilation(TH3, rectangle(9, 9)) #Dilatação
fecho = binary_erosion(dil,rectangle(7, 7)) #Erosão da dilatação que resulta no fecho da img

# dil = binary_erosion(TH3,rectangle(2, 2))
# fecho = binary_dilation(dil, rectangle(7, 7)) #Dilatação



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

ee=rectangle(2,2)
#watershed para cálculo da linha do rio
marker,n= ndimage.label(min_reg)
W = watershed(fecho, marker, mask = np.ones(fecho.shape))
Iint = W- erosion(W, ee) #Linha do rio

#linha do rio binaria
Iint_bin=Iint.astype(bool)

#skeletonize da linha do rio para menor espessura
sk1=skeletonize(Iint_bin)

a=reconstrucao_bin(TH3,sk1)

#Multiplicação da imagem do passo anterior pela imagem original para obter o output desejado
estradas_final = np.copy(Ik0) #Cópia da imagem original para obter as cores no output
for i in range(Ik0.shape[2]):
    estradas_final[:,:,i] = estradas_final[:,:,i] * a

# #Output final
# plt.figure(figsize=(15,10))
# plt.subplot(121);plt.imshow(img2,'gray')
# plt.title('Imagem original'); plt.axis('off')
# plt.subplot(122);plt.imshow(estradas_final,'gray')
# plt.title('Estradas'); plt.axis('off')


plt.subplot(231); plt.imshow(Ik, 'gray'); plt.title('Original Cinzento'); plt.axis('off')
plt.subplot(232); plt.imshow(dil,'gray'); plt.title('Limiarizacao'); plt.axis('off')
plt.subplot(233); plt.imshow(fecho, 'gray'); plt.title('Limiarizacao'); plt.axis('off')
plt.subplot(234); plt.imshow(Iint_bin, 'gray'); plt.title('Limiarizacao'); plt.axis('off')
plt.subplot(235); plt.imshow(a, 'gray'); plt.title('Limiarizacao'); plt.axis('off')
plt.subplot(236); plt.imshow(estradas_final, 'gray'); plt.title('Limiarizacao'); plt.axis('off')
