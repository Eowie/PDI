# -*- coding: utf-8 -*-

"""
Nome: João Miguel Pinto Ferreira
nº aluno: 50214
curso: Engenharia Geoespacial
disciplina: Processamento Digital de Imagem
Projeto 2, exercicio 1.2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
from copy import deepcopy
from imageio import imread
from skimage.morphology import disk, square, rectangle, reconstruction, \
binary_erosion, binary_dilation, binary_opening, binary_closing, dilation, erosion
from skimage.morphology import watershed
img=imread('lsat01.tif');
R=img[:,:,0]
G=img[:,:,1]
B=img[:,:,2]

#media das componentes rgb
I=(img[:,:,0].astype(float)+img[:,:,1].astype(float)+img[:,:,2].astype(float))/3

#após analise do histograma detetei que o valor de 27 é o valor anterior a um dos picos existentes
nova=I<27
#aplicou-se um kernel (5,5)
kernel1=rectangle(5,5)

#aplicou-se uma erosão
Er=binary_erosion(nova,kernel1)

#reconstrui-se o objeto através de uma reconstrução binária
def reconstrucao_bin(mask,marker):
    se=rectangle(3,3)
    ant=np.zeros(mask.shape)
    R=np.copy(marker)
    while np.array_equal(R,ant)==False:
        ant=R
        R=np.logical_and(binary_dilation(R,se),mask)
    return R
#obtencao da imagem final
binaria=reconstrucao_bin(nova,Er)

#alinea 1.2
F=binaria
#função distância
D=scipy.ndimage.distance_transform_edt(F==1)

def reconstrucao_dual(mask,marker):
    a=1
    ee=rectangle(2,2)
    while a!=0:
        E=erosion(marker,ee)
        R=np.maximum(mask.astype(float),E.astype(float))
        a=np.count_nonzero(marker!=R)
        marker=deepcopy(R)
    return R


D_mais1=D+1

#cálculo dos minimos, reconstrução Dual D+1  - D
minimos=reconstrucao_dual(D,D_mais1)-D

markers,n=ndimage.label(minimos)

#aplicação da função watershed
Ws=watershed(D,markers,mask=np.ones(I.shape))
d=2
ee=disk(d)

#Utilização de um disco de 2 para conseguir destacar bem a linha
linha=Ws-erosion(Ws,ee)

#obtenção da linha representativa do leito do rio
linha_f=np.logical_not(linha==0)*255

#Imagem final
L=linha_f+I

plt.subplot(121),plt.imshow(img,'gray');plt.axis('off')
plt.title('original')

plt.subplot(122),plt.imshow(L,'gray');plt.axis('off')
plt.title('meio')
