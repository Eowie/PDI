# -*- coding: utf-8 -*-

"""
Nome: João Miguel Pinto Ferreira
nº aluno: 50214
curso: Engenharia Geoespacial
disciplina: Processamento Digital de Imagem
Projeto 2, exercicio 1.1
"""

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.morphology import disk, square, rectangle, reconstruction, \
binary_erosion, binary_dilation, binary_opening, binary_closing, dilation, erosion
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


h,r=np.histogram(R, bins=256, range=(0,256))
plt.figure(figsize=(12,2))
plt.subplot(131);plt.plot(h,'g')
plt.title('Histograma')


plt.subplot(121),plt.imshow(img,'gray');plt.axis('off')
plt.title('original')

plt.subplot(122),plt.imshow(binaria,'gray');plt.axis('off')
plt.title('Binária leito do rio')


