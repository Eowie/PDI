# -*- coding: utf-8 -*-

"""
Nome: João Miguel Pinto Ferreira
nº aluno: 50214
curso: Engenharia Geoespacial
disciplina: Processamento Digital de Imagem
Projeto 2, exercicio 2
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from imageio import imread
from skimage.morphology import disk, square, rectangle, reconstruction, \
binary_erosion, binary_dilation, binary_opening, binary_closing, dilation, erosion


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

# transformacao das diferentes bandas
teste_r=R>200
teste_g=G>200
teste_b=B>180
# interseção das 3 imagens binárias em uma.
inter=teste_r&teste_g&teste_b

# criacao de filtros com orientacao aproximada das estradas
filtro=np.eye(7)
filtro2=np.rot90(filtro)
#criacao de mais um filtro para ficar com a estrada superior
filtro3=[[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
# operacões de abertura  com os diferentes filtros
a=binary_opening(inter,filtro)
b=binary_opening(inter,filtro2)
c=binary_opening(inter,filtro3)

juncao=a|b|c
filtro4=disk(2)
# operacao de abertura à imagem resultante das 3 aberturas com os filtros anteriores
final3=binary_opening(juncao,filtro4)
#recontrucao binaia
d=reconstrucao_bin(inter,juncao)

#operacao de fecho
e=binary_closing(d,juncao)

#multiplicação de cada banda pela imagem binária
R=R*e
G=G*e
B=B*e


rgb = np.dstack((R,G,B)) 
plt.subplot(121),plt.imshow(img,'gray');plt.axis('off')
plt.title('red')

plt.subplot(122),plt.imshow(rgb);plt.axis('off')
plt.title('finalissimo')
