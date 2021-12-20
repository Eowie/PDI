# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:41:31 2020

@author: Miguel
"""
from skimage.morphology import disk, rectangle, erosion, dilation, opening, closing, local_minima, local_maxima, watershed, binary_erosion, binary_dilation, binary_opening, binary_closing
from imageio import imread, imwrite
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# plt.close('all')

#%%
# EXERCICIO 1
# 1.1

#Importação e divisão da imagem RGB nas 3 bandas
img1=plt.imread("lsat01.tif")
img1_r=img1[:,:,0]
img1_g=img1[:,:,1]
img1_b=img1[:,:,2]

#média das bandas
IMG1=np.uint(img1_r/3 + img1_g/3 + img1_b/3)

#Threshold da imagem para produção da imagem binaria
lim=24 #valor maximo onde apenas se incluem pixeis do leito do rio

#Threshold da imagem
IMG1_bin=np.zeros(IMG1.shape)
IMG1_bin[IMG1>lim] = 0
IMG1_bin[IMG1<lim] = 1

#1.2
#Abertura e fecho da imagem binária originada pelo threshold
ee=rectangle(3,3)
abertura = binary_opening(IMG1_bin, ee)
fecho = binary_closing(abertura,ee)

#Funcao reconstrução dual
def reconstrucao_dual(mask,marker):
    a=1
    ee=rectangle(3,3)
    while a!=0:
        E=erosion(marker,ee)
        R=np.maximum(mask.astype(float),E.astype(float))
        a=np.count_nonzero(marker!=R)
        marker=deepcopy(R)
    return R


#Distancia euclidiana 
funcao_dist= ndimage.distance_transform_edt(fecho==1)

#Reconstrucao dual da funcao distancia
mini_dist=funcao_dist+1
reconstrucao_dual_dist= reconstrucao_dual(funcao_dist,mini_dist)

#minimo regional
minimo_reg = reconstrucao_dual_dist - funcao_dist



#Função watershed para cálculo da linha do rio
marker,n= ndimage.label(minimo_reg)
W = watershed(fecho, marker, mask = np.ones(fecho.shape))
Lint = W- erosion(W, ee) #Linha do rio
Lint_bin=Lint>0

#Imagem final com a linha sobreposta
Linha_imagem = IMG1 * np.logical_not(Lint_bin) + Lint_bin*255


#Output final
plt.figure(figsize=(15,10))
plt.subplot(121);plt.imshow(IMG1_bin,'gray')
plt.title('Imagem binária'); plt.axis('off')
plt.subplot(122);plt.imshow(Linha_imagem,'gray')
plt.title('Linha média do leito do rio'); plt.axis('off')

#%%
#EXERCICIO 2

#Importação e divisão da imagem RGB nas 3 bandas
img2=imread("ik02.tif")
img2_r=img2[:,:,0]
img2_g=img2[:,:,1]
img2_b=img2[:,:,2]

#Média das 3 bandas
IMG2=np.uint(img2_r/3 + img2_g/3 + img2_b/3)

#Função da reconstrucao geodesica binaria
def reconstrucao_bin (mask,marker):
    se= rectangle(3,3)
    ant=np.zeros(mask.shape)
    R=np.copy(marker)
    while np.array_equal(R,ant) == False:
        ant = R
        R=np.logical_and(binary_dilation(R,se),mask)
    return R

#Threshold dos valores da imagem da média das bandas para isolar as estradas
t2 = 250
t1 = 198
X = IMG2 <= t1
Y = IMG2 >= t2

#Aplicação da reconstrução à imagem binária anterior para obter a imagem binária desejada
M = np.logical_not(np.logical_or(X,Y))
Z = reconstrucao_bin(M, np.logical_and(binary_dilation(Y, rectangle(3, 3)), M))
TH3 = np.logical_or(Y, Z) #Imagem binária desejada

#Filtragem da imagem binária resultante do passo anterior
Dil = binary_dilation(TH3, rectangle(3, 3)) #Dilatação
fecho = binary_erosion(Dil,rectangle(3, 3)) #Erosão da dilatação que resulta no fecho da img

#Multiplicação da imagem do passo anterior pela imagem original para obter o output desejado
estradas_final = np.copy(img2) #Cópia da imagem original para obter as cores no output
for i in range(img2.shape[2]):
    estradas_final[:,:,i] = estradas_final[:,:,i] * fecho

#Output final
plt.figure(figsize=(15,10))
plt.subplot(121);plt.imshow(img2,'gray')
plt.title('Imagem original'); plt.axis('off')
plt.subplot(122);plt.imshow(estradas_final,'gray')
plt.title('Estradas'); plt.axis('off')

# #%%
# # EXERCICIO 3

# #Importação e divisão da imagem RGB nas 3 bandas
# img2=imread("ik02.tif")
# img2_r=img2[:,:,0]
# img2_g=img2[:,:,1]
# img2_b=img2[:,:,2]

# #Diferença entre a banda do vermelho e do azul para sobressair os telhados
# dif=img2_r.astype(float)-(img2_b.astype(float))

# #Histograma
# h, r = np.histogram(dif, bins = 256, range=(0,256))

# # Funçao distancia maxima
# xmax=np.where(h==np.max(h));xmax=xmax[0][0] #valor máximo em x no histograma
# x1=xmax
# y1= h[x1] #valor de y correspondente ao valor de x máximo
# x2=255 
# y2=h[x2] #valor de y correspondente ao valor de x=255
# a1=(y2-y1)/(x2-x1) #declive da curva
# b1=y1-a1*x1
# a2=-(1/a1)
# b2=np.zeros((256)); x=np.copy(b2); y=np.copy(b2); d=np.copy(b2)
# for i in range(x1,x2):
#     b2[i]=h[i]-a2*i
#     x[i]=(b1-b2[i])/(a2-a1)
#     y[i]=a2*(b1-b2[i])/(a2-a1)+b2[i]
#     d[i]=np.sqrt((x[i]-i)**2+(y[i]-h[i])**2)
    
# th=np.where(d==np.max(d));th=th[0][0] #Valor que vai originar o threshold
# #threshold
# telhados = dif < th

# #Multiplicação da imagem por 1 para ter a imagem binária com valores 0 e 1
# telhados=telhados*1 

# #Ciclo que suaviza a imagem e multiplica a imagem binária anterior pela imagem original
# telhados_final = np.copy(img2)
# for i in range(img2.shape[2]):
#     telhados_final[:,:,i] = telhados_final[:,:,i] * np.logical_not(telhados) #Terá de ser o inverso da imagem binária
#     telhados_final[:,:,i] = erosion(telhados_final[:,:,i], rectangle(3,3))
#     telhados_final[:,:,i] = dilation(telhados_final[:,:,i], rectangle(3,3))


# #Output final
# plt.figure(figsize=(15,10))
# plt.subplot(121);plt.imshow(img2,'gray')
# plt.title('Imagem original'); plt.axis('off')
# plt.subplot(122);plt.imshow(telhados_final,'gray')
# plt.title('Telhados das casas'); plt.axis('off')

