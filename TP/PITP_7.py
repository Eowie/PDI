# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:14:23 2021

@author: Eow
"""

import os
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, erosion, dilation, opening, \
    closing, local_minima, local_maxima, watershed,binary_opening, binary_dilation , reconstruction 
from scipy import ndimage
from copy import deepcopy

from mpl_toolkits import mplot3d

    
plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

#%%
#Ex 1.1


f=imread('Marilyn.tif')

g,r=np.histogram(f, bins=256, range=(0,256))
m = np.mean(f)
c1=f<=m
c2=~c1

# plt.figure(figsize=(20,15))
# plt.subplot(151); plt.imshow(f, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(152); plt.imshow(c1, 'gray'); plt.title('C1'); plt.axis('off')
# plt.subplot(153); plt.imshow(c2, 'gray'); plt.title('C2'); plt.axis('off')
# plt.subplot(153); plt.plot(g); plt.title('Histograma'); plt.axis('off')

#%%
#Ex 1.2

img=imread('hematite_mars.tif')

h,r=np.histogram(img, bins=256, range=(0,256))

t3_1= 120
t3_2= 180
X=img<=t3_1
Y=img>=t3_2
#para tentar eliminar mais background tentamos este:
Y=reconstruction(binary_opening(img>=t3_2, disk(2)),Y).astype(bool)

# fracos= ~(X|Y)
#alternativa de escrita dos fracos=M
M=np.logical_not(np.logical_or(X,Y))

#fazemos a dilatacao e intersectamos com os fracos para descobrir quais
#os mais proximos dos objetos
#dil=binary_dilation(Y,disk(1))&fracos
#rec=reconstruction(dil, fracos)

Z=reconstruction(np.logical_and(binary_dilation(Y,disk(1)),M),M).astype(bool)

# TH1=Y|Z same as below
TH3=np.logical_or(Y,Z)

# plt.figure()
# plt.plot(h)


# plt.figure(figsize=(20,15))
# plt.subplot(231); plt.imshow(img, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(232); plt.imshow(X, 'gray'); plt.title('X'); plt.axis('off')
# plt.subplot(233); plt.imshow(Y, 'gray'); plt.title('Px Fortes'); plt.axis('off')
# plt.subplot(234); plt.imshow(M, 'gray'); plt.title('Fracos'); plt.axis('off')
# plt.subplot(235); plt.imshow(Z, 'gray'); plt.title('Z'); plt.axis('off')
# plt.subplot(236); plt.imshow(TH3, 'gray'); plt.title('TH3'); plt.axis('off')

# plt.figure(figsize=(20,15))
# plt.subplot(231); plt.imshow(img, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(232); plt.imshow(TH3*img, 'gray'); plt.title('Final'); plt.axis('off')

#%% Ex 1.3

zebra=imread('zebra01.tif')
#media dos valores de cinzento da imagem nas tres bandas.
zebra=zebra[:,:,0]
# zebra=((zebra[:,:,0].astype(float)+zebra[:,:,1].astype(float)+zebra[:,:,2].astype(float))/3)

uu,r=np.histogram(zebra, bins=256, range=(0,256))

#vamos procurar a distancia maxima entre 2 pontos

# d=np.sqrt((y2-y1)**2+(x2-x1)**2)

#reta para o pico do histograma
# reta vai ser y=mx+b
#b vai ser o valor do histograma na primeira posicao
b1=uu[0]
#declive
ymax=np.max(uu)
xmax=np.where(uu==ymax)[0][0]
x1=0
y1=b1
x2=xmax
y2=ymax

m1=((y2-y1)/(x2-x1))
# m1=(y2-b1)/x2

m2=-(1/m1)

#interseccao das duas retas
#y1=m1x+b1
#y2=m2x+b2
#igualando as duas temos que
#y=m2*((b2-b1)/(m1-m2))+b2

b2=[]
x4=[]
y4=[]
dist=[]

for t in range (xmax):
    b2.append(uu[t]-m2*t)
    x4.append((b2[t]-b1)/(m1-m2))
    y4.append(m2*((b2[t]-b1)/(m1-m2))+b2[t])
    dist.append(np.sqrt((x4[t]-t)**2+(y4[t])-uu[t]))
    
th=np.where(dist==np.max(dist))[0][0]
#th=np.argmax(d) igual ao de cima


# plt.figure(figsize=(20,15))
# plt.subplot(231); plt.imshow(zebra, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(232); plt.plot(uu); plt.title('Histograma'); plt.axis('off')
# plt.subplot(233); plt.imshow(zebra<=th,'gray'); plt.title('distmax'); plt.axis('off')
# plt.subplot(234); plt.imshow(zebra>=th,'gray'); plt.title('distmax'); plt.axis('off')

# plt.figure()
# plt.plot(uu)
# plt.plot(th,h[th],'or')


#%% Ex 1.4


#imagem marilyn 'e a f
#g,r=np.histogram(f, bins=256, range=(0,256))

variancia=[]
lin,col=f.shape

# for t in range (255):
#     th_e=f<=t
#     th_d=f>t
#     we=np.sum(th_e)
#     wd=np.sum(th_d)
#     ima_e = th_e*f
#     ima_d = th_d*f
#     var_e=np.var(np.nonzero(ima_e))
#     var_d=np.var(np.nonzero(ima_d))
#     variancia.append((we*var_e)+(wd*var_d))
    
    
# plt.figure()
# plt.plot(variancia)

# th1=[]

# for item in variancia:
#     if str(item)!= 'nan':
#         th1.append(item)
        
# th=np.argmin(th1)

for t in range (256):
    th_e=f<=t
    th_d=f>t
    we=np.sum(th_e)
    wd=np.sum(th_d)
    if we==0:
        variancia.append(np.var(f)*lin*col)
    elif wd==0:
        variancia.append(np.var(f)*lin*col)
    else:
        ima_e=th_e*f
        ima_d=th_d*f
        var_e=np.var(f[np.nonzero(ima_e)])
        var_d=np.var(f[np.nonzero(ima_d)])
        variancia.append(we*var_e+wd*var_d)
    
    
th=np.argmin(variancia)    

plt.figure()
plt.plot(variancia)
plt.plot(np.argmin(variancia),variancia[th],'or')

plt.figure()
plt.subplot(111); plt.imshow(f>=th, 'gray')

#%%
#ex 2

rgb=imread('ik01.tif')
lin,col,ch = rgb.shape

#eixos de 0 a 255 - nr de tons de cinzento da imagem
hmv=np.zeros((256,256,3))
for i in range (lin):
    for j in range (col):
        hmv[rgb[i,j,0],rgb[i,j,1],0]+=1
        hmv[rgb[i,j,0],rgb[i,j,2],1]+=1
        hmv[rgb[i,j,1],rgb[i,j,2],2]+=1


plt.figure(figsize=(14,5))
plt.subplot(241); plt.imshow(np.uint8(hmv))
plt.subplot(242); plt.imshow(np.uint8(hmv[:,:,0]), 'gray')
plt.subplot(243); plt.imshow(np.uint8(hmv[:,:,1]), 'gray')
plt.subplot(244); plt.imshow(np.uint8(hmv[:,:,2]), 'gray')
plt.subplot(245); plt.imshow(hmv)
plt.subplot(246); plt.imshow(hmv[:,:,0]>0)
plt.subplot(247); plt.imshow(hmv[:,:,1]>0)
plt.subplot(248); plt.imshow(hmv[:,:,2]>0)




fig = plt.figure(figsize=(14,5))
cor = ['r','g','b']
for i in range (ch):
    zdata = np.reshape(hmv[:,:,i],(1,256**2))
    m=np.max(zdata)
    zdata[zdata==0] =np.nan
    vx, vy = np.linspace(0,255,256), np.linspace (0,255,256)
    xy= np.meshgrid(vx, vy)
    xdata = np.reshape(xy[0][:], 256**2)
    ydata= np.reshape(xy[1][:],256**2)
    ax=fig.add_subplot(1,3,i+1, projection='3d')
    ax.set_xlabel('col')
    ax.set_ylabel('lin')
    ax.set_zlabel('z')
    ax.scatter(xdata, ydata, zdata, c=cor[i], s=0.5)
    ax.set_xlim(0,255)
    ax.set_ylim(0,255)
    ax.set_zlim(0,m)
    

