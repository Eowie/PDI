# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:22:22 2021

@author: Eow
"""

import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import local_maxima

plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

#%%

#ex1

F=imread('ducks2.tif').astype(float)
T=imread('duck.tif').astype(float)

#e' necessario acrescentar espaco a zero a' imagem para ter a certeza q 
#o template corre a imagem toda

linf,colf=F.shape
lint,colt=T.shape
ypad=[lint//2,lint//2]
xpad=[colt//2,colt//2]

F1=np.pad(F,[ypad,xpad], mode='constant')
tmed=np.mean(T)
ccn=np.zeros(F.shape)

#f - media f
for i in range (0,linf):
    for j in range (0,colf):
        fuv=F1[i:i+lint,j:j+colt]
        mediafuv=np.mean(fuv)
        num=np.sum((fuv-mediafuv)*(T-tmed))
        den=np.sqrt(np.sum((fuv-mediafuv)**2)*np.sum((T-tmed)**2) )       
        ccn[i,j]=num/den
        
        

#%%
#ex2

maxccn=np.max(ccn)
# cheat=0.5

# yx_coords = np.column_stack(np.where(ccn >= cheat))

maximos=(local_maxima(ccn)*ccn)>0.5
r,c = np.where(maximos)
for i in range(len(r)):
    F[ r[i]-lint//2:r[i]+lint//2, c[i]-colt//2]=255
    F[ r[i]-lint//2:r[i]+lint//2, c[i]+colt//2]=255
    F[ r[i]-lint//2 ,c[i]-colt//2:c[i]+colt//2]=255
    F[ r[i]+lint//2 ,c[i]-colt//2:c[i]+colt//2]=255
   
    



#%%
plt.figure()
plt.subplot(231); plt.imshow(F,'gray'); plt.title('Inicial'); plt.axis('off')
plt.subplot(232); plt.imshow(T); plt.title('Template'); plt.axis('off')
plt.subplot(233); plt.imshow(F1); plt.title('Pad'); plt.axis('off')
plt.subplot(234); plt.imshow(ccn); plt.title('CCN'); plt.axis('off')
# plt.subplot(235); plt.imshow(F); plt.title('Outline'); plt.axis('off')
