# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:42:05 2021

@author: Eow
"""

import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, erosion, dilation, opening, \
    closing, local_minima, local_maxima, watershed, binary_dilation    
from skimage.filters import threshold_otsu
from scipy import ndimage
from copy import deepcopy

    
plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

#%%

#ex1

rgr=imread('rgrow.tif').astype(float)

lin,col=rgr.shape
ee=np.ones((3,3))
Er = erosion(rgr,ee)
Dl = dilation(rgr,ee)
Grad = dilation(rgr,ee)-erosion(rgr,ee)


Lm = local_minima(rgr,ee)
LmGrad = local_minima(Grad,ee)

mk=np.ones(rgr.shape)
markers, n = ndimage.label(LmGrad)
Ws = watershed (Grad, markers,mask=mk)
Ws1=deepcopy(Ws)

#%%
#criar a moldura da imagem para retirar as bacias q tocam a imagem
#ex 1.2
M=np.zeros(rgr.shape)
M[:,0]=1
M[:,col-1]=1
M[0,:]=1
M[lin-1,:]=1

T=Ws*M
nz=np.nonzero(T)

val=np.unique(T[nz])

for i in range (len(val)):
    Ws[Ws==val[i]]=0


#%%
#ex 1.3
seed=[8,15]
B = Ws==Ws[seed[0], seed[1]]
B=B*rgr


#%%


##solucao 1
# media0=np.mean(B[np.nonzero(B)])
# inters=(binary_dilation(B>0)&~(B>0)*Ws
# a=np.nonzero(inters)
# b=np.unique(Ws[a])
# regiao=B
# for i in range (len(b)):
#     bac1=(Ws==b[i])*rgr
#     media1=np.mean(bac1[np.nonzero(bac1)])
#     if np.abs(media1-media0)<=10:
#         regiao=regiao|(bac1>0)
# #ideia 'e continuar este ciclo ate a bacia ja nao crescer mais



##solucao 2

# m=[-2,-1]
# while m[-1]!=m[-2]:
#     A=B*rgr.astype(float)
#     m.append(np.mean(A[np.nonzero(A)]))
#     D=np.logical_and(binary_dilation(B,ee), np.logical_not(B))*Ws
#     val=np.unique(D[np.nonzero(D)])
#     for j in range (len(val)):
#         V=(D==val[j])*rgr.astype(float)
#         if abs(np.mean(V[np.nonzero(V)])-np.mean(m[2:len(m)]))<10:
#             B1=np.logical_or(B, Ws==val[j])
            
# del m[0:2]; del m[-1]
    


#%%

#2.1
C=imread('lsat01.tif').astype(float)
C=C[:,:,0]
gauss_denoised=ndimage.filters.gaussian_filter(C,1)

#2.2
sx=np.array([[-1,0,1],
             [-2,0,2],
             [-1,0,1]], dtype='float')
sy=np.array([[1,2,1],
             [0,0,0],
             [-1,-2,-1]],dtype='float')

conv000=ndimage.convolve(gauss_denoised.astype(float),sx, mode='constant', cval=0.0)
conv090=ndimage.convolve(gauss_denoised.astype(float),sy, mode='constant', cval=0.0)

#2.3
z=np.logical_and(conv090==0, conv000==0)*1
D0 = np.arctan2(conv090, conv000)
D1 = D0*180/np.pi
Theta = D1+180

#2.4
Theta0= np.logical_or(np.logical_and(Theta>=0, Theta<22.5), np.logical_and(Theta>=337.5, Theta<360))
Theta45= np.logical_and(Theta>=22.5, Theta <67.5)
Theta90= np.logical_and(Theta>=67.5, Theta <112.5)
Theta135= np.logical_and(Theta>=112.5, Theta <157.5)
Theta180= np.logical_and(Theta>=157.5, Theta <202.5)
Theta225= np.logical_and(Theta>=202.5, Theta <247.5)
Theta270= np.logical_and(Theta>=247.5, Theta <292.5)
Theta315= np.logical_and(Theta>=292.5, Theta <337.5)

Tt1 = np.logical_or(Theta45, Theta225)*45
Tt2 = np.logical_or(Theta90, Theta270)*90
Tt3 = np.logical_or(Theta135, Theta315)*135
Tt4 = np.logical_or(Theta180, Theta0)*180

ThetaQuantized=(Tt1+Tt2+Tt3+Tt4).astype(float)
ThetaQuantized[z==1]=0

#2.5
S=(abs(conv000)+abs(conv090))
lin,col=C.shape
G1=np.zeros((lin,col))
for k1 in range (1,lin-1):
    for k2 in range (1, col-1):
        if (k1==0 | k1==lin-1 | k2==0 | k2==col-1):
            G1[k1,k2]=0
        else:
            if ThetaQuantized[k1,k2]==180:
                if np.logical_and(S[k1,k2-1]<S[k1,k2], S[k1,k2+1]<S[k1,k2]):
                    G1[k1,k2]=S[k1,k2]
            if ThetaQuantized[k1,k2]==45:
                if np.logical_and(S[k1+1,k2-1]<S[k1,k2], S[k1-1,k2+1]<S[k1,k2]):
                    G1[k1,k2]=S[k1,k2]
            if ThetaQuantized[k1,k2]==90:
                if np.logical_and(S[k1-1,k2]<S[k1,k2], S[k1+1,k2]<S[k1,k2]):
                    G1[k1,k2]=S[k1,k2]
            if ThetaQuantized[k1,k2]==135:
                if np.logical_and(S[k1-1,k2-1]<S[k1,k2], S[k1+1,k2+1]<S[k1,k2]):
                    G1[k1,k2]=S[k1,k2]
                    
# # #2.6
CGrad = dilation(C,ee)-erosion(C,ee)
LMGrad = local_maxima(CGrad,ee)
t1=20
t2=80
tsup=t2*np.max(G1)/255
tinf=t1*np.max(G1)/255

PxFortes=G1>=tsup
PxFracos=np.logical_and(G1>=tinf,G1<tsup)
Edges= deepcopy(PxFortes)
_,n = ndimage.label(Edges)
a=-1;
while a!=n:
    _,n=ndimage.label(Edges)
    se = rectangle (3,3)
    Edges = np.logical_or(Edges, np.logical_and(binary_dilation(Edges, se),PxFracos))
    _,a=ndimage.label(Edges)
    

#%%
#plots


# plt.figure()
# plt.subplot(241); plt.imshow(rgr, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(242); plt.imshow(Grad, 'gray'); plt.title('Gradiente'); plt.axis('off')
# plt.subplot(243); plt.imshow(LmGrad,'gray'); plt.title('Local Minima Grad'); plt.axis('off')
# plt.subplot(244); plt.imshow(Ws1, 'gray'); plt.title('Ws'); plt.axis('off')
# plt.subplot(245); plt.imshow(Ws, 'gray'); plt.title('Ws que nao toca a moldura'); plt.axis('off')
# plt.subplot(246); plt.imshow(B, 'gray'); plt.title('Bacia 8,15'); plt.axis('off')
# plt.subplot(248); plt.imshow(regiao, 'gray'); plt.title(''); plt.axis('off')
# plt.subplot(247); plt.imshow(inters, 'gray'); plt.title(''); plt.axis('off')

# plt.figure()
# plt.subplot(241); plt.imshow(A, 'gray'); plt.title(''); plt.axis('off')
# plt.subplot(242); plt.imshow(V, 'gray'); plt.title(''); plt.axis('off')
# plt.subplot(243); plt.imshow(D, 'gray'); plt.title(''); plt.axis('off')
# plt.subplot(244); plt.imshow(B1, 'gray'); plt.title(''); plt.axis('off')

#ex 2
# plt.figure()
# plt.subplot(241); plt.imshow(C, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(242); plt.imshow(gauss_denoised, 'gray'); plt.title('Gauss'); plt.axis('off')
# plt.subplot(243); plt.imshow(conv000,'gray'); plt.title('Sobelx'); plt.axis('off')
# plt.subplot(244); plt.imshow(conv090, 'gray'); plt.title('Sobely'); plt.axis('off')
# plt.subplot(246); plt.imshow(ThetaQuantized); plt.title('ThetaQuant'); plt.axis('off')
# plt.subplot(245); plt.imshow(Theta); plt.title('Theta'); plt.axis('off')
# plt.subplot(247); plt.imshow(G1); plt.title('G1'); plt.axis('off')
# plt.subplot(248); plt.imshow(Edges, 'gray'); plt.title(''); plt.axis('off')

plt.figure()
plt.subplot(241); plt.imshow(C); plt.title('Inicial'); plt.axis('off')
plt.subplot(242); plt.imshow(G1); plt.title('G1'); plt.axis('off')
plt.subplot(243); plt.imshow(LMGrad); plt.title('Grad'); plt.axis('off')
plt.subplot(244); plt.imshow(PxFracos); plt.title('Fracos'); plt.axis('off')
plt.subplot(245); plt.imshow(PxFortes); plt.title('Fortes'); plt.axis('off')
plt.subplot(246); plt.imshow(Edges); plt.title('Edges'); plt.axis('off')
# plt.subplot(247); plt.imshow(G1); plt.title('G1'); plt.axis('off')
# plt.subplot(248); plt.imshow(Edges, 'gray'); plt.title(''); plt.axis('off')