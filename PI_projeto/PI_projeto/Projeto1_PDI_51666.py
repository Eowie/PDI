# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:23:36 2020

@author: Miguel
"""
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage
plt.close('all');

#1.1
#Leitura da Imagem
Img=imread('noisyland.tif')
dim= Img.shape
lin=dim[0]
col=dim[1]

#Filtragem Espacial

#1.2 e 1.3
#Correção do ruído horizontal
Filtro_1= np.ones((7,30)) / (7*30) #Filtro que contem o ruido horizontal dentro dele
PB1=ndimage.median_filter(Img, footprint=Filtro_1, mode='constant', cval=0.0) #Aplicação do filtro passa-baixa ao ruído 
PA1= Img.astype(float) - PB1.astype(float) #Cálculo do respetivo filtro passa-alta

Filtro_2=np.ones((1,30)) / (1*30) #Filtro que está contido dentro do ruído horizontal
PB2=ndimage.median_filter(Img, footprint=Filtro_2, mode='constant', cval=0.0) #Aplicação do filtro passa-baixa ao ruído
PA2= Img.astype(float) - PB2.astype(float) #Cálculo do respetivo filtro passa-alta

Imagem_corrigida_linhas= PB1.astype(float) + PA2 #Correção do ruído horizontal



#Correção do Ruído vertical (tendo já sido corrigidas o ruído horizontal)
Filtro_3= np.ones((30,6)) / (30*6) #Filtro que contem o ruído vertical dentro dele
PB3=ndimage.median_filter(Imagem_corrigida_linhas, footprint=Filtro_3, mode='constant', cval=0.0)
PA3= Imagem_corrigida_linhas.astype(float) - PB3.astype(float)

Filtro_4=np.ones((30,1)) / (1*30) #Filtro que está contido dentro do ruído vertical
PB4=ndimage.median_filter(Imagem_corrigida_linhas, footprint=Filtro_4, mode='constant', cval=0.0) #Aplicação do filtro passa-baixa ao ruído
PA4= Imagem_corrigida_linhas.astype(float) - PB4.astype(float)#Cálculo do respetivo filtro passa-alta

Imagem_filtrada= PB3.astype(float) + PA4 #Correção do ruído vertical



#1.4
plt.figure(figsize=(20,15));
plt.subplot(221);plt.imshow(Img,'gray',vmin=0,vmax=255)
plt.title('Imagem Original'); plt.axis('off')
plt.subplot(222);plt.imshow(PB3,'gray', vmin=0,vmax=255)
plt.title('Passa-Baixa'); plt.axis('off')
plt.subplot(223);plt.imshow(PA4,'gray', vmin=0,vmax=255)
plt.title('Passa-Alta'); plt.axis('off')
plt.subplot(224);plt.imshow(Imagem_filtrada,'gray', vmin=0,vmax=255)
plt.title('Imagem Filtrada'); plt.axis('off')




# Filtragen de Fourier

#1.2
ddft=np.fft.fft2(Img) #Transformada de Fourier à imagem 
espectro_c_log=np.log10(np.abs(np.fft.fftshift(ddft))) #Espectro centrado

#Máscara 1
mask1=espectro_c_log>=3 

#Máscara 2 
#Composta por quadrados centrados nos picos de frequência identificados no espectro centrado
mask2=np.zeros((lin,col))
q=5 #Dimenão do quadrado para os picos verticais e horizontais ao centro
p=10 #Dimenão do quadrado para os picos diagonais ao centro

#Elaboração dos quadrados centrados nas coordenadas dos picos
mask2[89-q:89+q,311-q:311+q]=1
mask2[189-q:189+q,211-q:211+q]=1
mask2[289-q:289+q,311-q:311+q]=1
mask2[189-q:189+q,411-q:411+q]=1
mask2[88-p:88+p,410-p:410+p]=1
mask2[88-p:88+p,210-p:210+p]=1
mask2[288-p:288+p,210-p:210+p]=1
mask2[288-p:288+p,410-p:410+p]=1

#Oposto da junção das duas máscaras
mask=np.logical_not(np.logical_and(mask1,mask2))

#1.3
#Imagem filtrada através da inversa de Fourier da multiplicação da máscara pela transformada de Fourier da imagem inicial
Final= np.real(np.fft.ifft2(np.fft.fftshift(mask)*ddft))


#1.4
plt.figure(figsize=(20,15))
plt.subplot(221);plt.imshow(Img, 'gray')
plt.title('Imagem inicial'); plt.axis('off')
plt.subplot(222); plt.imshow(espectro_c_log, 'gray')
plt.title('Espectro centrado')
plt.subplot(223); plt.imshow(mask, 'gray')
plt.title('Máscara')
plt.subplot(224); plt.imshow(Final, 'gray', vmin=0,vmax=255)
plt.title('Imagem Filtrada'); plt.axis('off')



