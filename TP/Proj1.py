# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:10:59 2021

@author: Eow
"""

from imageio import imread
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

f=imread('noisyland.tif').astype(float)



filtro_pb = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]],
                  dtype='float')/9

#aplicar um filtro de passa baixa e aplicar de seguida um filtro passa alta sobre o resultado
f_pb = ndimage.convolve(f,filtro_pb,mode='constant', cval=0)

filtro_pa = np.array([[-1,-1,-1],
                      [-1, 8,-1],
                      [-1,-1,-1]])/9

f_pa = ndimage.convolve(f, filtro_pa, mode='constant', cval=0)

#soma devera ser igual 'a imagem original
somaf=f_pb+f_pa                    
                        

#criar um filtro com tamanho de bandas que contenham o ruido
filtroruido = np.ones((5,30))/(5*30)
pb1 = ndimage.convolve(f, filtroruido, mode='constant', cval=0)
pa1 = f-pb1

#criar um filtro com tamanho de bandas mais pequeno que o ruido
filtroruido2 = np.ones((1,30))/(1*30)

pb2 = ndimage.convolve(f, filtroruido2, mode='constant', cval=0)
pa2 = f-pb2

final = pb1+pa2
#Correção do Ruído vertical (tendo já sido corrigidas o ruído horizontal)

filtroruido3 = np.ones((30,7))/(7*30)
#Filtro que contem o ruído vertical dentro dele
pb3=ndimage.convolve(final,filtroruido3, mode='constant', cval=0)
pa3=final-pb3

filtroruido4=np.ones((30,1))/(1*30)
#Filtro que está contido dentro do ruído vertical
pb4=ndimage.convolve(final, filtroruido4, mode='constant', cval=0.0) 
#Aplicação do filtro passa-baixa ao ruído
pa4= final-pb4
#Cálculo do respetivo filtro passa-alta

ff= pb3+pa4  #Correção do ruído vertical

pb5= ndimage.convolve(ff,filtroruido, mode='constant', cval=0)
pa5= ff-pb5

pb6= ndimage.convolve(ff, filtroruido2, mode='constant', cval=0)
pa6= ff-pb6

fff=pb5+pa6

pb7=ndimage.convolve(fff, filtroruido3, mode='constant', cval=0)
pa7=fff-pb7

pb8=ndimage.convolve(fff, filtroruido4, mode='constant', cval=0)
pa8=fff-pb8

ffff=pb7+pa8



#%%



# noisedfft=np.fft.fft2(f)
# noiseespectro=abs(noisedfft)
# noiseespectrolog=np.log10(abs(noisedfft))
# noisemi=np.log10(abs(np.fft.fftshift(noisedfft)))
# noiseifft=abs(np.fft.ifft2(noisedfft))


# lin=f.shape[0]
# col=f.shape[1]

# #define duas mascaras para a imagem, uma onde os valores de noisemi sao superiores a 3
# mask1= noisemi>=3
# #segunda mascara 'e uma imagem de zeros com o tamanho de noise
# mask2= np.zeros(f.shape)
# # espessura das bandas de ruido
# q=5
# #definir na mascara 2 quais as linhas e colunas que serao iguais a 1
# mask2[q:int(lin/2), 0:q] =1; 
# mask2[int(lin/2):lin-q, 0:q]=1;
# mask2[q:int(lin/2), col-q+1:col] =1; 
# mask2[int(lin/2):lin-q, col-q+1:col]=1;

# #criar uma terceira mascara que 'e uma ifft centrada da segunda mascara
# mask3 = np.fft.ifftshift(mask2)
# #cria uma mascara 4 que 'e a disjuncao entre a juncao das mascaras 1 e 3
# mask4 = np.logical_not(np.logical_and(mask1,mask3))
# #cria uma mascara x que 'e a disjuncao entre a juncao das mascaras 1 e 2
# maskx = np.logical_not(np.logical_and(mask1,mask2)) 
# #a filtragem 'e feita com a inversa da mascara x conv com a dfft original
# noisefilt=np.abs(np.fft.ifft2(maskx*noisedfft))

# #plot das figuras
# plt.figure()                                                                    
# plt.subplot(121); plt.imshow(f, 'gray')
# plt.subplot(122); plt.imshow(noisemi, 'gray')
# # plt.subplot(253); plt.imshow(mask4, 'gray')
# # plt.subplot(254); plt.imshow(noisefilt, 'gray')

#%%


ddft=np.fft.fft2(f) #Transformada de Fourier à imagem 
espectro_c_log=np.log10(np.abs(np.fft.fftshift(ddft))) #Espectro centrado

#Máscara 1
mask1=espectro_c_log>=3

#Máscara 2 
#Composta por quadrados centrados nos picos de frequência identificados no espectro centrado
mask2=np.zeros((f.shape[0],f.shape[1]))
q=5 #Dimenão do quadrado para os picos verticais e horizontais ao centro
p=3 #Dimenão do quadrado para os picos diagonais ao centro

#Elaboração dos quadrados centrados nas coordenadas dos picos
mask2[188-q:188+q,310-q:310+q]=1
mask2[87-q:87+q,309-q:309+q]=1
mask2[89-q:89+q,311-q:311+q]=1
mask2[187-p:187+p,409-p:409+p]=1
mask2[189-q:189+q,411-q:411+q]=1
mask2[187-q:187+q,209-q:209+q]=1
mask2[189-q:189+q,211-q:211+q]=1
mask2[88-p:88+p,210-p:210+p]=1
mask2[90-p:90+p,212-p:212+p]=1



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
Final1= np.real(np.fft.ifft2(np.fft.fftshift(mask)*ddft))


#ver o efeito de cada um dos filtros
# plt.figure()
# plt.subplot(221); plt.imshow(f, 'gray'); plt.title('original')
# plt.subplot(222); plt.imshow(pb1, 'gray'); plt.title('Passa Baixa')
# plt.subplot(243); plt.imshow(pa1, 'gray', vmin=0, vmax=255); plt.title('Passa Alta')



# plt.figure()
# plt.subplot(221); plt.imshow(f, 'gray', vmin=0,vmax=255); plt.title('original')
# plt.subplot(222); plt.imshow(final, 'gray', vmin=0,vmax=255); plt.title('final'); plt.axis('off')
# plt.subplot(223); plt.imshow(ff, 'gray', vmin=0,vmax=255); plt.title('final'); plt.axis('off')
# plt.subplot(224); plt.imshow(ffff, 'gray', vmin=0,vmax=255); plt.title('final'); plt.axis('off')
# plt.subplot(245); plt.imshow(u, 'gray', vmin=0,vmax=255); plt.title('final'); plt.axis('off')

plt.figure(); 
plt.subplot(221); plt.imshow(mask1, 'gray')
plt.subplot(222); plt.imshow(mask2, 'gray')
plt.subplot(223); plt.imshow(mask, 'gray')
plt.subplot(224); plt.imshow(Final1, 'gray')

