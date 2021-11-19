# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:10:59 2021

@author: Silvia
"""
#%%
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# =============================================================================
# # Fechar todos os plots
# =============================================================================
plt.close('all')

# =============================================================================
# #Ler a imagem
# =============================================================================
f=imread('noisyland.tif').astype(float)


#%%
# =============================================================================
# #Parte 1 - Filtragem no Domínio Espacial
# =============================================================================

#Construir um filtro para correcção do ruído horizontal
#Filtro1 - contem a banda de ruído horizontal dentro dele
Filtro1 = np.ones((7,30))/(7*30)
pb1 = ndimage.convolve(f, Filtro1, mode='constant', cval=0)
pa1 = f-pb1

#Filtro2 - contido dentro do ruído 
Filtro2 = np.ones((1,30))/(1*30)
pb2 = ndimage.convolve(f, Filtro2, mode='constant', cval=0)
pa2 = f-pb2

#Imagem Corrigida de Ruído Horizontal
fhor = pb1+pa2


#Construir um filtro para correcção do ruído vertical
#Filtro3 - contem a banda de ruído vertical dentro dele
Filtro3 = np.ones((30,5))/(5*30)
pb3=ndimage.convolve(fhor,Filtro3, mode='constant', cval=0)
pa3=fhor-pb3

#Filtro4 - contido dentro do ruído
Filtro4=np.ones((30,1))/(1*30)
pb4=ndimage.convolve(fhor, Filtro4, mode='constant', cval=0.0) 
pa4=fhor-pb4

#Imagem Corrigida de Ruído
fcorr = pb3+pa4

#Depois da primeira iteração de correcção, ainda existe ruído na imagem
#O ruído existente é maioritariamente vertical, por isso fazemos mais uma
#iteração de correcção na vertical

pb5=ndimage.convolve(fcorr,Filtro3, mode='constant', cval=0)
pa5=fcorr-pb5
pb6=ndimage.convolve(fcorr,Filtro4,mode='constant',cval=0)
pa6=fcorr-pb6

fcorr2=pb5+pa6

#A imagem está corrigida de ruído, no entanto podemos ainda tentar
#efetuar um unsharp suave para melhorar a definição da imagem

unsh=ndimage.median_filter(fcorr2,footprint=Filtro4)
k=0.3
u=fcorr2+k*(fcorr2-unsh)


#%%
# =============================================================================
# #Parte 2 - Filtragem no Dominio de Fourier
# =============================================================================

#Aplicar a Transformada de Fourier à imagem
ddft=np.fft.fft2(f)

#Espectro centrado
espectro_c_log=np.log10(np.abs(np.fft.fftshift(ddft)))

#Mask1 - Máscara para valores do espectro superiores a 3
Mask1=espectro_c_log>=3

#Mask2 - Máscara de quadrados centrados nos picos de frequencia 
#Estes picos foram identificados por análise da imagem do espectro centrado

Mask2=np.zeros((f.shape[0],f.shape[1]))
#Dimensão dos quadrados para os picos verticais e horizontais
a=5
#Dimensão dos quadrados para os picos diagonais
b=7

#Quadrados centrados nos picos identificados
Mask2[88-a:88+a,310-a:310+a]=1
Mask2[186-a:186+a,410-a:410+a]=1
Mask2[288-a:288+a,310-a:310+a]=1
Mask2[188-a:188+a,210-a:210+a]=1

Mask2[88-b:88+b,410-b:410+b]=1
Mask2[288-b:288+b,410-b:410+b]=1
Mask2[288-b:288+b,210-b:210+b]=1
Mask2[88-b:88+b,210-b:210+b]=1

#Máscara final
mask=np.logical_not(np.logical_and(Mask1,Mask2))

#Filtragem da Imagem
final= np.real(np.fft.ifft2(np.fft.fftshift(mask)*ddft))


#%%
# =============================================================================
# #Plots dos Resultados
# =============================================================================

fig1=plt.figure(figsize=(30,30))
fig1.suptitle('Filtro Espacial parte 1')
plt.subplot(333); plt.imshow(f, 'gray', vmin=0,vmax=255); plt.title('Original'); plt.axis('off')
plt.subplot(334); plt.imshow(pb1, 'gray', vmin=0,vmax=255); plt.title('Passa Baixa Horizontal'); plt.axis('off')
plt.subplot(335); plt.imshow(pa2, 'gray', vmin=0,vmax=255); plt.title('Passa Alta Horizontal'); plt.axis('off')
plt.subplot(336); plt.imshow(fhor, 'gray', vmin=0,vmax=255); plt.title('Correção Ruído Horizontal'); plt.axis('off')
plt.subplot(337); plt.imshow(pb3, 'gray', vmin=0,vmax=255); plt.title('Passa Baixa Vertical'); plt.axis('off')
plt.subplot(338); plt.imshow(pa4, 'gray', vmin=0,vmax=255); plt.title('Passa Alta Vertical'); plt.axis('off')
plt.subplot(339); plt.imshow(fcorr, 'gray', vmin=0,vmax=255); plt.title('Correção Ruído Vertical'); plt.axis('off')

fig2=plt.figure(figsize=(30,30))
fig2.suptitle('Filtro Espacial parte 2')
plt.subplot(332); plt.imshow(f, 'gray', vmin=0,vmax=255); plt.title('Original'); plt.axis('off')
plt.subplot(333); plt.imshow(fcorr, 'gray', vmin=0,vmax=255); plt.title('Correção Ruído Vertical')
plt.subplot(334); plt.imshow(pb5, 'gray', vmin=0,vmax=255); plt.title('Passa Baixa Vertical 2'); plt.axis('off')
plt.subplot(335); plt.imshow(pa6, 'gray', vmin=0,vmax=255); plt.title('Passa Alta Vertical 2'); plt.axis('off')
plt.subplot(336); plt.imshow(fcorr2, 'gray', vmin=0,vmax=255); plt.title('Imagem Corrigida'); plt.axis('off')
plt.subplot(337); plt.imshow(unsh, 'gray', vmin=0,vmax=255); plt.title('Filtro Unsharp'); plt.axis('off')
plt.subplot(339); plt.imshow(u, 'gray', vmin=0,vmax=255); plt.title('Imagem Final'); plt.axis('off')


fig3=plt.figure(figsize=(30,30))
fig3.suptitle('Filtro Fourier')
plt.subplot(333); plt.imshow(f, 'gray'); plt.title('Original'); plt.axis('off')
plt.subplot(334); plt.imshow(espectro_c_log, 'gray'); plt.title('Espectro Centrado'); plt.axis('off')
plt.subplot(335); plt.imshow(Mask1, 'gray'); plt.title('Máscara 1'); plt.axis('off')
plt.subplot(336); plt.imshow(Mask2, 'gray'); plt.title('Máscara 2'); plt.axis('off')
plt.subplot(337); plt.imshow(mask, 'gray'); plt.title('Máscara Final'); plt.axis('off')
plt.subplot(338); plt.imshow(final, 'gray'); plt.title('Imagem Final'); plt.axis('off')

fig4=plt.figure(figsize=(30,30))
fig4.suptitle('Comparação Resultados')
plt.subplot(131); plt.imshow(f, 'gray'); plt.title('Original'); plt.axis('off')
plt.subplot(132); plt.imshow(u, 'gray', vmin=0,vmax=255); plt.title('Imagem Final Espacial'); plt.axis('off')
plt.subplot(133); plt.imshow(final, 'gray'); plt.title('Imagem Final Fourier'); plt.axis('off')
