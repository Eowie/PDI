# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 18:40:16 2021

@author: silam
"""
#importar modulos necessarios
from imageio import imread
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

#fechar plots antigos antes de abrir novos
plt.close('all')

#definir variaveis consoante o pc a usar para facilitar mudanca de pc
laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'
#changing directory to where the image is located
os.chdir(pc)


#abrir imagem de 8 bits as type float - permite que sejam feitas operacoes
#matematicas com valores negativos ou acima de 255 sem ajustar os valores

nome = 'Marilyn'
ext = '.tif'
Img = imread('marilyn.tif').astype(float)

#fazer o plot da imagem
plt.figure()
plt.imshow(Img, 'gray'); plt.axis('off');

#criar um filtro para processamento - usando array e definindo a matriz ou
#com np.ones criar uma matriz de 1s com o numero de linhas e colunas definido

filtro = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]],
                  dtype='float')/9

#em cima 'e o mesmo que
#filtro= np.ones((3,3))/9

filtro1= np.ones((3,3))/9
filtro2= np.ones((9,9))/81
            

#define a convolucao da media manualmente
conv0= np.zeros(Img.shape)
for i in range (1, Img.shape[0]-1):
    for j in range (1,Img.shape[1]-1):
        conv0[i,j]= filtro[0, 0]*Img[i-1, j-1] + filtro[0, 1]*Img[i-1, j] + filtro[0, 2]*Img[i-1, j+1]  + \
                    filtro[1, 0]*Img[  i, j-1] + filtro[1, 1]*Img[  i, j] + filtro[1, 2]*Img[  i, j+1]  + \
                    filtro[2, 0]*Img[i+1, j-1] + filtro[2, 1]*Img[i+1, j] + filtro[2, 2]*Img[i+1, j+1]
                    
                    
#usa a funcao ndimage.convolve para fazer a convolucao da imagem, segundo os filtros
#definidos anteriormente                                  
conv1 = ndimage.convolve(Img, filtro1, mode='constant', cval=0)
conv2 = ndimage.convolve(Img, filtro2, mode='constant', cval=0)

#plot de todas as imagens obtidas
plt.figure(figsize=(14,5));
plt.subplot(141); plt.imshow(  Img, 'gray'); plt.axis('off'), plt.title(nome);
plt.subplot(142); plt.imshow(conv0, 'gray'); plt.axis('off'),plt.title('Conv0')
plt.subplot(143); plt.imshow(conv1, 'gray'); plt.axis('off'),plt.title('Conv1')
plt.subplot(144); plt.imshow(conv2, 'gray'); plt.axis('off'),plt.title('Conv2')

#abrir duas imagens com ruido as type float
gauss= imread('noisy_gauss.tif').astype(float)
sp = imread('noisy_sp.tif').astype(float)

#realizar funcoes de filtro passa baixa de media sobre as duas imagens
mean_denoisedga = ndimage.convolve(gauss,filtro,mode='constant', cval=0)
mean_denoisedsp = ndimage.convolve(sp,filtro,mode='constant', cval=0)

# gauss filter
gauss_denoisedga = ndimage.filters.gaussian_filter(gauss, 1)
gauss_denoisedsp = ndimage.filters.gaussian_filter(sp, 1)

#median filter
median_denoisedga = ndimage.filters.median_filter(gauss, 3)
median_denoisedsp = ndimage.filters.median_filter(sp, 3)

#plots the results of the filters above in a 2:4 matrix
plt.figure(figsize=(14,5));
plt.subplot(241); plt.imshow(gauss, 'gray'); plt.axis('off');plt.title('Noisy Gauss')
plt.subplot(242); plt.imshow(mean_denoisedga, 'gray'); plt.axis('off'),plt.title('Mean Denoised')
plt.subplot(243); plt.imshow(gauss_denoisedga, 'gray'); plt.axis('off'),plt.title('Gauss Denoised')
plt.subplot(244); plt.imshow(median_denoisedga, 'gray'); plt.axis('off'),plt.title('Median Denoised')
plt.subplot(245); plt.imshow(sp, 'gray'); plt.axis('off'); plt.title ('Noisy SP')
plt.subplot(246); plt.imshow(mean_denoisedsp, 'gray'); plt.axis('off'),plt.title('Mean Denoised')
plt.subplot(247); plt.imshow(gauss_denoisedsp, 'gray'); plt.axis('off'),plt.title('Gauss Denoised')
plt.subplot(248); plt.imshow(median_denoisedsp, 'gray'); plt.axis('off'),plt.title('Median Denoised')

#prewitt filter
#applies mask over x and then over y and combines the two
px = ndimage.prewitt(Img, axis=0, mode='constant')
py = ndimage.prewitt(Img, axis=1, mode='constant')
prw = np.abs(px)+np.abs(py)
prw= prw*255

# Sobel Filter
#applies mask over x and then over y and combines the two
sx = ndimage.sobel(Img, axis=0, mode='constant')
sy = ndimage.sobel(Img, axis=1, mode='constant')
sob= np.hypot(sx, sy)
sob=sob*255

#plot results above
plt.figure(figsize=(14,5))
plt.subplot(141); plt.imshow(Img,'gray'); plt.title(nome)
plt.subplot(142); plt.imshow(prw,'gray'); plt.title('Prewitt')
plt.subplot(143); plt.imshow(sob,'gray'); plt.title('Sobel')


#Unsharp gauss - ndimage.filters.gaussian_filter(image, sigma)
#unsharp is the difference between the image and the image modified by a passa baixa filter x k
#where k is a constant we can vary
Un1= ndimage.filters.gaussian_filter(Img, 1)
Un2= Img - Un1
k=0.2
k1=0.6
Un3 = Img+k*Un2
Un7 = Img+k1*Un2


#unsharp mean - as above but with a convolve filter instead of a gaussian filter
#remember we need to define the filter used in this case (using the old filter of 3x3 matrix of 1s)
Un4= ndimage.convolve(Img,filtro,mode='constant', cval=0)
Un5 = Img - Un4
k2=0.2
k3=0.6
Un6 = Img+k*Un5
Un8 = Img+k*Un5

plt.figure(figsize=(14,5))
plt.subplot(141); plt.imshow(Un3, 'gray'); plt.title('Unsharp Gauss k=0.2')
plt.subplot(142); plt.imshow(Un7, 'gray'); plt.title('Unsharp Gauss k=0.6')
plt.subplot(143); plt.imshow(Un6, 'gray'); plt.title('Unsharp Mean k=0.2')
plt.subplot(144); plt.imshow(Un8, 'gray'); plt.title('Unsharp Mean k=0.6')

#Gauss Laplace
# Para aplicar o filtro laplaciano do gaussiano temos de primeiro definir a formula LoG,
# o desvio padrao sigma e uma matriz quadrada com colunas usando meshgrid
# neste caso usamos x0 e y0 igual a 0

l=3
x, y = np.meshgrid( np.linspace(-l, l, 2*l+1), np.linspace(-l, l, 2*l+1))
sigma = 1
x0=0
y0=0

LoG = lambda x, y:-1/(np.pi*sigma**4)*(1-((x-x0)**2+(y-y0)**2)/(2*sigma**2))*np.e**(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))
filtro_log =LoG(x, y)

#depois de definirmos o filtro LoG podemos novamente fazer a convolucao com a imagem
#devido 'a propriedade associativa da convolucao

conv3 = ndimage.convolve(Img.astype(float), filtro_log, mode='constant', cval=0)

plt.figure(figsize=(14,5))
plt.subplot(141); plt.imshow(conv3, 'gray')


#exercicio 5
#ler uma nova imagem com ruido as type float
noise=imread('Stripping_Noise.tif').astype(float)

#aplicar um filtro de passa baixa e aplicar de seguida um filtro passa alta sobre o resultado
mean_noise = ndimage.convolve(noise,filtro,mode='constant', cval=0)


#passa baixa
#zfinal = 1/9 *z(i-1, j-1)+1/9*z(i-1,j).... etc
#
#Passa Alta
# Ima-PB = z(i,j)-PB
# z(i,j)-[z(i-1, j-1)+1/9*z(i-1,j).... etc]
# so o valor central tem valor positivo 8/9 z(i,j)

filtro_pa = np.array([[-1,-1,-1],
                      [-1, 8,-1],
                      [-1,-1,-1]])/9

conv4 = ndimage.convolve(noise, filtro_pa, mode='constant', cval=0)

#soma devera ser igual 'a imagem original
soma=mean_noise + conv4                        
                        
#plot das figuras
plt.figure()
plt.subplot(141); plt.imshow(noise, 'gray'); plt.title('original')
plt.subplot(142); plt.imshow(mean_noise, 'gray'); plt.title('Passa Baixa')
plt.subplot(143); plt.imshow(np.abs(conv4), vmin=np.min(conv4), vmax=np.max(conv4)); plt.title('Passa Alta')
plt.subplot(144); plt.imshow(soma, 'gray'); plt.title('Soma')
             


#criar um filtro com tamanho de bandas que contenham o ruido
filtroruido = np.ones((21,201))/(21*201)
pb1 = ndimage.convolve(noise, filtroruido, mode='constant', cval=0)
pa1 = noise-pb1

#criar um filtro com tamanho de bandas mais pequeno que o ruido
#se mudar-mos o tamanho dos filtros podemos tentar encontrar uma melhor solucao
filtroruido2 = np.ones((5,201))/(5*201)

pb2 = ndimage.convolve(noise, filtroruido2, mode='constant', cval=0)
pa2 = noise-pb2

#ver o efeito de cada um dos filtros
plt.figure()
plt.subplot(221); plt.imshow(noise, 'gray'); plt.title('original')
plt.subplot(222); plt.imshow(pb1, 'gray'); plt.title('Passa Baixa')
plt.subplot(243); plt.imshow(pa1, 'gray', vmin=0, vmax=255); plt.title('Passa Alta')


final = pb1+pa2

plt.subplot(121); plt.imshow(noise, 'gray'); plt.title('original')
plt.subplot(122); plt.imshow(final, 'gray'); plt.title('final'); plt.axis('off')

