"""

@author: Guilherme Rodrigues Gomes Martinez
fc50387
"""

"""Projeto 2 ex2"""
from skimage.morphology import disk, rectangle, skeletonize, reconstruction, erosion, dilation, opening, closing, local_minima, local_maxima, watershed, binary_erosion, binary_dilation, binary_opening, binary_closing
from imageio import imread, imwrite
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

img = imread('ik02.tif')


#Histogramas das 3 bandas

hr, r = np.histogram(img[:,:,0], bins=256, range=(0, 256))

hg, r = np.histogram(img[:,:,1], bins=256, range=(0, 256))

hb, r = np.histogram(img[:,:,2], bins=256, range=(0, 256))

img2 = img[:,:,0]/3+ img[:,:,1]/3+ img[:,:,2]/3

plt.figure()
plt.subplot(3,1,1)
plt.plot(hr, 'r')
plt.subplot(3,1,2)
plt.plot(hg, 'g')
plt.subplot(3,1,3)
plt.plot(hb, 'b')

#threshold da imagemoriginal
aux = 195

img2_bin = np.zeros(img2.shape)


img2_bin[img2 >= aux] = 1
img2_bin[img2 < aux] = 0

img2_3bandas = np.copy(img)

#Filtragem da imagem binária resultante do passo anterior
dil = binary_dilation(img2_bin, rectangle(5, 5)) #Dilatação
fecho = binary_erosion(dil,rectangle(5, 5)) #Erosão da dilatação que resulta no fecho da img



def reconstrucao_dual(mask,marker):
    a=1
    ee=rectangle(3,3)
    while a!=0:
        E=erosion(marker,ee)
        R=np.maximum(mask.astype(float),E.astype(float))
        a=np.count_nonzero(marker!=R)
        marker=deepcopy(R)
    return R


#Medicao da Distancia
d= ndimage.distance_transform_edt(fecho==1)

#Reconstrucao dual da distancia
d1=d+1
rd_d= reconstrucao_dual(d,d1)

#minimo regional
min_reg = rd_d - d

ee=disk(10)
#watershed para cálculo da linha do rio
marker,n= ndimage.label(min_reg)
W = watershed(fecho, marker, mask = np.ones(fecho.shape))
Iint = W- erosion(W, ee) #Linha do rio

#linha do rio binaria
Iint_bin=Iint.astype(bool)

#skeletonize da linha do rio para menor espessura
sk1=skeletonize(Iint_bin)

a=reconstruction(img2_bin, sk1)



#Voltar a transformar a imagem binaria numa imagem com 3 bandas
for i in range(img.shape[2]):
    img2_3bandas[:,:,i] = img[:,:,i] * a
    
img_final = np.copy(img)
x = 4
re = rectangle(x, x)

#Aplicar operaçoes morfologicas a cada uma das bandas da imagem
for i in range(img_final.shape[2]):
    img_final[:,:,i] = erosion(img2_3bandas[:,:,i], re)
    img_final[:,:,i] = dilation(img_final[:,:,i], re)
    img_final[:,:,i] = closing(img_final[:,:,i], re)
   

plt.figure(figsize=(12,10))
a = plt.subplot(121); plt.imshow(img, 'gray');plt.axis('off');
b = plt.subplot(122); plt.imshow(img_final, 'gray');plt.axis('off');
    
a.title.set_text('Imagem original')
b.title.set_text('Imagem com as estradas')