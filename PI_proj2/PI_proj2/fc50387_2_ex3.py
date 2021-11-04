"""

@author: Guilherme Rodrigues Gomes Martinez
fc50387
"""

"""Projeto 2 ex3"""
from skimage.morphology import disk, rectangle, erosion, dilation, opening, closing, local_minima, local_maxima, watershed, binary_erosion, binary_dilation, binary_opening, binary_closing
from imageio import imread, imwrite
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
plt.close('all')
img = imread('ik02.tif')

img2 = (img[:,:,2] - img[:,:,0] - 180)

h_img2, r = np.histogram(img2, bins = 256, range=(0,256))
plt.figure();
plt.plot(h_img2, 'b');

#determinar o maximo do histograma
xmax = np.where(h_img2 == np.max(h_img2)); xmax = xmax[0][0]
ymax = h_img2[xmax]
x1 = xmax
y1 = ymax
x2 = 255
y2 = h_img2[x2]

#definiçao da reta
a1 = (y2 - y1)/(x2 - x1)
b1 = y1 - a1 * x1
a2 = -a1 ** -1
b2 = np.zeros((256)); x = np.copy(b2); y = np.copy(b2); d = np.copy(b2)
for i in range(xmax, x2):
    #coordenada y na origem
    b2[i] = h_img2[i] - a2 * i 
    x[i] = (b1 - b2[i])/(a2 - a1)
    y[i] = a2 * (b1 - b2[i])/(a2 - a1) + b2[i]
    d[i] = np.sqrt((x[i] - i) ** 2 + (y[i] - h_img2[i]) ** 2)

#Threshold
th = np.where(d == np.max(d)); th = th[0][0]

img_bin = np.zeros(img2.shape)

img_bin[img2 < th] = 0
img_bin[img2 >= th] = 1

img_final = np.copy(img)
for i in range(img.shape[2]):
    img_final[:,:,i] = img_final[:,:,i] * img_bin
    img_final[:,:,i] = erosion(img_final[:,:,i], rectangle(3,3))
    img_final[:,:,i] = dilation(img_final[:,:,i], rectangle(3,3))

plt.figure(figsize=(12,10)) 
a = plt.subplot(121); plt.imshow(img, 'gray');plt.axis('off');
b = plt.subplot(122); plt.imshow(img_final, 'gray');plt.axis('off');

a.title.set_text('Imagem original');plt.axis('off');
b.title.set_text('Imagem com as casas')
