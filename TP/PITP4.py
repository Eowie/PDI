import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from imageio import imread
import scipy.ndimage
from skimage.morphology import disk, rectangle, reconstruction, \
    binary_erosion, binary_dilation, binary_opening, binary_closing
    
plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

F0=imread('bintree.tif')
#F0<120 erode parte preta
#F0>120 erode parte branca
F1=F0<120

# h, r = np.histogram(F0, bins=265, range=(0,256))
# plt.figure(); plt.plot(h, 'k')
# plt.title('Histograma')
# se = rectangle (d, d)

p=3
se=np.ones((p,p))


#erosao

Er = binary_erosion(F1, se)

#dilatacao

Di = binary_dilation (F1, se)

#Abertura

Ab = binary_opening(F1, se)

#Fecho


Fe = binary_closing(F1, se)

plt.figure(figsize=(12,10))
plt.subplot(231); plt.imshow(F1, 'gray'); plt.axis('off'); plt.title('Inicial')                              
plt.subplot(232); plt.imshow(Er, 'gray'); plt.axis('off'); plt.title('Erosao')  
plt.subplot(233); plt.imshow(Di, 'gray'); plt.axis('off'); plt.title('Dilatacao')  
plt.subplot(235); plt.imshow(Ab, 'gray'); plt.axis('off'); plt.title('Abertura')  
plt.subplot(236); plt.imshow(Fe, 'gray'); plt.axis('off'); plt.title('Fecho')  


#exercicio 1. 2

Img2=imread('bin03.tif')>0

#exemplo com rectangulo
# se2=rectangle(31,31)

#exemplo com circulo
se2= disk(20)
Ab2= binary_erosion(Img2, se2)
circle = binary_dilation(Ab2, se2)
sal = Img2 & ~circle

plt.figure()
plt.subplot(231); plt.imshow(Img2, 'gray')
plt.subplot(232); plt.imshow(Ab2, 'gray')
plt.subplot(233); plt.imshow(circle, 'gray')
plt.subplot(236); plt.imshow(sal, 'gray')

# exercicio 2.1

Img3=imread('barras.tif')>0

se3= rectangle(23,3)
test = binary_erosion(Img3, se3)
test2 = binary_dilation(test, se3)

se4 = rectangle(3,23)

#opening: dilatacao da erosao
test3 = binary_opening(Img3, se4)
test4 = binary_dilation(test3, se4)


#fica com buracos:
# test3 = Img3 & ~test2
# test4 = binary_dilation(test3)
# test5 = binary_dilation(test4)
# test6 = binary_dilation(test5)
# test7 = binary_dilation(test6)
# test8 = binary_dilation(test7)
# test9 = Img3 & test6

plt.figure()
plt.subplot(231); plt.imshow(Img3, 'gray')
plt.subplot(232); plt.imshow(test, 'gray')
plt.subplot(233); plt.imshow(test2, 'gray')
plt.subplot(236); plt.imshow(test4, 'gray')


#exercicio 2.2
#encontrar cantos - definir matriz para os cantos rectos e depois
#rodar as matrizes 90graus de cada vez

b1 = np.array([[0,0,0],
               [0,1,1],
               [0,1,0]])


b2 = np.array([[ 1, 1, 0],
               [ 1, 0, 0],
               [ 0, 0, 0]])

Img3corner=np.zeros(Img3.shape).astype(bool)

for i in range(4):
    bn=np.rot90(b1, i)
    bm=np.rot90(b2, i)
    e1= binary_erosion(Img3, bn)
    e2= binary_erosion(~Img3, bm)
    Img3corner = Img3corner | (e1 & e2)
    

plt.figure()
plt.imshow(Img3corner)


se4= np.ones((1,10))
aresta1 = binary_closing(Img3corner, se4)

se5= np.ones((10,1))
aresta2 = binary_closing(Img3corner, se5)


plt.figure()
plt.imshow(aresta1|aresta2)



#exercicio 3

Img4=imread('bin04.tif')>0

#3.1

def reconstrucao_bin(mask, marker):
    serb = rectangle(3,3)
    ant = np.zeros(mask.shape)
    R= np.copy(marker)
    while np.array_equal(R, ant)== False:
        ant = R
        R = binary_dilation(R,serb)&mask
    return R


#3.2

mold=np.zeros(Img4.shape).astype(bool)
mold[:,0]=True
mold[:,-1]=True
mold[0,:]=True
mold[-1,:]=True


inters=Img4 & mold
obj= reconstrucao_bin(Img4,inters)



subt = Img4 & ~obj

plt.figure()
plt.subplot(131); plt.imshow(Img4, 'gray')
plt.subplot(132); plt.imshow(obj, 'gray')
plt.subplot(133); plt.imshow(subt, 'gray')


#3.3

circleee=disk(30)
circle3=binary_opening(subt,circleee)

saliencias_ruido= subt&~circle3

saliencias_final=reconstrucao_bin(saliencias_ruido, circle3)



plt.figure()
plt.subplot(141); plt.imshow(subt, 'gray')
plt.subplot(142); plt.imshow(circle3, 'gray')
plt.subplot(143); plt.imshow(saliencias_ruido, 'gray')
plt.subplot(144); plt.imshow(saliencias_final, 'gray')




    