import os
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, diamond, binary_erosion, binary_opening, skeletonize, reconstruction     
from scipy import ndimage

# para eliminar as linhas soltas temos de fazer, progressivamente, as extremidades 

    
plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

#%%
#Exercicio 1

def hnt(f, tipo):
    out = np.zeros(f.shape).astype(bool)
    if tipo == 'extremidades':
      
   #construcao do elememto estruturante e criar um elemento que me veja os cantos
        b1 = np.array([[ 0,0 ,0 ],
                      [ 0, 1,0 ],
                      [ 0, 1, 0]])
        
        #Complementar do b1
        b2 = np.array([[ 1, 1, 1],
                      [ 1, 0, 1],
                      [ 0, 0, 0]])  #os 0 desta linha na posicao 1 e 3 é indiferente pq continua a ser uma extremidade
         #construcao do elememto estruturante e criar um elemento que me veja os cantos
        b3 = np.array([[ 0,0 ,0 ],
                      [ 0, 1,0 ],
                      [ 1, 0, 0]])
        
        #Complementar do b1
        b4 = np.array([[ 1, 1, 1],
                      [ 1, 0, 1],
                      [ 0, 1, 1]])
        
        
        for i in range(4):                   
            e1 = binary_erosion(f, np.rot90(b1,i))
            e2 = binary_erosion(~f, np.rot90(b2,i))
            e3 = binary_erosion(f, np.rot90(b3,i))
            e4 = binary_erosion(~f, np.rot90(b4,i))
            out = out | (e1 & e2) 
            out = out | (e3 & e4)
    

    elif tipo == 'isolados':
        
         #construcao do elememto estruturante e criar um elemento que me veja os cantos
        b1 = np.array([[ 0,0 ,0 ],
                      [ 0, 1,0 ],
                      [ 0, 0, 0]])
        
        #Complementar do b1
        b2 = np.array([[ 1, 1, 1],
                      [ 1, 0, 1],
                      [ 1, 1, 1]]) 
        e1 = binary_erosion(f,b1)
        e2 = binary_erosion(~f,b2)
        out = out | (e1 & e2)
                
    else:
        return 0
    
    return out



f = imread('spur.tif')> 0


#isolados = hnt(f, 'isolados')

f1 = np.copy(f)
extremidades=1
while (np.sum(extremidades))>0:
   extremidades = hnt(f1, 'extremidades')
   # extremidades = hnt(f1, 'isolados')
   final = f1 & ~extremidades 
   f1 = np.copy(final)
    
iso = hnt(f1, 'isolados')

isolados = f1 &~iso
    
    

plt.figure()



img=reconstruction(isolados,f)
img1=reconstruction(f1,f)>0
img2=img1&~iso

plt.subplot(231); plt.imshow(f, 'gray'); plt.title('Inicial'); plt.axis('off')
plt.subplot(232); plt.imshow(final, 'gray'); plt.title('Final sem extremidades'); plt.axis('off')
plt.subplot(233); plt.imshow(isolados, 'gray'); plt.title('Final sem os pontos isolados'); plt.axis('off')
plt.subplot(234); plt.imshow(img, 'gray'); plt.title('Reconstrucao Isolados'); plt.axis('off')
plt.subplot(235); plt.imshow(img1, 'gray'); plt.title('Reconstrucao Extremidades'); plt.axis('off')
plt.subplot(236); plt.imshow(img2, 'gray'); plt.title('Reconstrucao Extremidades and Not Iso'); plt.axis('off')
#%%
#Exercicio 2

F4=imread('binary_file.tif')>0

#n nr de objectos binarios que existem na imagem
#nrs de euler dos objetos da imagem
L, n = ndimage.measurements.label(F4)
eu= np.arange(n)
E=np.zeros(F4.shape)
for i in range(1,n+1):
   bw= L==i
   _, n = ndimage.measurements.label(np.logical_not(bw))
   eu[i-1]=1-(n-1)
   if eu[i-1]==0:
       eu[i-1]=2
   E=E+bw*eu[i-1]

plt.figure()
plt.subplot(231); plt.imshow(F4, 'gray'); plt.title('Inicial'); plt.axis('off')
plt.subplot(232); plt.imshow(E); plt.title('E'); plt.axis('off')


#%%

# Exercicio3

F1=imread('cxhull1.tif')>0

Df=ndimage.distance_transform_edt(F1==0)
L,n=ndimage.measurements.label(F1)
Skiz=np.zeros(F1.shape).astype(bool)
m=disk(1)

#calcular n com funcao label
for i in range (1,n+1):
    Di= ndimage.distance_transform_edt(~(L==i))
    Iz= binary_erosion(Df==Di,m)
    Skiz= Skiz | Iz


t2=skeletonize(~(Skiz))
t1=~t2&~F1

plt.figure()
plt.subplot(231); plt.imshow(F1); plt.title('Inicial'); plt.axis('off')
plt.subplot(232); plt.imshow(Df); plt.title('Df'); plt.axis('off')
plt.subplot(233); plt.imshow(Skiz); plt.title('.'); plt.axis('off')
plt.subplot(234); plt.imshow(t2); plt.title('.'); plt.axis('off')
plt.subplot(235); plt.imshow(t1); plt.title('.'); plt.axis('off')


























