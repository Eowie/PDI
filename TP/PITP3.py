import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from imageio import imread

plt.close('all')
#definir variaveis consoante o pc a usar para facilitar mudanca de pc
laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'
#changing directory to where the image is located
os.chdir(laptop)

#ex1

serie= np.load('soma3freq.npy')
# ima=imread('Marilyn.tif')
# serie= ima[50]-np.average(ima)
    #serie 'e a linha 50 da imagem)


 
#plot da serie por defeito cria um vector ate ao tamanho da serie
#cria um vector 1 ate 1 divido em 10000 partes - 10000 abcissas entre 0-1
#
xx = np.arange(0 , 1, 1/len(serie))
plt.plot(serie, 'r')
plt.xlabel('t', fontsize=8); plt.ylabel('f(t)', fontsize=8)
plt.tick_params(labelsize=8); plt.grid()

plt.figure()
plt.plot(xx, serie, 'r')

dft= np.fft.fft(serie)
xx1 = np.arange(-len(serie)/2, len(serie)/2, 1)
# espectro = np.abs(dft)
espectromi = np.abs(np.fft.fftshift(np.abs(dft)))
# plt.figure(); plt.plot(xx1, espectro)
plt.figure(); plt.plot(xx1, espectromi)

indices = np.where(dft!=0)
dft1=np.copy(dft)
dft1[10]=0; dft1[100]=0; 
dft1[9990]=0; dft1[9900]=0


ift1=np.fft.ifft(dft1)
plt.figure(); plt.plot(xx, ift1)

dft1=np.copy(dft)
dft1[1]=0; dft1[100]=0; 
dft1[9999]=0; dft1[9900]=0
ift1=np.fft.ifft(dft1)
plt.plot(xx, ift1)



# plt.ylabel('f1(t)', fontsize=8)
# plt.xlabel('t', fontsize=8)
# plt.tick_params(labelsize=8)





#exercicio 2

#

f =  np.array([[97, 150],
               [123, 27]])

plt.figure(); plt.imshow(f, 'gray', vmin=0, vmax=255)

M=f.shape[0]
N=f.shape[1]

#preencher a zeros uma matriz com a mesma dimensao da imagem new
#imashape e' um vector de 2 por 2 podemos usar em vez de 2
#ima.shape[0] e ima.shape[1] para a dimensoes do vector

F=np.zeros(f.shape)
for u in range (2):
    for v in range (2):
        F[u,v]= f[0,0]*np.e**(complex(0,-2*np.pi*((u*0)/M+(v*0)/N)))+ \
                f[1,0]*np.e**(complex(0,-2*np.pi*((u*0)/M+(v*1)/N)))+ \
                f[0,1]*np.e**(complex(0,-2*np.pi*((u*1)/M+(v*0)/N)))+ \
                f[1,1]*np.e**(complex(0,-2*np.pi*((u*1)/M+(v*1)/N)))


# F[u,v]= f[x,y]*np.e**(complex(0,-2*np.pi*((u*x)/M)+((v*y)/N)))=
#percorre x e y
# F[u,v]= f[0,0]*np.e**(complex(0,-2*np.pi*((u*0)/M)+((v*0)/N)))+ \
#         f[1,0]*np.e**(complex(0,-2*np.pi*((u*0)/M)+((v*1)/N)))+ \
#         f[0,1]*np.e**(complex(0,-2*np.pi*((u*1)/M)+((v*0)/N)))+ \
#         f[1,1]*np.e**(complex(0,-2*np.pi*((u*1)/M)+((v*1)/N)))
#representacao nr complexo z=complex(2,3)

#tentar comprovar se a transformada inversa funciona - trazer na proxima aula
