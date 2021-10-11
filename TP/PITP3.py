import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from imageio import imread

plt.close('all')
#definir variaveis consoante o pc a usar para facilitar mudanca de pc
laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'
#changing directory to where the image is located
os.chdir(laptop)


#ex1
serie= np.load('soma3freq.npy')


 
#plot da serie por defeito cria um vector ate ao tamanho da serie
#xx = cria um vector 1 ate 1 divido em 10000 partes - 10000 abcissas entre 0-1

xx = np.arange(0 , 1, 1/len(serie))

# plot: criar um grafico, argumento 'e a serie, com representacao a vermelho 'r'
plt.plot(serie, 'r')

#label no eixo x = tempo, no eixo y funcao do tempo, tamanho da fonte
plt.xlabel('t', fontsize=8); plt.ylabel('f(t)', fontsize=8)
# ?
plt.tick_params(labelsize=8); plt.grid()

#criar uma figura com o plot, usando xx como o eixo, argumento serie, em vermelho
plt.figure()
plt.plot(xx, serie, 'r')

#criar dft como transformada de fourier directa da serie
dft= np.fft.fft(serie)

#criar um eixo
xx1 = np.arange(-len(serie)/2, len(serie)/2, 1)
#criar o espectro absoluto
espectro = np.abs(dft)
#criar o espectro espelhado
espectromi = np.abs(np.fft.fftshift(np.abs(dft)))

#plot das figuras
# plt.figure(); plt.plot(xx1, espectro)
plt.figure(); plt.plot(xx1, espectromi)

#procurar os picos do espectro
#a funcao indices nao funciona pois o calculo do python nao iguala os valores
#a zero mas a valores infinitesimais
# indices = np.where(dft!=0)
#podemos encontrar os picos com a tab plots e abrir a funcao dft

#criar uma copia da derivada de fourier da serie
dft1=np.copy(dft)
#a serie tem 6 picos 1,10,100, 9900, 9990, 9999
#igualamos 4 dos picos (2 espelhados) a zero
dft1[10]=0; dft1[100]=0; 
dft1[9990]=0; dft1[9900]=0

ift1=np.fft.ifft(dft1)
plt.figure(); plt.plot(xx, ift1)


dft1=np.copy(dft)
#igualamos outros picos a zero
dft1[1]=0; dft1[100]=0; 
dft1[9999]=0; dft1[9900]=0
ift1=np.fft.ifft(dft1)
plt.plot(xx, ift1)


#finalmente igualamos os ultimos picos a zero
dft1=np.copy(dft)
dft1[1]=0; dft1[10]=0; 
dft1[9999]=0; dft1[9990]=0
ift1=np.fft.ifft(dft1)
plt.plot(xx, ift1)

#no grafico obtemos as tres ondas sinusoidais que combinadas geram a onda
#da funcao

#labels para o grafico de representacao
plt.ylabel('f1(t)', fontsize=8)
plt.xlabel('t', fontsize=8)
plt.tick_params(labelsize=8)


#ex 1.4
ima=imread('Marilyn.tif')
#escolhemos como serie a linha 50 da imagem)
#ima[50] e retiramos a media para a serie estar em 0 no eixo
serie1= ima[50]-np.average(ima)

plt.plot(serie1, 'r')
dft1= np.fft.fft(serie1)
xx2 = np.arange(-len(serie1)/2, len(serie1)/2, 1)
espectromi1 = np.abs(np.fft.fftshift(np.abs(dft1)))

plt.figure(); plt.plot(xx2, espectromi1)


#exercicio 2
#criar a imagem f
f =  np.array([[97, 150],
               [123, 27]])

#visualizar a imagem f em tons de cinzento
plt.figure(); plt.imshow(f, 'gray', vmin=0, vmax=255)


#Definir M e N como o tamanho em pixels da imagem, em colunas e linhas
M=f.shape[0]
N=f.shape[1]

#preencher a zeros uma matriz com a mesma dimensao da imagem new
#imashape e' um vector de 2 por 2 podemos usar em vez de 2
#ima.shape[0] e ima.shape[1] para a dimensoes do vector

F=np.zeros(f.shape)

for u in range (2):
    for v in range (2):
                F[u,v]= f[0,0]*np.e**(complex(0,-2*np.pi*((u*0)/M+(v*0)/N)))+ \
                        f[1,0]*np.e**(complex(0,-2*np.pi*((u*1)/M+(v*0)/N)))+ \
                        f[0,1]*np.e**(complex(0,-2*np.pi*((u*0)/M+(v*1)/N)))+ \
                        f[1,1]*np.e**(complex(0,-2*np.pi*((u*1)/M+(v*1)/N)))

#representacao nr complexo z=complex(2,3)

#tentar comprovar se a transformada inversa funciona - trazer na proxima aula

F1=np.zeros(f.shape)

for x in range (2):
    for y in range (2):
                F1[x,y]= 1/(M*N)*  (F[0,0]*np.e**(complex(0,2*np.pi*((x*0)/M+(y*0)/N)))+ \
                                    F[1,0]*np.e**(complex(0,2*np.pi*((x*1)/M+(y*0)/N)))+ \
                                    F[0,1]*np.e**(complex(0,2*np.pi*((x*0)/M+(y*1)/N)))+ \
                                    F[1,1]*np.e**(complex(0,2*np.pi*((x*1)/M+(y*1)/N))))


#ex 2.3
mardfft=np.fft.fft2(ima)
marespectro=abs(mardfft)

marespectrolog=np.log10(abs(np.fft.fft2(ima)))
marmi=np.log10(abs(np.fft.fftshift(mardfft)))
marifft=abs(np.fft.ifft2(mardfft))

#experimentar em casa fazer a ifft do espectro
# marinv= np.fft.ifft2(np.log10(abs(np.fft.fftshift(mardfft))))


plt.figure()
plt.subplot(161); plt.imshow(ima, 'gray'); plt.axis('off'), plt.title('Original');
plt.subplot(162); plt.imshow(marespectro, 'gray'); plt.axis('off'),plt.title('DFFT')
plt.subplot(163); plt.imshow(marespectrolog, 'gray'); plt.axis('off'),plt.title('Espectro')
plt.subplot(164); plt.imshow(marmi, 'gray'); plt.axis('off'),plt.title('Espelhado')
plt.subplot(165); plt.imshow(marifft, 'gray'); plt.axis('off'),plt.title('IFFT')
# plt.subplot(166); plt.imshow(np.uint8(np.abs(marinv)), 'gray'); plt.axis('off'),plt.title('IFFT do Espectro')



#ex 3.1

circles = imread('circles.tif')

#a nr de linhas
#b nr de colunas
#f valor da frequencia
#definir o filtro plano
#criar uma mascara toda preenchida com valores a 1 com a linhas e b colunas
#coloca no primeiro pixel um valor 0
#coloca no ultimo pixel em coluna, em linha etc tudo a zeros
#com D vai procurar todos os pixels onde o valor central seja inferior 



circlesdfft=np.fft.fft2(circles)
circlesespectro=np.log10(abs(circlesdfft))
circlesmi=np.fft.fftshift(circlesespectro)


def filtro_lp(a,b,freq):
    mask= np.ones(circles.shape)
    mask[0,0]=0
    mask[0,-1]=0
    mask[-1,0]=0
    mask[-1,-1]=0
    D= scipy.ndimage.distance_transform_edt(mask)
    p= D<=freq
    return p


plt.figure()
plt.subplot(251); plt.imshow(circles, 'gray')
plt.subplot(252); plt.imshow(circlesespectro, 'gray')
plt.subplot(253); plt.imshow(circlesmi,'gray')

frequencia=50
filtrox=filtro_lp(circles.shape[0],circles.shape[1],frequencia)

plt.subplot(254); plt.imshow(filtrox)
plt.subplot(255); plt.imshow(filtrox*circlesespectro, 'gray')

circles_filtrada= np.fft.ifft2(filtrox*circlesdfft)
plt.subplot(256); plt.imshow(np.abs(circles_filtrada),'gray')
#vmin vmax para homogenezar a imagemm na frequencia no 0



#mudanca dos quadrantes
#permite saber quais as frequencias q estao a ser retiradas do espectro centrado
# plt.imshow(np.fft.fftshift(circlesfiltro))

#para fazer isto no espectro nao centrado tem de se multiplicar o filtro original pelo espectro original
#fazer estes para a imagem marylin tambem


def gauss_lp_2d(lin,col,mu,sigma):
    x, y = np.meshgrid(np.linspace(-col/2, col/2, col),\
                        np.linspace(-lin/2, lin/2, lin))
    g= np.exp(-((x-mu)**2+(y-mu)**2)/(2*sigma**2))
    return g
    

mu= 0
sigma= frequencia
filtroy= np.fft.fftshift(gauss_lp_2d(circles.shape[0], circles.shape[1], mu, sigma ))
circles_filtradoy=np.fft.ifft2(filtroy*circlesdfft)

plt.subplot(257); plt.imshow(np.abs(circles_filtradoy),'gray')
    



# # ex 3.2

noise = imread('Stripping_Noise.tif')
noisedfft=np.fft.fft2(noise)
noiseespectro=abs(noisedfft)
noiseespectrolog=np.log10(abs(noisedfft))
noisemi=np.log10(abs(np.fft.fftshift(noisedfft)))
noiseifft=abs(np.fft.ifft2(noisedfft))


lin=noise.shape[0]
col=noise.shape[1]
mask1= noisemi>=3
mask2= np.zeros(noise.shape)
q=5
mask2[q:int(lin/2), 0:q] =1; mask2[int(lin/2):lin-q, 0:q]=1
mask2[q:int(lin/2), col-q+1:col] =1; mask2[int(lin/2):lin-q, col-q+1:col]=1
mask3 = np.fft.ifftshift(mask2)
mask4 = np.logical_not(np.logical_and(mask1,mask3))
maskx = np.logical_not(np.logical_and(mask1,mask2)) 
test=np.abs(np.fft.ifft2(maskx*noisedfft))


plt.figure()                                                                    
plt.subplot(251); plt.imshow(noise, 'gray')
plt.subplot(252); plt.imshow(noisemi, 'gray')
plt.subplot(253); plt.imshow(mask4, 'gray')
plt.subplot(254); plt.imshow(test, 'gray')