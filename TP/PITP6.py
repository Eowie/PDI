import os
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, erosion, dilation, opening, \
    closing, local_minima, local_maxima, watershed    
from scipy import ndimage
from copy import deepcopy

    
plt.close('all')

laptop='C:/Users/Eow/Desktop/Mestrado/PDI/TP'
pc='C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP'

os.chdir(laptop)

#%%
#ex 1
#erosao numerica valor min da vizinhanca no elemento estruturante
#dilatacao numerica valor max da vizinhanca no elemento estruturante
#abertura dilatacao da erosao
#fecho erosao da dilatacao

Img=imread('Marilyn.tif').astype(float)
ee=np.ones((3,3))
Er = erosion(Img,ee)
Dl = dilation(Img,ee)
Op = opening(Img,ee)
Cl = closing(Img, ee)
Grad = dilation(Img,ee)-erosion(Img,ee)
GradInt = Img - erosion(Img,ee)
GradExt = dilation(Img,ee)-Img



#%%

#ex 2
#suavizacao de uma imagem podemos utilizar a abertura e o fecho
#primeiro a abertura e depois o fecho ou a media entre a erosao e a dilatacao

Sua= closing(Op,ee)
TH = Img - Op
BH = Cl - Img
Realce = (Sua + (TH-BH)).clip(0,255)


#%%
#ex 3

# #Img - imagem Marilyn

Ik=imread('ik01.tif').astype(float); Ik=Ik[:,:,0]
Grains=imread('grains.tif').astype(float)


#definir uma imagem com as mesmas dimensoes que a imagem original
#onde os valores de todos os pixels sao iguais a zero exceto o pixel
#indicado como marcador no enunciado
#escolher um ponto com cor clara para ficar destacado na reconstrucao

mIk=np.zeros(Ik.shape)
mIk[64,90]=Ik[64,90]
mGrains=np.zeros(Grains.shape)
mGrains[35,68]=Grains[35,68]
mImg=np.zeros(Img.shape)
mImg[171,113]=Img[171,113]

def reconstrucao_gray(mask, marker):
    a=1
    ee=rectangle(3,3)
    while a!=0:
        D=dilation(marker,ee)
        R=np.minimum(mask.astype(float), D.astype(float))
        a=np.count_nonzero(marker!=R)
        marker = deepcopy(R)
    return R

rgImg=reconstrucao_gray(Img,mImg)
rgIk=reconstrucao_gray(Ik,mIk)
rgGrains=reconstrucao_gray(Grains,mGrains)


mIk1=np.ones(Ik.shape)*255
mIk1[68,111]=Ik[68,111]
mGrains1=np.ones(Grains.shape)*255
mGrains1[163,128]=Grains[163,128]
mImg1=np.ones(Img.shape)*255
mImg1[165,107]=Img[165,107]

def reconstrucao_dual(mask, marker):
    a=1
    ee=rectangle(3,3)
    while a!=0:
        E = erosion (marker,ee)
        R=np.maximum(mask.astype(float), E.astype(float))
        a= np.count_nonzero(marker!=R)
        marker= deepcopy(R)
    return R


rdImg=reconstrucao_dual(Img,mImg1)
rdIk=reconstrucao_dual(Ik,mIk1)
rdGrains=reconstrucao_dual(Grains, mGrains1)

minregGrains = reconstrucao_dual(Grains, Grains+1)
minimos=(minregGrains-Grains)>0

maxregGrains = reconstrucao_gray(Grains, Grains-1)
maximos=Grains-maxregGrains

minregIk= local_minima(Ik,ee)
maxregIk = local_maxima(Ik,ee)


#%%
# ex 4
Lm = local_minima(Img,ee)
LmGrad = local_minima(Grad,ee)

mk=np.ones(Img.shape)

markers, n = ndimage.label(Lm)
Ws = watershed (Img, markers,mask=mk)

markers1, n = ndimage.label(LmGrad)
Ws1 = watershed (Grad, markers1,mask=mk)

d = 1
ee = disk(d)
Lint_Img=Ws-erosion(Ws,ee)
Lext_Img=dilation(Ws,ee)-Ws
LextImg=(~(Lext_Img>0))*Img
LintImg=(~(Lint_Img>0))*Img


Lint_Grad=Ws1-erosion(Ws1,ee)
Lext_Grad=dilation(Ws1,ee)-Ws1
LintGrad=(~(Lint_Grad>0))*Img
LextGrad=(~(Lext_Grad>0))*Img


#%%

#plots
# plt.figure()
# plt.subplot(151); plt.imshow(Img, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(152); plt.imshow(Er, 'gray'); plt.title('Erosion'); plt.axis('off')
# plt.subplot(153); plt.imshow(Dl, 'gray'); plt.title('Dilation'); plt.axis('off')
# plt.subplot(154); plt.imshow(Op, 'gray'); plt.title('Opening'); plt.axis('off')
# plt.subplot(155); plt.imshow(Cl, 'gray'); plt.title('Closing'); plt.axis('off')

# plt.figure()
# plt.subplot(141); plt.imshow(Img, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(142); plt.imshow(Grad, 'gray'); plt.title('Gradiente'); plt.axis('off')
# plt.subplot(143); plt.imshow(GradInt, 'gray'); plt.title('Gradiente Interno'); plt.axis('off')
# plt.subplot(144); plt.imshow(GradExt, 'gray'); plt.title('Gradiente Externo'); plt.axis('off')

# plt.figure()
# plt.subplot(151); plt.imshow(Img, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(152); plt.imshow(Sua, 'gray'); plt.title('Suavizacao'); plt.axis('off')
# plt.subplot(153); plt.imshow(TH, 'gray'); plt.title('Top Hat'); plt.axis('off')
# plt.subplot(154); plt.imshow(BH, 'gray'); plt.title('Bottom Hat'); plt.axis('off')
# plt.subplot(155); plt.imshow(Realce, 'gray'); plt.title('Realce'); plt.axis('off')


# plt.figure()
# plt.subplot(331); plt.imshow(Img, 'gray'); plt.title('Inicial'); plt.axis('off')
# plt.subplot(334); plt.imshow(rgImg, 'gray'); plt.title('rg_Img'); plt.axis('off')
# plt.subplot(332); plt.imshow(Ik, 'gray'); plt.title('Ik'); plt.axis('off')
# plt.subplot(335); plt.imshow(rgIk, 'gray'); plt.title('rg_Ik'); plt.axis('off')
# plt.subplot(333); plt.imshow(Grains, 'gray'); plt.title('Grains'); plt.axis('off')
# plt.subplot(336); plt.imshow(rgGrains, 'gray'); plt.title('rg_grains'); plt.axis('off')
# plt.subplot(338); plt.imshow(rdIk, 'gray'); plt.title('rd_IK'); plt.axis('off')
# plt.subplot(339); plt.imshow(rdGrains, 'gray'); plt.title('rd_grains'); plt.axis('off')
# plt.subplot(337); plt.imshow(rdImg, 'gray'); plt.title('rdImg'); plt.axis('off')

# plt.figure()

# plt.subplot(151); plt.imshow(maxregIk, 'gray'); plt.title('Max'); plt.axis('off')
# plt.subplot(152); plt.imshow(minregIk, 'gray'); plt.title('Min'); plt.axis('off')
# plt.subplot(153); plt.imshow(maximos, 'gray'); plt.title('Max'); plt.axis('off')
# plt.subplot(154); plt.imshow(minimos, 'gray'); plt.title('Min'); plt.axis('off')



plt.figure()
plt.subplot(351); plt.imshow(Img, 'gray'); plt.title('Inicial'); plt.axis('off')
plt.subplot(352); plt.imshow(Lm, 'gray'); plt.title('Local Minima'); plt.axis('off')
plt.subplot(353); plt.imshow(LmGrad, 'gray'); plt.title('Local Minima Grad'); plt.axis('off')
plt.subplot(354); plt.imshow(Lint_Img, 'gray'); plt.title('Lint_Img'); plt.axis('off')
plt.subplot(355); plt.imshow(Lext_Img, 'gray'); plt.title('Lext_Img'); plt.axis('off')
plt.subplot(356); plt.imshow(Lint_Grad, 'gray'); plt.title('Lint_Grad'); plt.axis('off')
plt.subplot(357); plt.imshow(Lext_Grad, 'gray'); plt.title('Lext_Grad'); plt.axis('off')

plt.figure()
plt.subplot(351); plt.imshow(LintImg, 'gray'); plt.title('LintImg'); plt.axis('off')
plt.subplot(352); plt.imshow(LextImg, 'gray'); plt.title('LextImg'); plt.axis('off')
plt.subplot(353); plt.imshow(LintGrad, 'gray'); plt.title('LintGrad'); plt.axis('off')
plt.subplot(354); plt.imshow(LextGrad, 'gray'); plt.title('LextGrad'); plt.axis('off')
