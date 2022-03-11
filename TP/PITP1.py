"""
Spyder Editor

This is a temporary script file.
"""

# importing from image library commands to read and write images
from imageio import imread, imwrite
#why?
import os
# importing mat lab plotting library, assigning plt to avoid calling the long file name
import matplotlib.pyplot as plt

#closes any windows left over from previous runs of the program
plt.close('all')

#changing directory to where the image is located
# os.chdir('C:/Users/silam/OneDrive/Desktop')
#assign variable ima to a read of the picture we want to use
ima=imread('lena.tif')

#assign variables to characteristics of the image
dim = ima.shape
tipo = ima.dtype
npix = ima.size

#plotting the figure in original color and in each shade of RGB
#plotting the figure with a certain size
plt.figure (figsize=(14,5))
#plotting original figure, adding a title
plt.subplot(141); plt.imshow(ima); plt.title('RGB')
#plotting each other shade, representing all (:) y and x axis, in gray color
#and each channel on the color spectrum [0,1,2], removing the axis and adding title
plt.subplot(142); plt.imshow(ima[:,:,0], 'gray'), plt.axis('off'); plt.title('R')
plt.subplot(143); plt.imshow(ima[:,:,1], 'gray'), plt.axis('off'); plt.title('G')
plt.subplot(144); plt.imshow(ima[:,:,2], 'gray'), plt.axis('off'); plt.title('B')

# code below creates and saves a new tif image file for each of the RGB channels
# imwrite('R.tif',ima[:,:,0])
# imwrite('G.tif',ima[:,:,1])
# imwrite('B.tif',ima[:,:,2])

#gives you the color of the pixel on the location below (remember y,x,color)
i=27
j=67
val = ima[i, j, 0]

#defines two points on the picture(cse and cid) in opposite ends of the rectangle we want to cut
linha_cse = 120
coluna_cse = 120
linha_cid= 140
coluna_cid = 180
#crops image from one point to the other in all channels
cropIma = ima[linha_cse:linha_cid, coluna_cse:coluna_cid,:];
#plots the cropped image
plt.figure(); plt.imshow(cropIma);
#add title an axis to the crop
plt.title('crop'); plt.axis('off')

#importing numpy - usually all imports are done at the beginning of the program
import numpy as np
#creating a new copy of the original image to avoid overwritting
img1= np.copy(ima)
#replace the area we cropped in the last exercise with a box of a certain color (10 in this case)
img1[linha_cse:linha_cid, coluna_cse:coluna_cid] = 149
#plot the new figure
plt.figure (); plt.imshow(img1, 'gray')
plt.title('substituicao'); plt.axis ('off')


# plotting image profiles
import numpy as np
import scipy.ndimage

# build a line with a number of points between two coordinate points
# (lin, column) : (y0,x0) and (y1,x1)

#define the coordinate points we want to use
y0, x0 = 100,50
y1, x1 = 200,200
#how many points we want our line to have
num = 100
#creates a line with starting and ending points defined below
y,x = np.linspace(y0,y1,num), np.linspace(x0,x1,num)

#extracts the value of each pixel along the line
d=np.vstack((y,x))
perfil = scipy.ndimage.map_coordinates(ima[:,:,0], d)

#plots figure with certain size, creates a subplot which shows the image in R
plt.figure(figsize=(15, 3))
plt.subplot(121); plt.imshow(ima[:,:,0],'gray')
plt.title('inicial'); plt.axis('off')
#plots the line we defined on the picture,"ro-" r=red, o=circles, - line
plt.plot([x0,x1],[y0,y1],'ro-')
#plots the value of each pixel defined before, "ro--" red circles with non continuous line
plt.subplot(122); plt.plot(perfil,'ro--')
plt.title('Perfil')