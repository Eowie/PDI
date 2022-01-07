# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 22:16:40 2021

@author: silam
"""

import numpy as np
import math

R3= np.array([[0.998360417,-0.057240522, 0],
      [0.057240522, 0.998360417, 0],
      [0,0,1]])


R3x= np.array([[-0.990238,0.139386, 0],
      [0.139386, 0.990238, 0],
      [0,0,1]])

R2= np.array([[0.634275267, 0, -0.773107],
      [0, 1, 0],
      [   0.773107, 0,0.634275267]])

P = np.array([[1,0,0],
              [0,-1,0],
              [0,0,1]])

dr=np.array ([[-1238.865],
              [1809.476],
              [-85.500]])


R=R3x@R2@P@dr

x=4889267.627
y=-690147.710
z=4023075.788

dx = -230.994
dy = 102.591
dz = 25.199
alfa=1.00000195
thx=(-0.633/3600)*math.pi/180
thy=(0.239/3600)*math.pi/180
thz=(-0.9/3600)*math.pi/180

x1=dx+alfa*x+alfa*thz*y-alfa*thy*z
y1=dy-alfa*thz*x+alfa*y+alfa*thx*z
z1=dz+alfa*thy*x-alfa*thx*y+alfa*z

e=0.00672267
a=6378137

fi0=(1/1-e)*z1/math.sqrt(x1**2+y1**2)
N0=a/math.sqrt(1-e*math.sin(fi0))

fi1=(z1+e*N0*math.sin(fi0))/math.sqrt(x1**2+y1**2)
N1=a/math.sqrt(1-e*math.sin(fi1))


fi2=(z1+e*N1*math.sin(fi1))/math.sqrt(x1**2+y1**2)
N2=a/math.sqrt(1-e*math.sin(fi2))


h=((math.sqrt(x1**2+y1**2))/math.cos(fi2))
h1=h-N2