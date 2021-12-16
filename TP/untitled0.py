# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:23:20 2021

@author: silam
"""




## Coordenadas Terreno dos PFs (ERTS89-PT-TM06) [m]:

#PF1000
X1=-90158.653
Y1=-100565.532
Z1= 93.799

#PF2000
X2=-89794.832
Y2=-100478.941
Z2= 89.069

#PF3000
X3=-89788.239
Y3=-101679.518
Z3= 81.974

#PF4000
X4=-89514.366
Y4=-101489.343
Z4= 123.038

#PF6 - Nome original PF101-2012
X5=-90019.770
Y5=-101048.356
Z5=93.223

#PF7 - Nome original PF12-2010
X6=-89696.832
Y6=-101008.761
Z6= 95.277

## Informações sobre a obtenção da foto

# A imagem foi obtida por uma câmara
# digital DMC

#Características da câmara DMC #[m]
c=120; #mm
pixel=(12*10^-6);
s1=7680*pixel;
s2=13824*pixel;
x0=0;
y0=0;

## Coordenadas foto dos PFs [m]:

#PF1000
x1=-1.026;#*(10^-3);
y1=68.670;#*(10^-3);

#PF2000
x2=44.310;#*(10^-3);
y2=65.634;#*(10^-3);

#PF3000
x3=2.526;#*(10^-3);
y3=-70.734;#*(10^-3);

#PF4000
x4=41.166;#*(10^-3);
y4=-61.722;#*(10^-3);

#PF5000
x5=-1.962;#*(10^-3);
y5=7.386;#*(10^-3);

#PF6000
x6=36.534;#*(10^-3);
y6=0.642;#*(10^-3);


## DETERMINAÇÃO DA ORIENTAÇÃO EXTERNA DE UMA FOTO POR INTERSECÇÃO INVERSA

#Valores iniciais aproximados dos parâmetros de orientação externa:
X0=-89710.75
Y0=-100990.128
Z0=1095.365
omega=0.1208*(pi/180)
fi=0.2895*(pi/180)
kapa=((pi/2)-15.7612*(pi/180))

#Vector com os valores iniciais dos parâmetros
P=[X0,Y0,Z0,omega,fi,kapa]

#Ajustamento pelo Método dos Mínimos Quadrados

#Número total de observações
n=6

#Parâmetros
u=6 #desconhecidos

#Número mínimo de observações
n0=3

#Número de graus de liberdade
df=n-n0

#Variância a priori
sigma0_2 = 1

#Vectores das coordenadas X,Y e Z dos PFs
X=[X1,X2,X3,X4,X5,X6]
Y=[Y1,Y2,Y3,Y4,Y5,Y6]
Z=[Z1,Z2,Z3,Z4,Z5,Z6]

#Vector coordenadas foto dos PFs
x=[x1,x2,x3,x4,x5,x6]
y=[y1,y2,y3,y4,y5,y6]

#delta=[1;1;1;1;1;1]

delta_c=[1,1,1]
delta_a=[1,1,1]

aux_x=[]
aux_y=[]

A=[]
l=[]

s=1;

while ((norm(delta_c,inf) > 0.1) | (norm(delta_a,inf) > 0.0001)):
#Matrizes de rotação
    #Rotação Omega
    Rw=([1,0,0]
        [0,cos(P(4)),-sin(P(4))]
        [0,sin(P(4)),cos(P(4))])
    
    #Rotação Fi
    Rf=([cos(P(5)),0,sin(P(5))]
        [0,1,0]
        [-sin(P(5)),0,cos(P(5))])
    
    #Rotação Kapa
    Rk=([cos(P(6)),-sin(P(6)),0]
        [sin(P(6)),cos(P(6)),0]
        [0,0,1])
    
    #Matriz de rotação (omega fi kapa)
    R=Rw*Rf*Rk;
    
    
    for i in range (n0):  
    # Elementos das equações de colinearidade
        Nx= R(1,1)*(X(i)-P(1))+R(2,1)*(Y(i)-P(2))+R(3,1)*(Z(i)-P(3))
        Ny= R(1,2)*(X(i)-P(1))+R(2,2)*(Y(i)-P(2))+R(3,2)*(Z(i)-P(3))
        D= R(1,3)*(X(i)-P(1))+R(2,3)*(Y(i)-P(2))+R(3,3)*(Z(i)-P(3))
    
    #Equações de colinearidade
        x_ecol=x0-c*(Nx/D)
        y_ecol=y0-c*(Ny/D)
        aux_x.append([aux_x,x_ecol])
        aux_y.append([aux_y,y_ecol])
    
    #Expressões das derivadas parciais que entram nas equações linearizadas das
    #equações de colinearidade
    
    dpx_dpX0 = -(c/D^2)*(R(1,3)*Nx-R(1,1)*D);
    dpy_dpX0 = -(c/D^2)*(R(1,3)*Ny-R(1,2)*D);
    
    dpx_dpY0 = -(c/D^2)*(R(2,3)*Nx-R(2,1)*D);
    dpy_dpY0 = -(c/D^2)*(R(2,3)*Ny-R(2,2)*D);
    
    dpx_dpZ0 = -(c/D^2)*(R(3,3)*Nx-R(3,1)*D);
    dpy_dpZ0 = -(c/D^2)*(R(3,3)*Ny-R(3,2)*D);
    
    dpx_dpomega = -(c/D)*( ( (Y(i)-P(2))*R(3,3)-(Z(i)-P(3))*R(2,3) )*(Nx/D)-(Y(i)-P(2))*R(3,1)+(Z(i)-P(3))*R(2,1) );
    dpy_dpomega = -(c/D)*( ( (Y(i)-P(2))*R(3,3)-(Z(i)-P(3))*R(2,3) )*(Ny/D)-(Y(i)-P(2))*R(3,2)+(Z(i)-P(3))*R(2,2) );
    
    dpx_dpfi = (c/D)*( ( Nx*cos(P(6))-Ny*sin(P(6)) )*(Nx/D)+D*cos(P(6)) );
    dpy_dpfi = (c/D)*( ( Nx*cos(P(6))-Ny*sin(P(6)) )*(Ny/D)-D*sin(P(6)) );
    
    dpx_dpkapa = -(c/D)*Ny;
    dpy_dpkapa =  (c/D)*Nx;
    
    #Matriz de configuração (coeficientes dos parâmetros)
    
    A_Aux=[dpx_dpX0 dpx_dpY0 dpx_dpZ0 dpx_dpomega dpx_dpfi dpx_dpkapa;
           dpy_dpX0 dpy_dpY0 dpy_dpZ0 dpy_dpomega dpy_dpfi dpy_dpkapa];
    
    A=[A;A_Aux]

    #Equações linearizadas das equações de colinearidade
    
    # vx =dx0+ (dpx/dpc)*dc (dpx/dpf)*df + (dpx/dpw)*dw + (dpx/dpk)*dk + ...
    #     (dpx/dpX0)*dX0 + (dpx/dpY0)*dY0 + (dpx/dpZ0)*dZ0 + (x-xm);
    #
    # vy =dy0+ (dpy/dpc)*dc (dpy/dpf)*df + (dpy/dpw)*dw + (dpy/dpk)*dk + ...
    #     (dpy/dpX0)*dX0 + (dpy/dpY0)*dY0 + (dpy/dpZ0)*dZ0 + (y-ym);
    
  
        l_aux=[aux_x(i)-x(i);aux_y(i)-y(i)];
        l=[l;l_aux]
        #l=-l;
  
 end  
    delta = inv(A'*A)*A'*l
    
    delta_c=[delta(1);delta(2);delta(3)];
    delta_a=[delta(4);delta(5);delta(6)];
    
    P = P + delta

    A=[];
    l=[];
    aux_x=[];
    aux_y=[];
    
    s=s+1
end

#v_a = A * delta + l;

#obs=[x1;y1;x6;y6;x8;y8;x10;y10];

#obs_a= obs + v_a;

## Verificação dos resultados
