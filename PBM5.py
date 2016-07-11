# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:52:30 2016

@author: Burak Akar
"""

import numpy
import pylab
import math
from scipy.integrate import odeint
from scipy.optimize import fmin

Na = 6.02214e+23 # Avogadro's constant in 1/mol
AuM = 98.0 # molar density in mol/L

#DATA
Time=numpy.asarray([0.0,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,7.0])#h
Grays=numpy.array([0.0,5.0,10.0,15.0,25.0,35.0])
Diameters=numpy.array([0.0,121.105,	102.325,	88.465,	72.5875,	57.27])#nm
Absorbances=numpy.matrix([[0.0,0.002333333,0.002666667,0.003733333,0.001633333,0.001733333,0.0025,0.001733333,0.002433333,0.003133333],[0.0,0.0062,0.0093,0.013433,0.0144,0.015567,0.018167,0.018633,0.02,0.0199],[0.0,0.033366667,0.042,0.054066667,0.053266667,0.053933333,0.054866667,0.055633333,0.055333333,0.056033333],[0.0,0.0809,0.0931333333,0.1076333333,0.1022666667,0.1009666667,0.0973333333,0.0985,0.0968,0.1031],[0.0,0.2208,0.231,0.230733333,0.213866667,0.205266667,0.210433333,0.2127,0.2061,0.212333333],[0.0,0.31,0.305633333,0.3042,0.299133333,0.2968,0.2987,0.2953,0.304066667,0.3033]])
ICs=numpy.array([0.003257421,0.039755701,0.036040704,0.018990198,0.075461916,0.069681971])#mM

bins = 50 
atoms_bin = 1228117.0    # atoms added per bin, depends on number of bins and max particle size
min_radius = 5.0e-10
nucleus=20          # atoms required for nucleation
Dmax = 138.375  #nm

#Initial conditions
num=3
N0= numpy.zeros(bins,dtype=numpy.float)
Abs=numpy.ndarray.flatten(numpy.asarray(Absorbances[num])) 
N0[0]=ICs[num]/1000   #M
kparams=numpy.array([4.0e-6 , 4.0e+13])     #k1,k2        

def growth(bin_num, A, params):
    k2=params[1]
    Vp = (bin_num * atoms_bin) / (1000.0 * Na * AuM) # minimum crysal volume in m^3
    radius = math.pow(3.0 * Vp / (4.0 * math.pi), 1.0/3.0) # radius in m
    if A>0.0:
        return k2 * A*min_radius / (radius)
    else:
        return 0.0

def df(N, t, params):                                   # ODE definition
    k1=params[0]
    A = N[0]
    dNdt = numpy.zeros(bins,dtype=numpy.float)
    dNdt[0] = -k1*A - growth(1, A, params)*N[1]
    dNdt[1] = k1*A/nucleus - growth(1, A, params)*N[1]/atoms_bin
    for i in range(2,bins-1):
        dNdt[0] = dNdt[0] - growth(i, A, params)*N[i]
        dNdt[i] = growth(i-1, A, params)*N[i-1]/atoms_bin - growth(i,A,params)*N[i]/atoms_bin
    dNdt[bins-1] = growth(bins-2,A,params)*N[bins-2]/atoms_bin
    return numpy.array(dNdt) 

def Atotal(C):                                                #Absorbance
    for j in range (0,tsteps):
        for i in range (1,bins):
            if C[j,i]<0:
                C[j,i]=0
            Vp = (i * atoms_bin) / (1000.0 * Na * AuM)
            radius = math.pow(3.0 * Vp / (4.0 * math.pi), 1.0/3.0)
            Atot[j]+=0.5*C[j,i]*math.exp(3.32111*math.log1p(radius*2*10**9)+10.80505)# D in nm
    return numpy.array(Atot)

gap = 5 # number of model time steps per experimental time step (for plotting)
def ConError(params):                                          #Error
    S=odeint(df, N0, t,args=(params,),rtol=1e-9,atol=1e-9,hmax=1e-1)
    Atot=numpy.zeros(tsteps)
    Atotal(S)
    error=0
    for i in range (0,numpy.size(Abs)):
        error += (Atot[i*gap]-Abs[i])**2
        print(error,params[0],params[1])
    return numpy.sqrt(error)
    

tsteps = gap*numpy.size(Abs)
time=7.7*60
t = numpy.linspace(0,time,tsteps)
sol = odeint(df, N0, t, args=(kparams,),rtol=1e-9,atol=1e-9,hmax=1e-1)
Atot=numpy.zeros(tsteps)
Atotal(sol)

A=0.0
Ctot=0.0
diameter=numpy.zeros(bins)
for i in range (1,bins):
    Vp = (i * atoms_bin) / (1000.0 * Na * AuM)
    diameter[i]=(2*10**9)*math.pow(3.0 * Vp / (4.0 * math.pi), 1.0/3.0)#nm
    A+=sol[tsteps-1,i]*diameter[i]
    Ctot+=sol[tsteps-1,i]
avgD=A/Ctot   #nm
end=numpy.array(sol[tsteps-1,:])
end[0]=0
median=(2*10**9)*math.pow(3.0 *(end.argmax()* atoms_bin) / (1000.0 * Na * AuM) / (4.0 * math.pi), 1.0/3.0)

Finalconcs=numpy.zeros(numpy.size(Grays))
for i in range(1,numpy.size(Grays)):
    Finalconcs[i]=numpy.array(Absorbances[i,5])/math.exp(3.32111*math.log1p(Diameters[i])+10.80505)

pylab.figure(1)
pylab.plot(t,Atot,Time*60,Abs,'ro')
pylab.title('Total Absorbance')
pylab.legend(('A','Data'))
pylab.ylabel('Absorbance')
pylab.xlabel('time(min)')
#pylab.savefig('Abs.png')

pylab.figure(2)
pylab.plot(t,sol[:,0],t,sol[:,1],t,sol[:,2])
pylab.legend(('Bin0','Bin1','Bin2'))
pylab.ylabel('Concentration(M)')
pylab.xlabel('time(min)')
#pylab.savefig('Conc1.png')

pylab.figure(3)
pylab.plot(t,sol[:,15],t,sol[:,25],t,sol[:,45])
pylab.legend(('Bin15','Bin25','Bin45'))
pylab.ylabel('Concentration(M)')
pylab.xlabel('time(min)')
#pylab.savefig('Conc2.png')

pylab.figure(4)
pylab.plot(sol[tsteps-1,:])
pylab.ylabel('Concentration(M)')
pylab.xlabel('Bins')
pylab.title('Final Size Distribution')
#pylab.savefig('Bins.png')
pylab.show()

'''
#REQUIRE MULTIPLE RUNS
pylab.figure(5)
pylab.plot(diameter,sol[tsteps-1,:])#,Diameters,Finalconcs,'ro')
pylab.ylabel('Concentration(M)')
pylab.xlabel('Particle Diameter(nm)')
pylab.title('Size Distributions for Different Grays')

pylab.figure(6)
pylab.plot(Grays,Diameters,'ro',Grays[num],median,'bo')
pylab.ylabel('Particle Diameter(nm)')
pylab.xlabel('Radiation(Gy)')
pylab.title('Median Particle Diameter for Different Grays')
'''

out=numpy.array([Atot[tsteps-1],avgD])     #Final Abs,Average Size
print(out)
#fmin=numpy.array(fmin(ConError,kparams))
#out=numpy.r_[kparams[None,:],out[None,:],fmin[None,:]]
#numpy.savetxt('out.txt',out)    
out=numpy.r_[kparams[None,:],out[None,:]]
print(out)
#fmin(ConError,kparams)
