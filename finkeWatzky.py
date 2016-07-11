# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:24:41 2014

@author: jheys
"""

import numpy
import pylab
from scipy.integrate import odeint

k1 = 0.001
k2 = 1.0

# ODE definition
def df(c,t):
    A = c[0]
    B = c[1]
    dAdt = -k1*A-k2*A*B
    dBdt = k1*A+k2*A*B
    return numpy.array([dAdt,dBdt]) 
    
# initial condition
c0 = numpy.array([2.0, 0.0])
t = numpy.linspace(0,10.0,1000)
sol = odeint(df, c0, t)
pylab.plot(t,sol)
pylab.xlabel("time (dimensionless)")
pylab.ylabel("concentration (dimensionless)")
