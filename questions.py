#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:13:48 2018

@author: user1
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as loiT

#Questions
#Q1
vect1 =  np.array ([1,4,2, 9, 14, 3, 16], int)
vect2 = np.array([[6, 2, 15, 8, 24, 7]], int)
vect3 = np.array([[1,4,5,7], [5,7,3,11], [18,4,9,2]], int)
print(vect1)
print(vect2)
print(np.shape(vect2))#(nbre de ligne, nbre de colonne)
print(vect3)

#Q2
vectIncr = np.arange(3,  20 + 1, 0.5)
print(vectIncr)
#Q3
mat1 = 4 * np.eye(3,3)
print(mat1)
a = np.array([[1,1],[1,1],[0,0]])
b = 2 * np.eye(3,3)
mat2 = np.concatenate((b, a), axis = 1)
print(mat2)
#Q4
B = np.random.rand(3,5)
print(B)
C = np.random.rand(3,5)
print(C)
#Q5
F = B*C
print(F)
Ct = np.transpose(C)
G = np.dot(B, Ct)
print(G)
#Q6
print("Q6 \n")
X = (F<0.6) * (F>0.2) #le * est l'equivalent du and
print(F[X])
print("G = ")
print(G[G>=0.6])

#Representation des lois statistiques
#Q1
print("Q1")
absX = np.arange(-1, 5)
LoiNormale = loiT.norm.pdf(absX, 1, 2)
plt.figure()
plt.plot(absX, LoiNormale)
plt.show()
#Q2
print("Q2")
absX2 = np.arange(0,10)
LoiNormale2 = loiT.norm.pdf(absX2, 5, 3)
LoiExp = loiT.expon.pdf(absX2,1)
plt.figure()
plt.plot(absX2, LoiExp, LoiNormale2)
plt.show()
#Q3

LoiNormale3 = np.random.normal(5, 1, 1000)
plt.figure()
hist, bins, patch = plt.hist(LoiNormale3)
print("bins =")
print(bins)
print("hist =")
print(hist)
print("patch =")
print(patch)
plt.plot()
plt.show()


