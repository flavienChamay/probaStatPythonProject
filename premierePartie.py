# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Installer avec pip3 numpy, scipy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as loiT

#Questions
#Q1

vect1 = np.array([1, 4, 2, 9, 14, 3, 16],int)
vect2 = np.array([[6, 2, 15, 8, 24, 7]],int)
vect3 = np.array([[1,4,5,7], [5,7,3,11], [18,4,9,2]], int)
print(vect1)
print(vect2)
print(np.shape(vect2))#Nbre de ligne, nbre de colonne
print(vect3)

#Q2
vectIncr = np.arange(3, 20, 0.5)
print(vectIncr)

#Q3
mat1 = np.eye(3) * 4
print(mat1)

a = np.array([[1,1],[1,1], [0,0]])
b = 2 * np.eye(3)
mat2 = np.concatenate((b,a), axis = 1)
print(mat2)

#Q4
B = np.random.rand(3, 5)
print(B)

C = np.random.rand(3, 5)
print(C)

#Q5
F = B * C
print(F)

Ct = np.transpose(C)
G = np.dot(B, Ct)
print(G)

#Q6
liste = (F < 0.6) * (F > 0.2) #Le * est l'equivalent du and
print(F[liste])
print(G[G >= 0.6])

#Representation des lois statistiques 
print("Representation des lois statistiques")

#Q1
print("Q1")
abscisseX = np.arange(-1, 5)
loiNormale = loiT.norm.pdf(abscisseX, 2, 1) #N(2,1) veut dire dans l'ordre de rentree des valeurs
plt.figure()
plt.plot(abscisseX, loiNormale)
plt.show()

#Q2
print("Q2")
abscisseX2 = np.arange(0, 10)
loiNormale2 = loiT.norm.pdf(abscisseX2, 5, 3)
loiExp = loiT.expon.pdf(abscisseX2, 1)
plt.figure()
plt.plot(abscisseX2, loiExp, loiNormale2)
plt.show()

#Q3
print("Q3")
loiNormale3 = np.random.normal(5, 1, 1000)
plt.figure()
histo, bins, patch = plt.hist(loiNormale3)
print("bins = ")
print(bins)
print("histo = ")
print(histo)
print("patch = ")
print(patch)
plt.plot()
plt.show()


