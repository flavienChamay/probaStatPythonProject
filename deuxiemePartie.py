# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Installer avec pip3 numpy, scipy, matplotlib

import numpy as np
import numpy.random as loi
import scipy.stats as loiT
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power

# TP PROBAS et STATISTIQUES en PYTHON 3
# EXERCICE 1 : comparaison de lois uniformes : theorique & pratique

#Loi uniforme reelle
def loi_unif(n, a, b, barres, pas):
    
    #Loi pratique (valeurs aleatoires)
    xp = loi.uniform(a, b , size = n)
    
    #Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)
    
    #Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.uniform.pdf(vec, a, b - a)
    
    #Affichage
    plt.figure()
    plt.hist(xp, barres, density = True, label = 'resultat pratique')
    plt.plot(vec, xt, 'r', label = 'resultat theorique')
    plt.title('Loi uniforme')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

#Loi exponentielle
def loi_exp(n, mu, barres, pas):
    
    #Loi pratique (valeurs aleatoires)
    xp = loi.exponential(mu, size = n)
    
    #Normalisation
    mini = 0
    maxi = np.max(xp)
    
    #Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.expon.pdf(vec, loc = 0, scale = mu)
    
    #Affichage
    plt.figure()
    plt.hist(xp, barres, density = True, label = 'resultat pratique')
    plt.plot(vec, xt, 'r', label = 'resultat theorique')
    plt.title('Loi exponentielle')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

#Loi geometrique
def loi_geo(n, p, barres, pas):
    
    #Loi pratique (valeurs aleatoires)
    xp = loi.geometric(p, size = n)
    
    #Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)
    
    #Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.geom.pmf(vec, p)
    
    #Affichage
    plt.figure()
    plt.hist(xp, barres, density = True, label = 'resultat pratique')
    plt.plot(vec, xt, 'or', label = 'resultat theorique')
    plt.title('Loi geometrique')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

#Loi de Poisson
def loi_poisson(n, mu, barres, pas):
    
    #Loi pratique (valeurs aleatoires)
    xp = loi.poisson(mu, size = n)
    
    #Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)
    
    #Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.poisson.pmf(vec, mu)
    
    #Affichage
    plt.figure()
    plt.hist(xp, barres, density = True, label = 'resultat pratique')
    plt.plot(vec, xt, 'or', label = 'resultat theorique')
    plt.title('Loi de Poisson')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

#Loi normale
def loi_normale(n, mu, sigma, barres, pas):
    
    #Loi pratique (valeurs aleatoires)
    xp = loi.normal(mu, sigma, size = n)
    
    #Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)
    
    #Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.norm.pdf(vec, mu, sigma)
    
    #Affichage
    plt.figure()
    plt.hist(xp, barres, density = True, label = 'resultat pratique')
    plt.plot(vec, xt, 'r', label = 'resultat theorique')
    plt.title('Loi normale')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

def esperanceTheo_uniformeReel(a, b):
    return (a+b)/2

def varianceTheo_uniformeReel(a, b):
    return ((b - a)**2)/12

def esperanceTheo_normale(mu):
    return mu

def varianceTheo_normale(sigma):
    return sigma**2

#Attention le parametre est mu = 1/lambda avec lambda ce quil  y a dans le memento
def esperanceTheo_exponentielle(muParam):
    return muParam

def varianceTheo_exponentielle(muParam):
    return muParam**2

def ligne_regression(borneMin, borneMax, taille):
    X = loi.uniform(borneMin, borneMax, taille)
    Y = loi.uniform(borneMin, borneMax, taille)
    vectAbscisse = np.arange(borneMin, borneMax + 1, 1)
    esperanceX = np.mean(X)
    esperanceY = np.mean(Y)
    varianceX = np.var(X)
    covarianceXY = np.cov(X, Y)[0,1]
    vectOrdonnee = (covarianceXY/varianceX) * (vectAbscisse - esperanceX) + esperanceY
    return (vectAbscisse, vectOrdonnee)

#criteredechi2= N * Sommede 0 a N((ProbaStat - ProbaTheo)^2 / ProbaTheo)
#np.sum() : pour la somme de 0 a N et il le fait automatiquement
def chi2(probaStat, probaTheo, taille):
    return (taille * np.sum(((probaStat - probaTheo)**2) /probaTheo))

#Donne l'etape correspondant a la convergence vers les probabilités limities d'une chaine de Markov
def convergenceProbaLimites(P, probaInit, nbEtapes):
    matP = np.mat(P)
    matProbaInit = np.mat(probaInit)
    
    matPowerP = matrix_power(matP, nbEtapes)
    prodMat = np.dot(matPowerP, matProbaInit)
    sommeProbas = np.sum(prodMat)
    
    return sommeProbas

#Programme Principal

#Constantes:
nb_barres = 20
pas_reel = 0.02
pas_discret = 1

#Exercice1
print("Exercice 1:")

# (a) Tests de la loi Uniforme : loi discrete ou reelle au choix...
print("a)")
# 50 valeurs suivant une loi uniforme (min=0 & max=20)
loi_unif(50, 0, 20, nb_barres, pas_reel)
# 10000 valeurs suivant une loi uniforme (min=0 & max=20)
loi_unif(10000, 0, 20, nb_barres, pas_reel)
# 10000 valeurs suivant une loi uniforme (min=-5 & max=5)
loi_unif(10000, -5, 5, nb_barres, pas_reel)

# (b) Tests de la loi Exponentielle : loi reelle
print("b)")
loi_exp(50, 0.02, nb_barres, pas_reel)
loi_exp(10000, 0.02, nb_barres, pas_reel)
loi_exp(10000, 0.8, nb_barres, pas_reel)

#(c) Tests loi geometrique (en discret)
print("c)")
loi_geo(50, 0.07, nb_barres, pas_discret)
loi_geo(10000, 0.07, nb_barres, pas_discret)
loi_geo(10000, 0.2, nb_barres, pas_discret)

#(d) Tests loi de Poisson (endiscret)
print("d)")
loi_poisson(50, 5, nb_barres, pas_discret)
loi_poisson(10000, 5, nb_barres, pas_discret)
loi_poisson(10000, 0.5, nb_barres, pas_discret)
loi_poisson(10000, 50, nb_barres, pas_discret)

#(e) Tests loi normale : loi reelle
print("e)")
loi_normale(50, 0, 1, nb_barres, pas_reel)
loi_normale(10000, 0, 1, nb_barres, pas_reel)
loi_normale(10000, 5, 0.5, nb_barres, pas_reel)
loi_normale(10000, 50, 500, nb_barres, pas_reel)

#Exercice 2
print("Exercice 2:")

#1)
print("Q1:")
X1 = loi.uniform(10, 20, 1000)
X2 = loi.uniform(10, 20, 10000)
X3 = loi.uniform(10, 20, 100000)

#2)
print("Q2:")
print("Epserance exprimentale de X1 = ", np.mean(X1))
print("Epserance exprimentale de X2 = ", np.mean(X2))
print("Epserance exprimentale de X3 = ", np.mean(X3))
print("variance experimentale de X1", np.var(X1))
print("variance experimentale de X2", np.var(X2))
print("variance experimentale de X3", np.var(X3))

#3)
print("Q3:")
print("Esperance theo uniforme = ", esperanceTheo_uniformeReel(10, 20))
print("Variance theo uniforme = ", varianceTheo_uniformeReel(10, 20))
#plus lechantillon est grand et plus les valerus theo et exp sont proches

#############Pour ecart type on peut utiliser: np.std(X1)

#4)
print("Q4:")

#a)
print("a)")
ExperimentalNormale1 = loi.normal(0, 1, 1000)
ExperimentalNormale2 = loi.normal(0, 1, 10000)
ExperimentalNormale3 = loi.normal(0, 1, 100000)
print ("esperance experimentale ExperimentalNormale1 = ", np.mean(ExperimentalNormale1))
print ("esperance experimentale ExperimentalNormale2 = ", np.mean(ExperimentalNormale2))
print ("esperance experimentale ExperimentalNormale3 = ", np.mean(ExperimentalNormale3))
print ("variance experimentale ExperimentalNormale1 = ", np.var(ExperimentalNormale1))
print ("variance experimentale ExperimentalNormale2 = ", np.var(ExperimentalNormale2))
print ("variance experimentale ExperimentalNormale3 = ", np.var(ExperimentalNormale3))
print ("experance theo normale = ", esperanceTheo_normale(0))
print ("variance theo normale = ", varianceTheo_normale(1))

#b)
E1Exp = loi.exponential(0.5, size = 1000)
E2Exp = loi.exponential(0.5, size = 10000)
E3Exp = loi.exponential(0.5, size = 100000)
print ("experance E1Exp = ", np.mean(E1Exp))
print ("experance E2Exp = ", np.mean(E2Exp))
print ("experance E3Exp = ", np.mean(E3Exp))
print ("variance exp E1Exp = ", np.var(E1Exp))
print ("variance exp E2Exp = ", np.var(E2Exp))
print ("variance exp E3Exp = ", np.var(E3Exp))
print ("experance exponentielle theo = ", esperanceTheo_exponentielle(0.5))
print ("variance exponentielle theo = ", varianceTheo_exponentielle(0.5))

#Exercice 3
print("Exercice 3: ")
#1)
print("1)")
X = loi.normal(0, 1, 1000)
#2)
print("2)")
Y = loi.uniform(10, 20, 1000)
#3)
print("3)")
Z = loi.uniform(0, 1, 1000)
#4)
print("4)")
print("covariance de X et Y = ", np.cov(X, Y))
print("covariance de X et Z = ", np.cov(X, Z))
print("covariance de Y et Z = ", np.cov(Y, Z))

#resultat de np.cov: est matrice 4 valeurs 
# var(X)   cov(X,Y)
# cov(Y,X) var(Y)
#5)
#matrice 4 valeurs prendre la [0,1]
#6)
#Si covariance faible alors donnees beaucoup de differences entre elles
#Si covariance faible alors donnees sont independantes
#!!!!!!!Ce nest pas reciproque!!!!!!!!!!!
#cest avec la correlation quon va savoir si la reciproque est vraie

#Exercice 4
print("Exercice 4: ")
print("Coefficient de correlation de X et X + Y = ", np.corrcoef(X, X + Y))
print("Coefficient de correlation de X et X * Y = ", np.corrcoef(X, X * Y))
print("Coefficient de correlation de 2 * X + Y et 3 * X + Y = ", np.corrcoef(2 * X + Y, 3 * X + Y))
print ("coeff de corr de 2*X+Y et 3*X + Y (juste resultat pertinent)  = ", np.corrcoef(2*X+Y, 3*X+Y)[0,1])
#np.corrcoef renvoie une matrice 2*2 de 4 valeurs contenant:
#corrcoef(X, X)
#corrcoef(X, X + Y) --> resultat qui nous interresse
#corrcoef(X + Y, X) -->idem (resultat identique au precedent)
#corrcoef(X + Y, X + Y)
#on peut mettre [0,1] a la fin pour extraire uniquement le resultat pertinent
print ("coeff de corr de X et X + Y = ", np.corrcoef(X, -X))
#si coeff de corr negatif alors donnees correlees de maniere oppose
#coeff de corr appartient a intervalle de -1 a 1

#Exercice 5
print("Exercice 5:")
#1)
print("Q1")
X = loi.uniform(0, 9, 20)
Y = loi.uniform(0, 9, 20)

#2)
print("Q2")
#X cest les abscisses et Y sont les ordonnes
plt.figure()
plt.plot(X, Y, 'kx') #kx pour croix noire
plt.title('exercice5')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#3
print("Q3")
esperanceX = np.mean(X)
esperanceY = np.mean(Y)
varianceX = np.var(X)
covarianceXY = np.cov(X, Y)[0,1] #pour prendre la valeur pertinente
print(covarianceXY)
#4
print("Q4")
minimum = np.min(X)
maximum = np.max(X)
vectAbscisse = np.arange(minimum, maximum + 1, 1) #De min a max+1 par pas de 1
#ou alors min = 0 et max = 9 car tout est dit sur lintervalle code avant uniqument pour la loi uniforme
droite = (covarianceXY/varianceX) * (vectAbscisse - esperanceX) + esperanceY
plt.plot(vectAbscisse, droite, 'r')#par defaut plot il fait une droite, r pour red
plt.show()

#5
print("Q5")
#voir fonction

#6
print("Q6")
(abscisseDroite, ordonneeDroite) = ligne_regression(minimum, maximum, 20)
plt.figure()
plt.title('exercice ligne de regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(abscisseDroite, ordonneeDroite, 'r')
plt.show()

#7
print("Q7")
(x10, y10) = ligne_regression(minimum, maximum, 10)
(x100, y100) = ligne_regression(minimum, maximum, 100)
(x1000, y1000) = ligne_regression(minimum, maximum, 1000)
plt.figure()
plt.title('exercice ligne de regression 10, 100, 1000')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x10, y10, 'r')
plt.plot(x100, y100, 'b')
plt.plot(x1000, y1000, 'g')
plt.show()
#on remarque que plus lechantillion augmente et plus la droite devient horizontale
#le x et le y de la droite de regression correspondent a la moyenne de la 
#loi uniforme utilise pour laxe X et laxe Y
#On ne peut pas dire que cest une loi uniforme

#Exercice 6
print("Exercie 6:")

#1
print("Q1")
N = 1000#nombre de valeur tiree aleatoirement
X = loi.uniform(0, 1, N) #on tire 1000 valeurs aleatoirement suivant la loi uniforme

#2
print("Q2")
#khi2 = difference daire entre courbe theorique et courbe statistique
pas = 0.2#sert pour les intervalles qui vont de 0.2 en 0.2
plt.figure()
#ici dans histogramme  on a un histogramme qui compte le nombre de valeur qui appartiennent a lintervalle
histogramme = plt.hist(X, int(1/pas), density = True)[0]#hist retourne un tableau a 3 valeurs et je prends le premier
N_intervalles = int((1-0)/pas) #nombres dintervalles ; ce nest pas le nombre de valeurs tire aleatoirement
plt.title('Loi uniforme de X')
plt.xlabel('Intervalles')
plt.ylabel('Nombre d elements dans lintervalle')
plt.legend()
plt.show()

#3
print("Q3")
#a)
#on foit faire les proba theo et stat pour avoir le chi2
proba_stat = histogramme / N #on divise #on divise le nombre de valeurs pour chaque intervalle par le nombre de valeur totalle nombre de valeurs pour chaque intervalle par le nombre de valeur total
proba_theo = 1 / N_intervalles #loi de densite de la loi uniforme est toujour de (b-a)/n_intervalles
critereChi2 = chi2(proba_stat, proba_theo, N_intervalles)
print("critere du chi2 pour la loi uniforme = ", critereChi2)

#b)
#degres de liberte = nombre dintervalles - nombre de contraintes (ie nbre dequations; fixe a 1 par defaut)
#par defaut, nbre de contraintes = 1
degDeLib = N_intervalles - 1
probaChi2 = 1-loiT.chi2.cdf(critereChi2, degDeLib)
print ("probabilite de conformite des lois theoriques et statistiques = ", probaChi2)

#conclusion ???????????????


#Exercice 7
print("Exercice 7:")
#1
print("Q1")
arrayP = np.array([[0.5, 0.25, 0.25], [1/3, 1/3, 1/3], [0.2, 0.8, 0]])
arrayProbaInit = np.array([[1],[0],[0]])

#2
print("Q2")
print("Pour 1 etape = ", convergenceProbaLimites(arrayP, arrayProbaInit, 1))
print("Pour 2 etapes = ", convergenceProbaLimites(arrayP, arrayProbaInit, 2))
print("Pour 4 etapes = ", convergenceProbaLimites(arrayP, arrayProbaInit, 4))
print("Pour 8 etapes = ", convergenceProbaLimites(arrayP, arrayProbaInit, 8))
print("Pour 16 etapes = ", convergenceProbaLimites(arrayP, arrayProbaInit, 16))
print("Pour 32 etapes = ", convergenceProbaLimites(arrayP, arrayProbaInit, 32))

#3
print("Q3")
#On converge vers 16 étapes

#Exercice 8
print("Exercice 8:")








