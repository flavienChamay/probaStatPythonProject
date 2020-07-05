import numpy as np
import numpy.random as loi
import scipy.stats as loiT
import matplotlib.pyplot as plt

# TP PROBAS et STATISTIQUES en PYTHON 2
# EXERCICE 1 : comparaison de lois uniformes : theorique & pratique

# Loi uniforme reelle
def loi_unif(n,a,b,barres,pas):

    # Loi pratique (valeurs aleatoires)
    xp = loi.uniform(a, b, size = n)

    # Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)

    # Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.uniform.pdf(vec, a, b - a)

    # Affichage
    plt.figure()
    plt.hist(xp, barres, density=True, label='resultat pratique')
    plt.plot(vec, xt, 'r', label='resultat theorique')
    plt.title('Loi uniforme')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

# Loi exponentielle
def loi_exp(n, mu, barres, pas):

    # Loi pratique (valeurs aleatoires)
    xp = loi.exponential(mu, size = n)

    # Normalisation
    mini = 0
    maxi = np.max(xp)

    # Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.expon.pdf(vec, loc = 0, scale = mu)

    # Affichage
    plt.figure()
    plt.hist(xp, barres, density=True, label='resultat pratique')
    plt.plot(vec, xt, 'r', label='resultat theorique')
    plt.title('Loi exponentielle')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

#loi geometrique
def loi_geo(n, p, barres, pas):

    # Loi pratique (valeurs aleatoires)
    xp = loi.geometric(p, size = n)

    # Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)

    # Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.geom.pmf(vec,p)

    # Affichage
    plt.figure()
    plt.hist(xp, barres, density=True, label='resultat pratique')
    plt.plot(vec, xt, 'or', label='resultat theorique')
    plt.title('Loi geometrique')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

#loi de Poisson
def loi_poisson(n, mu , barres, pas):

    # Loi pratique (valeurs aleatoires)
    xp = loi.poisson(mu, size=n)

    # Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)

    # Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.poisson.pmf(vec, mu)

    # Affichage
    plt.figure()
    plt.hist(xp, barres, density=True, label='resultat pratique')
    plt.plot(vec, xt, 'or', label='resultat theorique')
    plt.title('Loi de poisson')
    plt.xlabel('Intervalles')
    plt.ylabel('Probabilites')
    plt.legend()
    plt.show()

#loi normale
def loi_normale(n, mu, sigma, barres, pas):

    # Loi pratique (valeurs aleatoires)
    xp = loi.normal(mu, sigma, size=n)

    # Normalisation
    mini = np.min(xp)
    maxi = np.max(xp)

    # Loi theorique
    vec = np.arange(mini, maxi, pas)
    xt = loiT.norm.pdf(vec, mu, sigma)

    # Affichage
    plt.figure()
    plt.hist(xp, barres, density=True, label='resultat pratique')
    plt.plot(vec, xt, 'r', label='resultat theorique')
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
    X = loi.uniform(low = borneMin, high = borneMax, size = taille)
    Y = loi.uniform(low = borneMin, high = borneMax, size = taille)
    vectX = np.arange(borneMin, borneMax + 1, 1)
    esperanceX = np.mean(X)
    esperanceY = np.mean(Y)
    varianceX = np.var(X)
    covarianceXY = np.cov(X, Y)[0,1]
    vectY = (covarianceXY/varianceX) * (vectX - esperanceX) + esperanceY
    return (vectX, vectY)

#criteredechi2= N * Sommede 0 a N((ProbaStat - ProbaTheo)^2 / ProbaTheo)
#np.sum() : pour la somme de 0 a N et il le fait automatiquement
def chi2(probaStat, probaTheo, taille):
    return(taille * np.sum(((probaStat - probaTheo)**2)/probaTheo))

# DEBUT DU PROGRAMME PRINCIPAL

# Constante
nb_barres = 20
pas_reel = 0.02
pas_discret = 1

# (a) Tests de la loi Uniforme : loi discrete ou reelle au choix...
# 50 valeurs suivant une loi uniforme (min=0 & max=20)
loi_unif(50, 0, 20, nb_barres, pas_reel)
# 10000 valeurs suivant une loi uniforme (min=0 & max=20)
loi_unif(10000, 0, 20, nb_barres, pas_reel)
# 10000 valeurs suivant une loi uniforme (min=-5 & max=5)
loi_unif(10000, -5, 5, nb_barres, pas_reel)

# (b) Tests de la loi Exponentielle : loi reelle
print("b) ")
loi_exp(50, 0.02, nb_barres, pas_reel)
loi_exp(10000, 0.02, nb_barres, pas_reel)
loi_exp(10000, 0.8, nb_barres, pas_reel)

#(c) Tests loi geometrique (en discret)
loi_geo(50, 0.07, nb_barres, pas_discret)
loi_geo(10000, 0.07, nb_barres, pas_discret)
loi_geo(10000, 0.2, nb_barres, pas_discret)
#(d) Tests loi de Poisson (endiscret)
loi_poisson(50, 5, nb_barres, pas_discret)
loi_poisson(10000, 5, nb_barres, pas_discret)
loi_poisson(10000, 0.5, nb_barres, pas_discret)
loi_poisson(10000, 50, nb_barres, pas_discret)
#(e) Tests loi normale : loi reelle
loi_normale(50, 0, 1, nb_barres, pas_reel)
loi_normale(10000, 0, 1, nb_barres, pas_reel)
loi_normale(10000, 5, 0.5, nb_barres, pas_reel)
loi_normale(10000, 50, 500, nb_barres, pas_reel)

#Exercice 2

#1)
X1 = loi.uniform(low = 10, high = 20, size = 1000)
X2 = loi.uniform(low = 10, high = 20, size = 10000)
X3 = loi.uniform(low = 10, high = 20, size = 100000)
#2)
print ("esperance exp X1 = ", np.mean(X1))
print ("esperance exp X2 = ", np.mean(X2))
print ("esperance exp X3 = ", np.mean(X3))
print ("variance exp X1 = ", np.var(X1))
print ("variance exp X2 = ", np.var(X2))
print ("variance exp X3 = ", np.var(X3))
#3)
print ("experance theo uniforme = ", esperanceTheo_uniformeReel(10, 20))
print ("variance theo uniforme  = ", varianceTheo_uniformeReel(10, 20))
#plus lechantillon est grand et plus les valerus theo et exp sont proches

#4)
#a)
E1nor = loi.normal(0, 1,size = 1000)
E2nor = loi.normal(0, 1,size = 10000)
E3nor = loi.normal(0, 1,size = 100000)
print ("experance exp E1nor = ", np.mean(E1nor))
print ("experance exp E2nor = ", np.mean(E2nor))
print ("experance exp E3nor = ", np.mean(E3nor))
print ("variance exp E1nor = ", np.var(E1nor))
print ("variance exp E2nor = ", np.var(E2nor))
print ("variance exp E3nor = ", np.var(E3nor))
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
#1)
X = loi.normal(0, 1, size = 1000)
#2)
Y = loi.uniform(low = 10, high = 20, size = 1000)
#3)
Z = loi.uniform(low = 0, high = 1, size =1000)
#4)
print ("covariance de X et Y = ", np.cov(X, Y))
print ("covariance de X et Z = ", np.cov(X, Z))
print ("covariance de Y et Z = ", np.cov(Y, Z))
#5)
#matrice 4 valeurs prendre la [0,1]
#6)
#Si covariance faible alors donnees beaucoup de differences entre elles
#Si covariance faible alors donnees sont independantes
#!!!!!!!Ce nest pas reciproque!!!!!!!!!!!
#cest avec la correlation quon va savoir si la reciproque est vraie

#Exercice 4
print ("coeff de corr de X et X + Y = ", np.corrcoef(X, X + Y))
print ("coeff de corr de X et X * Y = ", np.corrcoef(X, X*Y))
print ("coeff de corr de 2*X+Y et 3*X + Y  = ", np.corrcoef(2*X+Y, 3*X+Y))
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
#1)
X = loi.uniform(low = 0, high = 9, size = 20)
Y = loi.uniform(low = 0, high = 9, size = 20)
#2)
#X cest les abscisses et Y sont les ordonnes
plt.figure()
plt.plot(X, Y, 'kx') #kx pour croix noire
plt.title('exercice5')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
#3
esperanceX = np.mean(X)
esperanceY = np.mean(Y)
varianceX = np.var(X)
covarianceXY = np.cov(X, Y)[0,1] #pour prendre la valeur pertinente
#4
minimum = np.min(X)
maximum = np.max(X)
vect = np.arange(minimum, maximum + 1, 1)
#ou alors min = 0 et max = 9 car tout est dit sur lintervalle code avant uniqument pour la loi uniforme
droite = (covarianceXY/varianceX) * (vect - esperanceX) + esperanceY
plt.plot(vect, droite, 'r')
#par defaut plot il fait une droite
#r pour red
plt.show()
#5
#voir fonction
#6
#est-ce correct?
(petitx, petity) = ligne_regression(minimum, maximum, 20)
plt.figure()
plt.title('exercice ligne de regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(petitx, petity, 'r')
plt.show()
#7
(x10,y10) = ligne_regression(minimum, maximum, 10)
(x100,y100) = ligne_regression(minimum, maximum, 100)
(x1000,y1000) = ligne_regression(minimum, maximum, 1000)
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
print("Exo6")
#1
N = 1000#nombre de valeur tiree aleatoirement
X = loi.uniform(low = 0, high = 1, size = N) #on tire 1000 valeurs aleatoirement suivant la loi uniforme
#2
#khi2 = difference daire entre courbe theorique et courbe statistique
pas = 0.2#sert pour les intervalles qui vont de 0.2 en 0.2
plt.figure()
histogramme = plt.hist(X, int(1/pas), normed=True)[0] #hist retourne un tableau a 3 valeurs et je prends le premier
#ici dans histogramme  on a un histogramme qui compte le nombre de valeur qui appartiennent a lintervalle
N_intervalles = int((1-0)/pas) #nombres dintervalles ; ce nest pas le nombre de valeurs tire aleatoirement

plt.title('Loi uniforme de X')
plt.xlabel('Intervalles')
plt.ylabel('Nombre d elements dans lintervalle')
plt.legend()
plt.show()
#3 !!!!!!!Ne marche pas!!!!!!!!!
#a)
#on foit faire les proba
proba_stat = histogramme / N #on divise le nombre de valeurs pour chaque intervalle par le nombre de valeur total
proba_theo = 1 /N_intervalles #loi de densite de la loi uniforme est toujour de (b-a)/n_intervalles
critereChi2 = chi2(proba_stat, proba_theo, N_intervalles)
print ("critere du chi2 = ", critereChi2)
#degres de liberte = nombre dintervalles - nombre de contraintes (ie nbre dequations; fixe a 1 par defaut)
#par defaut, nbre de contraintes = 1
#b)
degDeLib = N_intervalles - 1
probaChi2 = 1-loiT.chi2.cdf(critereChi2, degDeLib)
print ("probabilite de conformite des lois theoriques et statistiques = ", probaChi2)
