#!/usr/bin/python

import numpy as np
from numpy.linalg import inv
from numpy import dot

from Probleme_R import *

###############################################################################
#                                                                             #
#  STRUCTURES DE DONNEES NECESSAIRES A LA RESOLUTION DES EQUATIONS DU RESEAU  #
#                                                                             #
#  Structures_N : matrices normales                                           #
#                                                                             #
###############################################################################

# Matrices issues de la topologie du reseau
#
# A    : matrice d'incidence noeuds-arcs du graphe        : M(m,n)
# Ar   : sous-matrice de A correspondant aux reservoirs   : M(mr,n)
# Ad   : sous-matrice complementaire de Ar pour A         : M(md,n)
# AdT  : plus grande sous-matrice carree inversible de Ad : M(md,md)
# AdI  : matrice inverse de AdT                           : M(md,md)
# AdC  : sous-matrice complementaire de AdT pour Ad       : M(md,n-md)
# B    : matrice d'incidence arcs-cycles du graphe        : M(n,n-md)
#
# Debit admissible
#
# q0   : vecteur des debits admissibles des arcs          : M(n,1)

from Probleme_R import *

##### Matrice d'incidence et sous-matrices associees

# Matrice d'incidence noeuds-arcs du graphe
A = np.zeros((m, n))
for i in range(m):
    A[i, orig == i] = -1
    A[i, dest == i] = +1

# Partition de A suivant le type des noeuds
Ar = A[:mr,:]
Ad = A[mr:m,:]

# Sous-matrice de Ad associee a un arbre et inverse
AdT = Ad[:,:md]
AdI = inv(AdT)

# Sous matrice de Ad associee a un coarbre
AdC = Ad[:,md:n]

# Matrice d'incidence arcs-cycles
B = np.zeros((n, n-md))
B[:md,:] = -dot(AdI, AdC)
B[md:,:] = np.eye(n-md)

##### Vecteur des debits admissibles
q0 = np.zeros(n)
q0[:md] = dot(AdI,fd)


### Oracle

def OraclePG(qc,ind):
    '''
    qc:  vecteur de R^(n-md)
    ind: entier
    '''
    # Fonction F: R^(n-md) -> R
    Bqc = dot(B,qc)
    u = q0 + Bqc
    F1 = (1/3)*( dot( u,np.transpose( r*u*abs(u) ) ) )
    F2 = dot( dot( Ar,u ) , pr )
    F = F1 + F2
    
    # Fonction G : R^(n-md) -> R^(n-md)
    Bt = np.transpose(B)
    G = dot( Bt,r*Bqc*abs(Bqc) ) + dot( np.transpose( dot(Ar,B) ),pr )
    
    # Condition bool
    if ind == 2:
        return F
    elif ind == 3:
        return G
    elif ind == 4:
        return (F,G)
    else:
        raise ValueError("ind must be in {2,3,4}")

def OraclePH(qc,ind):
    '''
    qc:  vecteur de R^(n-md)
    ind: entier
    '''
    # Fonction F: R^(n-md) -> R
    Bqc = dot(B,qc)
    u = q0 + Bqc
    F1 = (1/3)*( dot( u,np.transpose( r*u*abs(u) ) ) )
    F2 = dot( dot( Ar,u ) , pr )
    F = F1 + F2
    
    # Fonction G : R^(n-md) -> R^(n-md)
    Bt = np.transpose(B)
    G = dot( Bt,r*Bqc*abs(Bqc) ) + dot( np.transpose( dot(Ar,B) ),pr )
    
    # Fonction H: R^(n-md) -> R^(n-md)xn
    H = 2*dot( dot(B,r) , dot(abs(u),B ))
    
    # Condition Bool
    
    if ind==2:
        return F
    elif ind == 3:
        return G
    elif ind == 4:
        return (F,G)
    elif ind == 5:
        return H
    elif ind == 6:
        return (G,H)
    elif ind == 7:
        return (F,G,H)
    else:
        raise ValueError("ind must be in {2;...;7}")
    
        
    
    

