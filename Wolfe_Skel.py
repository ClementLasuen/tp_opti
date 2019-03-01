#!/usr/bin/python

import numpy as np
from numpy import dot

from Structures_N import *
########################################################################
#                                                                      #
#          RECHERCHE LINEAIRE SUIVANT LES CONDITIONS DE WOLFE          #
#                                                                      #
#          Algorithme de Fletcher-Lemarechal                           #
#                                                                      #
########################################################################

#  Arguments en entree
#
#    alpha  : valeur initiale du pas
#    x      : valeur initiale des variables
#    D      : direction de descente
#    Oracle : nom de la fonction Oracle
#
#  Arguments en sortie
#
#    alphan : valeur du pas apres recherche lineaire
#    ok     : indicateur de reussite de la recherche 
#             = 1 : conditions de Wolfe verifiees
#             = 2 : indistinguabilite des iteres

def Wolfe(alpha, x, D, Oracle):
    
    ##### Coefficients de la recherche lineaire

    omega_1 = 0.1
    omega_2 = 0.9

    alpha_min = 0
    alpha_max = np.inf

    ok = 0
    dltx = 0.00000001

    ##### Algorithme de Fletcher-Lemarechal
    
    # Appel de l'oracle au point initial
    argout = Oracle(x)
    critere = argout[0]
    gradient = argout[1]
    
    # Initialisation de l'algorithme
    alpha_n = alpha
    xn = x
    
    # Boucle de calcul du pas
    while ok == 0:
        
        # xn represente le point pour la valeur courante du pas,
        # xp represente le point pour la valeur precedente du pas.
        xp = xn
        xn = x + alpha_n*D
        
        # Calcul des conditions de Wolfe
        
        new_argout = Oracle(xn)
        
        cond1 = new_argout[0] - critere - omega_1*np.dot(gradient, D) <=0
        
        cond2 =  np.dot( new_argout[1] , D) - omega_2 * np.dot(argout[1], D) >=0
        
        # Test des conditions de Wolfe
        # - si les deux conditions de Wolfe sont verifiees,
        #   faire ok = 1 : on sort alors de la boucle while
        # - sinon, modifier la valeur de alphan : on reboucle.
        
        argout = new_argout
        
        if not cond1 : # condition 1 non verifiée
            alpha_max = alpha_n
            alpha_n = (alpha_min + alpha_max)/2
        else :
            if not cond2 <0 : # condition 2 non verifiée
                alpha_min = alpha_n
                if alpha_max == np.inf :
                    alpha_n = 2*alpha_min
                else : 
                    alpha_n = (alpha_min + alpha_max)/2
            else : 
                ok = 1
                
        # Test d'indistinguabilite
        if np.linalg.norm(xn - xp) < dltx:
            ok = 2

    return alpha_n, ok

##-------------------------------------------------------------------------------
    
def Gradient_V(x0,OraclePG):
    
    x = x0
    epsilon = 0.0001
    compteur = 0
    gradient = OraclePG(x0,3)

    alpha = 1
    
    while np.linalg.norm(gradient) < epsilon and compteur < 1000 :
        alpha = Wolfe(alpha,x,  gradient  ,OraclePG)[0]
        x -= alpha*gradient
        gradient = OraclePG(x,3)
        compteur +=1
    return x, OraclePG(x,2)

def polak(x0,OraclePG):
    
    
    epsilon = 0.0001
    k = 1
    g_k_1 = OraclePG(x0,3) # gradient à l'étape k-1
    g_k = OraclePG(x0,3) # gradient à l'étape k
    d_k = -gradient # direction de descente à l'étape k
    
    beta = 1
    alpha = 1
    
    x = x0 + d_k
    
    while np.linalg.norm(gradient) >= epsilon and k < 1000 :
        
        # gradient à l'etape k
        g_k = OraclePG(x,3)
            
        alpha = Wolfe(alpha, x ,  gradient  ,OraclePG)[0]
        beta = np.dot(g_k, g_k - g_k_1)/np.linalg.norm(g_k_1)**2
        
                
        d_k = -g_k + beta*d_k
        
        x += beta*d_k # x_k+1

        gradient_k_1 = gradient_k
    
    return x,OraclePG(x,3),gradient_k

def BFGS(x0,OraclePG):
    
    epsilon = 0.0001
    k = 0
    gradient_k_1 = OraclePG(x0,3) # gradient à l'étape k-1
    gradient_k = OraclePG(x0,3) # gradient à l'étape k
    d_k = -gradient # direction de descente à l'étape k
    
    beta = 1
    alpha = 1
    
    x = x0 + d_k
    delta_x = x
    delta_g = 
    while np.linalg.norm(gradient) >= epsilon and k < 1000 :
        

        
    
    
    
    
