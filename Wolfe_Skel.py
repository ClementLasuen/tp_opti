#!/usr/bin/python

import numpy as np
from numpy import dot
from numpy.linalg import norm

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
        
        # Point precedent pour tester l'indistinguabilite
        xp = xn
        
        # Point actuel
        xn = x + alpha_n*D
        
        # Calcul des conditions de Wolfe

        argout_n = Oracle(xn)
        critere_n = argout_n[0]
        gradient_n = argout_n[1]
        
        
        cond1 = critere_n - critere - alpha_n*omega_1*np.dot(gradient, D) <=0
        
        cond2 =  np.dot( gradient_n , D) - omega_2 * np.dot(gradient, D) >=0
        
        # Test des conditions de Wolfe
        # - si les deux conditions de Wolfe sont verifiees,
        #   faire ok = 1 : on sort alors de la boucle while
        # - sinon, modifier la valeur de alphan : on reboucle.
        
    
        if not cond1 : # condition 1 non verifiée
            alpha_max = alpha_n
            alpha_n = (alpha_min + alpha_max)/2
        else :
            if not cond2 : # condition 2 non verifiée
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
    
def Gradient_V(Oracle,x0):
    print("Begin Gradient_V ***")
    x = x0
    epsilon = 0.0001
    compteur = 0
    gradient = Oracle(x0)[1]

    alpha = 1
    print("Begin boucle")
    while norm(gradient) > epsilon and compteur < 1000 :
        
        alpha = Wolfe(alpha,x,  -gradient  ,Oracle)[0]
        ok = Wolfe(alpha,x,  -gradient  ,Oracle)[1]

        print("n° ",compteur," alpha = ",alpha," norme grad = ",norm(gradient)," OK = ",ok)
        x -= alpha*gradient
        gradient = Oracle(x,3)
        compteur +=1
    print("End boucle")
    return Oracle(x)[0],Oracle(x)[1],x,compteur

def Polak_Ribiere(x0,Oracle):
    
    
    epsilon = 0.0001
    k = 1
    g_k_1 = Oracle(x0,3) # gradient à l'étape k-1
    g_k = Oracle(x0,3) # gradient à l'étape k
    d_k = -g_k # direction de descente à l'étape k
    
    beta = 1
    alpha = 1
    
    x = x0 + d_k
    
    while np.linalg.norm(g_k) >= epsilon and k < 1000 :
        
        # gradient à l'etape k
        g_k = Oracle(x,3)
            
        alpha = Wolfe(alpha, x ,  -g_k  ,Oracle)[0]
        beta = np.dot(g_k, g_k - g_k_1)/np.linalg.norm(g_k_1)**2
        
                
        d_k = -g_k + beta*d_k
        
        x += beta*d_k # x_k+1

        g_k_1 = g_k
    
    return Oracle(x)[0],Oracle(x)[1],x,k

 def BFGS(x0,Oracle):
     
     epsilon = 0.0001
     k = 0
     g_k_1 = Oracle(x0,3) # gradient à l'étape k-1
     

     
     beta = 1
     alpha = 1
     
     n = d_k.shape[0]
     W = np.eye(n)
     
     x_k = x0 -g_k_1
     x_k_1 = x0
     delta_x = x_k
     delta_g = g_k
     
     g_k = Oracle(x_k,3) # gradient à l'étape k
     
     while np.linalg.norm(g_k) >= epsilon and k < 10 :
         
         g_k = Oracle(x_k,3)
         
         # J'arrive pas à calculer la MATRICE delta_x ^T * delta_g
         # dot calcule toujours le produit scalaire
         delta_x = x_k- x_k_1
         delta_g = g_k - g_k_1
         A_xg = W
         A_gx = W
         A_xx = W
         for i in range(n):
             for j in range(n):
                 A_xg[i,j] = delta_x[i]*delta_g[j] /(delta_x @ delta_g)
                 A_xg[i,j] = delta_g[i]*delta_x[j] /(delta_x @ delta_g)
                 A_xg[i,j] = delta_x[i]*delta_x[j] /(delta_x @ delta_g)
                 
         W = (np.eye(n)- A_xg) @ W @ ((np.eye(n)- A_gx)) + A_xx
         alpha = Wolfe(alpha, x ,  -g_k  ,Oracle)[0]
         
         x_k_1 = x_k
         x_k -= alpha* (W @ g_k)
         
         k+=1
    return Oracle(x_k)[0],Oracle(x_k)[1],x_k,k

def Newton(x0,Oracle):
    
    epsilon = 0.0001
    k=0
    x = x0
    
    G, H = Oracle(x0,6)
    
    d = np.linalg.solve(H,G)
    alpha =1
    
    while np.linalg.norm(G) >= epsilon and k < 10 and np.linalg.det(H) != 0 :
        
        d = np.linalg.solve(H,G)
        k+=1
        alpha = Wolfe(alpha, x ,  d  ,Oracle)[0]
        x += alpha*d
        G, H = Oracle(x,6)
        
    return Oracle(x)[0],G,x,k
