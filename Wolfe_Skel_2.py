
#!/usr/bin/python

import numpy as np
from numpy import dot
from numpy.linalg import norm
from Visualg import Visualg

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

N_it = 5000
epsilon = 0.0001
visualisation = True

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

### Algorithme de résolutions


def Gradient_V(Oracle,x0):
    #print("Begin Gradient_V ***")
    x = x0
    compteur = 0
    gradient = Oracle(x0)[1]
    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    alpha = 1
    #print("Begin boucle")
    while norm(gradient) > epsilon and compteur < N_it :
        
        alpha = Wolfe(alpha,x,  -gradient  ,Oracle)[0]
        ok = Wolfe(alpha,x,  -gradient  ,Oracle)[1]

        #print("n° ",compteur," alpha = ",alpha," norme grad = ",norm(gradient)," OK = ",ok)
        x -= alpha*gradient
        gradient = Oracle(x,3)
        compteur +=1
        
        gradient_norm_list.append(norm(gradient))
        gradient_step_list.append(alpha)
        critere_list.append(Oracle(x,2))
        
    #print("End boucle")
    if visualisation == True:
        Visualg(gradient_norm_list, gradient_step_list, critere_list)
    if compteur>=N_it : print("BFGS, k saturated")
    return Oracle(x)[0],Oracle(x)[1],x,compteur

## *************************************

def Polak_Ribiere(Oracle, x0):
    print("begin PR")
    # Initialisation des variables
    
    iter_max = 1000
    alpha = 1
    epsilon = 0.000001
    
    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    #time_start = process_time()
    
    x = x0
    
    old_g = Oracle(x)[1]
    D = np.zeros(n-md)
    compteur =0

    # Boucle sur les iterations

    for k in range(iter_max):
        
        # Valeur du critere et du gradient
        #critere, gradient = Oracle(x)
        critere, new_g = Oracle(x)[0],Oracle(x)[1]

        # Test de convergence
        gradient_norm = np.linalg.norm(new_g)
        if gradient_norm <= epsilon:
            break

        # Direction de descente
        Beta = np.vdot(new_g, new_g-old_g)/(norm(old_g))**2
        D = -new_g + Beta*D
        alpha = Wolfe(1, x, D, Oracle)[0]
        
        # Mise a jour des variables
        x = x + (alpha*D)
        
        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(alpha)
        critere_list.append(critere)
        
        old_g = new_g
        compteur +=1
    if visualisation == True:
        Visualg(gradient_norm_list, gradient_step_list, critere_list)
    return Oracle(x)[0],Oracle(x)[1],x,compteur


def BFGS(Oracle, x0):
    # Initialisation des variables
    
    iter_max = 1000
    alpha = 1
    epsilon = 0.000001
    
    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    #time_start = process_time()
    
    x_k_1 = x0
    g_k_1 = Oracle(x_k_1)[1]
    W_k_1 = np.eye(n-md)
    D_k_1 = -np.dot(W_k_1,g_k_1)
    compteur =0

    # Boucle sur les iterations

    for k in range(iter_max):

        # Direction de descente
        
        # etape k
        alpha = Wolfe(1, x_k_1, D_k_1, Oracle)[0]
        x_k = x_k_1 + alpha*D_k_1
        
        # Valeur du critere et du gradient
        #critere, gradient = Oracle(x)
        critere, g_k = Oracle(x_k)[0],Oracle(x_k)[1]

        # Test de convergence
        gradient_norm = np.linalg.norm(g_k)
        if gradient_norm <= epsilon:
            break
        
        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(alpha)
        critere_list.append(critere)
        
        # - Calcul de new_W
        d_x_k = x_k - x_k_1
        d_g_k = g_k - g_k_1
        
        # - + Reshape les veCteur array pour les manipuler comme de vrai colonnes
        d_x_k = d_x_k.reshape((-1,1))
        d_g_k = d_g_k.reshape((-1,1))
        # - + Calcul
        coef_gauche = np.eye(n-md) - (np.dot(d_x_k,np.transpose(d_g_k)))/(np.vdot(d_g_k,d_x_k))
        coef_droite = np.eye(n-md) - (np.dot(d_g_k,np.transpose(d_x_k)))/(np.vdot(d_g_k,d_x_k))
        coef_isole = (np.dot(d_x_k,np.transpose(d_x_k)))/(np.vdot(d_g_k,d_x_k))
        W_k = np.dot(np.dot(coef_gauche,W_k_1),coef_droite) + coef_isole
        D_k = -np.dot(W_k,g_k)
        # - Calcul de la direction
        # alpha = Wolfe(1, x_k, D_k, Oracle)[0]
        
        # Mise a jour des variables
        # x_k_plus1 = x_k + (alpha*D_k)
        
        # Iteration
        x_k_1 = x_k
        g_k_1 = g_k
        D_k_1 = D_k
        
        compteur +=1
    if visualisation == True:
        Visualg(gradient_norm_list, gradient_step_list, critere_list)
    return Oracle(x_k)[0],Oracle(x_k)[1],x_k,compteur




def Newton(Oracle, x0):
    ##### Initialisation des variables    
    
    iter_max = 100
    alpha = 1
    threshold = 0.000001
    
    gradient_norm_list = []
    gradient_step_list = []
    critere_list = []

    #time_start = process_time()
    
    x = x0
    compteur = 0

    ##### Boucle sur les iterations
    
    for k in range(iter_max):
        
        # Valeur du critere et du gradient
        critere, gradient, hessien = Oracle(x)

        # Test de convergence
        gradient_norm = norm(gradient)
        if gradient_norm <= threshold:
            break
        
        # Direction de descente
        D = - dot(inv(hessien), gradient)
        
        # Mise a jour des variables
        alpha = Wolfe(1,x,D,Oracle)[0]
        print("alpha n°",k,": ",alpha)
        x = x + (alpha*D)
        
        # Evolution du gradient, du pas, et du critere
        gradient_norm_list.append(gradient_norm)
        gradient_step_list.append(alpha)
        critere_list.append(critere)
        compteur +=1
    if visualisation == True:
        Visualg(gradient_norm_list, gradient_step_list, critere_list)
    return Oracle(x)[0],Oracle(x)[1],x,compteur




# 
#### Code Cléments
# def Polak_Ribiere(Oracle,x0):
# 
# 
#     epsilon = 0.0001
#     k = 1
#     g_k_1 = Oracle(x0,3) # gradient à l'étape k-1
#     g_k = Oracle(x0,3) # gradient à l'étape k
#     d_k = -g_k # direction de descente à l'étape k
#     
#     beta = 1
#     alpha = 1
#     
#     x = x0 + d_k
#     
#     while np.linalg.norm(g_k) >= epsilon and k < 1000 :
#         
#         # gradient à l'etape k
#         g_k = Oracle(x,3)
#         
#         # Alpha démarre toujours à 1
#         alpha = Wolfe(1, x ,  -g_k  ,Oracle)[0]
#         
#         # Alpha demarre à l'ancien alpha    
#         # alpha = Wolfe(alpha, x ,  -g_k  ,Oracle)[0]
#         beta = np.dot(g_k, g_k - g_k_1)/np.linalg.norm(g_k_1)**2
#         
#                 
#         d_k = -g_k + beta*d_k
#         
#         x += beta*d_k # x_k+1
#     
#         g_k_1 = g_k
# 
#     return Oracle(x)[0],Oracle(x)[1],x,k
# 
# def BFGS(Oracle,x0):
#     
#     epsilon = 0.0001
#     k = 0
#     g_k_1 = Oracle(x0,3) # gradient à l'étape k-1
#     
#     
#     
#     beta = 1
#     alpha = 1
#     
#     n = d_k.shape[0]
#     W = np.eye(n)
#     
#     x_k = x0 -g_k_1
#     x_k_1 = x0
#     delta_x = x_k
#     delta_g = g_k
#     
#     g_k = Oracle(x_k,3) # gradient à l'étape k
#     
#     while np.linalg.norm(g_k) >= epsilon and k < 10 :
#         
#         g_k = Oracle(x_k,3)
#         
#         # J'arrive pas à calculer la MATRICE delta_x ^T * delta_g
#         # dot calcule toujours le produit scalaire
#         delta_x = x_k- x_k_1
#         delta_g = g_k - g_k_1
#         A_xg = W
#         A_gx = W
#         A_xx = W
#         for i in range(n):
#             for j in range(n):
#                 A_xg[i,j] = delta_x[i]*delta_g[j] /(delta_x @ delta_g)
#                 A_xg[i,j] = delta_g[i]*delta_x[j] /(delta_x @ delta_g)
#                 A_xg[i,j] = delta_x[i]*delta_x[j] /(delta_x @ delta_g)
#                 
#         W = (np.eye(n)- A_xg) @ W @ ((np.eye(n)- A_gx)) + A_xx
#         alpha = Wolfe(alpha, x ,  -g_k  ,Oracle)[0]
#         
#         x_k_1 = x_k
#         x_k -= alpha* (W @ g_k)
#         
#         k+=1
#     return Oracle(x_k)[0],Oracle(x_k)[1],x_k,k
# 
# def Newton(Oracle,x0):
# 
#     epsilon = 0.0001
#     k=0
#     x = x0
#     
#     G, H = Oracle(x0,6)
#     
#     d = np.linalg.solve(H,G)
#     alpha =1
#     
#     while np.linalg.norm(G) >= epsilon and k < 10 and np.linalg.det(H) != 0 :
#         
#         d = np.linalg.solve(H,G)
#         k+=1
#         alpha = Wolfe(alpha, x ,  d  ,Oracle)[0]
#         x += alpha*d
#         G, H = Oracle(x,6)
#         
#     return Oracle(x)[0],G,x,k


## Vieux Code Nico Bug
# def Polak_Ribiere(Oracle,x0):
#     print("BeGin Polak_Ribiere 52")
#     k = 1
#     g_k_1 = Oracle(x0,3) # gradient à l'étape k-1
#     g_k = Oracle(x0,3) # gradient à l'étape k
#     d_k = -g_k # direction de descente à l'étape k
#     
#     beta = 1
#     alpha = 1
#     
#     x_k = x0 + d_k
#     
#     while np.linalg.norm(g_k) >= epsilon and k < N_it :
#         
#         # gradient à l'etape k
#         g_k = Oracle(x_k,3)
#         #print("Normes: g_k_1, g_k : ",np.linalg.norm(g_k_1)," ; ",np.linalg.norm(g_k))
#         alpha = Wolfe(alpha, x_k ,  g_k  ,Oracle)[0]
#         beta = np.vdot(g_k, g_k - g_k_1)/(np.linalg.norm(g_k_1)**2)
#         
#                 
#         d_k = -g_k + beta*d_k
#         
#         x_k += alpha*d_k # x_k+1
# 
#         g_k_1 = g_k
#         #print("Beta ",k," = ",beta)
#         k+=1
#     if k >=N_it:
#         print("Polak_Ribiere, k saturated")
#     return Oracle(x_k)[0],Oracle(x_k)[1],x_k,k
# 


# ##------------------------------------------------------------------------------
# 
# def BFGS(OraclePG,x0): 
#     
#     # Initialisation des variables
# 
#     k = 0
#     g_k_1 = OraclePG(x0,3) # gradient à l'étape k-1
#     g_k = OraclePG(x0,3) # gradient à l'étape k
#     d_k = -g_k # direction de descente à l'étape k
#     I = np.eye(d_k.size)
#     W_k = I 
#     
#     beta = 1
#     alpha = 1
#     
#     x_k_1 = x0
#     x_k = x0 + d_k
#     
#     
#     while np.linalg.norm(g_k) >= epsilon and k < N_it :
#         alpha = Wolfe(alpha, x_k ,  g_k  ,OraclePG)[0]
#         # Delta à l'étape k
#         delta_x = x_k - x_k_1
#         delta_g = g_k - g_k_1
#         
#         # W à l'étape k :/!\  Verifier les transposes
#         W_k_1 = W_k
#         Coef_gauche = I - (dot(delta_x,delta_g)/dot(delta_g,delta_x))
#         Coef_droite = I - (dot(delta_g,delta_x)/dot(delta_g,delta_x))
#         Coef_fin = (dot(delta_x,delta_x)/dot(delta_g,delta_x))
#         W_k = dot(dot(Coef_gauche,W_k_1),Coef_droite) + Coef_fin
#         
#         d_k = -dot(W_k,g_k)
#         
#         # Update des x,g
#         x_k_1 = x_k
#         x_k = x_k + alpha*dot(d_k,g_k)
# 
#         g_k_1 = g_k
#         g_k = OraclePG(x_k)
#         
#         # Compteur 
#         k+=1
#     if k>=N_it : 
#         print("BFGS, k saturated")
#     return OraclePG(x_k)[0],OraclePG(x_k)[1],x_k,k
# 
# 
# 
# ## ---------------------------
# 
# def Newton(OraclePH,x0):
#     
#     # Initialisatoin des variables
#     print("Begin Newton")
#     k = 0
#     x_k = x0
#     g_k = OraclePH(x_k)[1]
#     h_k = OraclePH(x_k)[2]
#     
#     alpha = 1
#     
#     while np.linalg.norm(g_k) >= epsilon and k < N_it :
#         
#         # Direction de descente
#         d_k = -dot(np.linalg.inv(h_k),g_k)
#         
#         # Recherche linéaire
#         alpha = Wolfe(1, x_k ,  g_k  ,OraclePG)[0]
#         
#         # Update du x_k
#         x_k+= alpha*d_k
#         g_k = OraclePH(x_k)[1]
#         h_k = OraclePH(x_k)[2]
#         
#         
#         # Compteur 
#         k+=1
#     if k>=N_it :
#          print("Newton, k saturated")
#     return OraclePH(x_k)[0],OraclePH(x_k)[1],x_k,k
#     
#     
