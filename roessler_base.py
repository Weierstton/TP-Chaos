# -*- coding: utf-8 -*-
# Ce code est destiné au TP numérique de Chaos en M1 Physique
# de l'Université Grenoble Alpes. 
# Il a été écrit par Vincent Rossetto (vincent.rossetto@grenoble.cnrs.fr)

## INSTRUCTIONS ##
# Pour le travail de TP demandé, ce code n'est qu'un point de départ.
# Il permet de réaliser des opérations simples mais tel quel, il ne permet
# pas de répondre aux questions. Pour répondre aux questions, il faut le
# modifier en changeant des paramètres ou en ajoutant des fonctions.
# 
# Le programme obtenu pour chaque réponse devra se baser sur celui-ci,
# ou sur les programmes des questions précédentes. À chaque 
# question, faire une sauvegarde du code utilisé, en le nommant par exemple
#   TP_chaos-1.py 
# pour le programme qui répond à la question 1.
# Faites des sauvegardes régulièrement pour éviter de perdre le travail
# effectué. 

# Des documents sur python sont disponibles aux adresses suivantes :
# Sur le site de l'UE Chaos et applications :
#    
# Un petit cours rapide d'introduction / rappels sur le langage :    
#   https://cours.univ-grenoble-alpes.fr/mod/resource/view.php?id=505241
#
# Un cours de calcul scientifique avec python, niveau L3 physique
#   https://cours.univ-grenoble-alpes.fr/mod/resource/view.php?id=505243
#
# Le site de référence en français
#   https://docs.python.org/fr/3
#
# Le code ci-dessous contient beaucoup de commentaires destinés à vous aider
# à comprendre comment il fonctionne, quelles sont les fonctions # utilisées. 
# N'écrivez pas de commentaires aussi détaillés pour votre code ; cependant 
#
############################################################################## 
#           IL EST INDISPENSABLE DE COMMENTER VOTRE CODE !                   #
##############################################################################
#
# Bon TP !
# Vincent Rossetto
# 2026-04-03

# Importation des paquets standards pour le calcul
# Numpy est renommé np, comme usuellement
import  numpy as np

# Importation de la fonction odeint de scipy qui permet d'intégrer 
# l'équation numériquement
from scipy.integrate import odeint

# Les outils d'affichage de matplotlib pour python
# Le paquet est renommé plt
import matplotlib.pyplot as plt

# Mise en activité du mode interactif
plt.ion()
# Définition de la méthode d'affichage des graphiques.
# Entrer '%matplotlib qt' dans la console si la fenêtre ne s'ouvre pas.
plt.switch_backend('Qt5Agg')
# Avec VS-Code, utiliser la ligne suivante
#plt.switch_backend('TkAgg')

# Importation de quelques widgets pour manipuler les graphiques et
# les images. Voir plus bas comment les utiliser (c'est facile !)
from matplotlib.widgets import Slider, Button

# DÉFINITION DE LA FIGURE
# On commence par ouvrir une figure à l'aide du module plt
# qui est matplotlib.pyplot
fig=plt.figure(figsize=(8,6))
# Cette fonction récupère la définition des axes de la figure.
ax=fig.add_subplot(projection='3d')
ax.set_position([0, 0.15, 1, 0.8])
# Avec une ancienne version de matplotlib, remplacer la ligne ci-dessus par
# ax=fig.gca(projection='3d')
 
# PARAMÈTRES 
# Nombre de points calculés
N = 10000
# Définition des paramètres du système dynamique
c0=5.5
# Paramètres initiaux (a,b,c) 
(a,b,c)=(0.25, 1, c0) 
# Position initiale (x,y,z)
R_in = [0, 1, 0.5]   # ATTENTION : condition initiale "mauvaise"

# La fonction qui définit le système dynamique:
# Il s'agit simplement qui à un vecteur R=(x,y,z) associe
# les dérivées de (x,y,z). 
def Roessler(R, t, a, b, c):
  """Les équations de Rössler"""
  return [-R[1] - R[2],
          R[0] + a*R[1],
          b + (R[0] -c)*R[2]]

# Une fonction auxiliaire qui donne la solution
# des équations à partir d'une condition initiale
# pour une certaine durée. 
def solve_Roessler(r0, parametres, duree, npoints=N) :
  """La solution des équations de Rössler, calculée
     pour une durée donnée à partir de la condition
     initiale r0.
     Le résultat contient un vecteur temps et un
     tableau des positions."""
  # Définition des temps
  t=np.linspace(0, duree, npoints)
  # Calcul de la solution jusqu'au temps demandé .
  # Le résultat est un tableau de valeurs R[0], ... , R[N]
  # tels que R[i] est le vecteur position au temps i.
  # R[i,j] est la coordonnée j du vecteur R[i].
  R=odeint(Roessler, r0, t, args=parametres)
  # La fonction retourne les temps et les positions.
  return t, R

# Le point fixe du système de Rössler
# Il correspond au point tel que Roessler(R,t,a,b,c)=0
def Roessler_fixed_point(parametres) :
  a=parametres[0]
  c=parametres[2]
  D = c**2-4*a*parametres[1]
  xp0 = (c - np.sqrt(D))/2
  xp1 = -xp0/a
  xp2 = xp0/a
  
  # Tracé du point fixe en rouge
  ax.plot3D([xp0], [xp1], [xp2],  marker='.', linestyle='none', color='red')

def trace_Roessler(r0, parametres, t0, t1, npoints=N) :
  """Fonction de tracé d'une trajectoire du système de Rössler
  avec les paramètres a, b et c, pour une durée t1 à partir du temps t0
  et de la position initiale R0"""
  # On avance dans le temps d'une durée t0
  # le nombre de pas est réduit proportionnellement à t0 pour éviter
  # de prendre trop de temps sur cette initialisation.
  n=int(t0/t1*npoints) + 1
  # Le signe _ signifie qu'on ne garde pas en mémoire
  # le vecteur temps (il est inutile pour tracer cette courbe)
  _, r=solve_Roessler(r0, parametres, t0, n)
  # On récupère la dernière valeur calculée qui sera la première
  # valeur tracée.
  r1=r[-1]
  # On résoud de nouveau à partir de la nouvelle origine.
  _, R=solve_Roessler(r1, parametres, t1, npoints)
  # R.T est la transposée de R, donc R.T[j,i] est la coordonnée j au temps i.
  # Ainsi R.T[0] est le vecteur des valeur de la coordonnée 0 (soit x)
  # à tous les temps i.
  [X, Y, Z] = R.T 
  # On efface la figure avant de tracer.
  ax.clear()
  # On trace la courbe calculée.
  ax.plot3D(X, Y, Z, 'blue')
  # On ajoute le point fixe sur la figure.
  Roessler_fixed_point(parametres)


def section_poincarre(r0, parametres, t0, t1, npoints=N, G='yOz'):
    """
    Calcule les points d'intersections entre la trajectoire de l'attracteur et le plan de poincarre choisis
    par l'utilisateur'
    
    Paramètres à commenter
    G : str, 'yOz', 'xOz' ou 'xOy'
        Plan de section choisi via les boutons de l'interface
    
    Notes
    -----
    Les boutons de l'interface utilisateur permettent de choisir dynamiquement
    la valeur de G, ce qui change automatiquement le plan de section visualisé.
    """
    # On avance dans le temps d'une durée t0
    # le nombre de pas est réduit proportionnellement à t0 pour éviter
    # de prendre trop de temps sur cette initialisation.
    n=int(t0/t1*npoints) + 1
    # Le signe _ signifie qu'on ne garde pas en mémoire
    # le vecteur temps (il est inutile pour tracer cette courbe)
    _, r=solve_Roessler(r0, parametres, t0, n)
    # On récupère la dernière valeur calculée qui sera la première
    # valeur tracée.
    r1=r[-1]
    # On résoud de nouveau à partir de la nouvelle origine.
    _, R=solve_Roessler(r1, parametres, t1, npoints)
    # R.T est la transposée de R, donc R.T[j,i] est la coordonnée j au temps i.
    # Ainsi R.T[0] est le vecteur des valeur de la coordonnée 0 (soit x)
    # à tous les temps i.
    [X, Y, Z] = R.T 
    
    # Dictionnaire définissant pour chaque plan quel axe est normal à la section
    # et quels sont les axes dans le plan de section.
    # Format : (indice_normal, indice_abscisse, indice_ordonnée)
    plans = {
        'yOz' : (0, 1, 2),  # Plan YOZ : axe X est normal, on trace (Y, Z)
        'xOz' : (1, 0, 2),  # Plan XOZ : axe Y est normal, on trace (X, Z)
        'xOy' : (2, 0, 1),  # Plan XOY : axe Z est normal, on trace (X, Y)
        }
    
    # Sélection des indices en fonction du plan choisi par l'utilisateur
    # G est modifié par les boutons de l'interface (yOz, xOz ou xOy)
    if G == 'yOz':
        i, j, k = plans['yOz']  # i : axe normal, j : abscisse, k : ordonnée
        A = R.T[i]  # Coordonnée normale à la section (dont on teste le signe)
        B = R.T[j]  # Coordonnée pour l'axe des abscisses du graphique
        C = R.T[k]  # Coordonnée pour l'axe des ordonnées du graphique
        
    elif G == 'xOz':
        i, j, k = plans['xOz']
        A = R.T[i]
        B = R.T[j]
        C = R.T[k]
        
    else:  # G == 'xOy'
        i, j, k = plans['xOy']
        A = R.T[i]
        B = R.T[j]
        C = R.T[k]
    
    # Le plan de section est défini par A = 0 (axe normal à la section)
    # On capture uniquement les passages de A > 0 à A < 0 (sens descendant)
    # Cette condition capture une seule branche de l'attracteur
    
    B_pointcarre = []  # Coordonnées des points sur l'axe des abscisses
    C_pointcarre = []  # Coordonnées des points sur l'axe des ordonnées
    
    # Parcours de tous les segments de la trajectoire (entre deux points consécutifs)
    for e in range(len(A)-1):
    
        # Détection d'un passage de la section A = 0 dans le sens décroissant
        # On sélectionne les segments où A passe de positif à négatif
        if A[e] > 0 and A[e+1] < 0:
            
            # Calcul du coefficient de pondération (interpolation linéaire)
            # Ce poids détermine la position exacte de l'intersection sur le segment
            # Formule : poid = |A[e]| / (|A[e]| + |A[e+1]|)
            poid = -A[e] / (A[e+1] - A[e])
            
            # Interpolation linéaire pour obtenir les coordonnées exactes
            # au point d'intersection avec le plan A = 0
            B_points = B[e] + poid * (B[e+1] - B[e])
            C_points = C[e] + poid * (C[e+1] - C[e])
            
            # Stockage des points d'intersection
            # NOTE : La restriction implicite (ex: Y>0 pour le plan yOz) vient du fait
            # que la condition de capture (descente) sélectionne une seule branche
            B_pointcarre.append(B_points)
            C_pointcarre.append(C_points)
    

    """ 
    Trace la section de Poincarre choisis par l'utilisateur.
    
    Paramètres : 
        On récupere les points calculer avec la fonction section_pointcarre
        représenter par B_pointcarre et C_pointcarre
    
    """
    # Création d'une nouvelle figure dédiée à la section de Poincaré
    fig2 = plt.figure(figsize=(6, 6))
    
    # Ajout d'un axe 2D (un seul graphique sur la figure)
    ax2 = fig2.add_subplot(111)

    # Affichage des points d'intersection
    # On utilise des points ('.') car la section de Poincaré est un ensemble discret
    ax2.plot(B_pointcarre, C_pointcarre, '.', color='blue', markersize=2)
    
    # Étiquettes des axes : on utilise la première et troisième lettre du nom du plan
    # Exemple : 'yOz' → 'y' en abscisse, 'z' en ordonnée
    ax2.set_xlabel(G[0])  # Premier caractère : 'y', 'x' ou 'x'
    ax2.set_ylabel(G[2])  # Troisième caractère : 'z', 'z' ou 'y'

    # Titre incluant le paramètre c (troisième paramètre du système de Roessler)
    ax2.set_title(f'Section de Poincaré ({G[0]}{G[2]} - plan {G}) pour c={parametres[2]}')

    # Activation de la grille pour faciliter la lecture
    ax2.grid(True)
    
    return B_pointcarre
    
def application_poincarre(B_pointcarre, G='yOz'):
    """
    Construit et trace l'application de Poincaré f(u_n) = u_{n+1}
    à partir des points d'intersection d'une section.
    
    Paramètres
    ----------
    B_pointcarre : 
        Coordonnées des points sur l'axe étudié (celui qui sert de variable)
        Pour G='yOz', B_pointcarre correspond aux coordonnées Y
        Pour G='xOz', B_pointcarre correspond aux coordonnées X
        Pour G='xOy', B_pointcarre correspond aux coordonnées X
    G : str, par défaut 'yOz'
        Plan de section utilisé (détermine l'axe d'étude)
        - 'yOz' : étude selon l'axe Y
        - 'xOz' : étude selon l'axe X
        - 'xOy' : étude selon l'axe X
    """
    
    # Conversion en tableau NumPy pour bénéficier des opérations vectorisées
    f_array = np.array(B_pointcarre)
    
    # Vérification qu'il y a assez de points
    if len(f_array) < 2:
        print(f"Erreur : Pas assez de points pour l'application de Poincaré ({len(f_array)} points)")
        return None, None
    
    # Récupération des valeurs extrêmes
    f_max = np.max(f_array)
    f_min = np.min(f_array)
    
    # Normalisation sur l'intervalle [0, 1]
    # Cette opération permet de se ramener à une échelle commune
    if f_max > f_min:
        f_norm = (f_array - f_min) / (f_max - f_min)

    # u_n : représente les états successifs (tous les points sauf le dernier)
    # On exclut le dernier point car il n'a pas de successeur connu
    u_n = f_norm[:-1]
    
    # u_{n+1} = f(u_n) : l'image par l'application de Poincaré
    # On exclut le premier point pour créer l'appariement (u_n, u_{n+1})
    u_np1 = f_norm[1:]
    
    # Création de la figure
    fig3 = plt.figure(figsize=(6, 6))
    ax3 = fig3.add_subplot(111)
    
    # Tracé des couples (u_n, u_{n+1})
    ax3.plot(u_n, u_np1, '.', color='blue', markersize=2)
    
    # Détermination du nom de l'axe étudié à partir de G
    if G == 'yOz':
        axe_nom = 'Y'
    elif G == 'xOz' or G == 'xOy':
        axe_nom = 'X'
    else:
        axe_nom = 'U'
    
    # Étiquettes des axes (correction de la syntaxe)
    ax3.set_xlabel(f'{axe_nom}_n normalisée')
    ax3.set_ylabel(f'{axe_nom}_{{n+1}} = f({axe_nom}_n) (normalisé)')
    
    # Titre du graphique
    ax3.set_title(f'Application de Poincaré (restriction implicite à {axe_nom} > 0)')
    
    # Grille pour faciliter la lecture
    ax3.grid(True)
    
    plt.show()

    
    
# FONCTIONS POUR LES WIDGETS
# La fonction quitter ne fait que fermer la fenêtre en cours d'utilisation.
# Cette fonction est nécessaire pour créer un bouton qui effectue 
# l'action de fermer la fenêtre.
def quitter(_):
    plt.close()

# Fonction de mise à jour de l'affichage, pour prendre compte la modification 
# d'un paramètre. Elle sera activée à chaque modification du paramètre c
# ou du temps t0 (voir plus bas).

def update(_):
    # On récupère la valeur du paramètre c indiqué par la barre de glissement.
    c=barre_c.val
    # On récupère la valeur du temps indiqué par la barre de glissement.
    t0=barre_t0.val
    t1=barre_t1.val
    # On récupère le nombre de pas 
    n=np.power(10., barre_N.val)
    barre_N.valtext.set_text(f"{n:5.1e}")
    # On trace la nouvelle figure
    trace_Roessler(R_in, (a,b,c), t0, t1, int(n))

# Fonction de remise à zéro des glisseurs.
# Une fois cela fait, on réactualise l'affichage
def reset(_):
    barre_c.reset()
    barre_t0.reset()
    barre_t1.reset()
    barre_N.reset()
    update(0)

# Fonction qui ajoute 0.01 à c
def plus(_): 
    # On définis une nouvelle valeur de la barre c limité à sa valeur maximal
    plus_val = min(barre_c.val + 0.01, barre_c.valmax)
    # On incrémente cette nouvelle valeur à la barre
    barre_c.set_val(plus_val)
    
# Fonction qui retire 0.01 à c
def moins(_): 
    # On définis une nouvelle valeur de la barre c limité à sa valeur maximal
    plus_val = min(barre_c.val - 0.01, barre_c.valmax)
    # On incrémente cette nouvelle valeur à la barre
    barre_c.set_val(plus_val)

# TRACÉ DES WIDGETS
# Dessin de la barre de glissement pour le paramètre c 
# Les nombres entrés sont les coordonnées du rectangle contenant la barre.
axe_c = plt.axes([0.1, 0.07, 0.65, 0.04])
# On crée ensuite un widget de type slider avec le rectangle défini axe_c.
# Les valeurs de la barre vont de 1 (à gauche) à 15 (à droite).
# Le nom indiqué à gauche de la barre est 'c' et sa valeur initiale est c0.
barre_c= Slider(ax=axe_c, label='c', valmin=0.01, valmax=15, valinit=c0, track_color='darkgreen')

# Dessin de la barre de glissement pour les paramètres de temps (t0 et t1)
axe_t0 = plt.axes([0.1, 0.01, 0.25, 0.03])
axe_t1 = plt.axes([0.45, 0.01, 0.30, 0.03])
# Widget de type slider pour les temps (la valeur initiale est t0=0).
barre_t0= Slider(ax=axe_t0, label='t0', valmin=0, valmax=10000, valinit=0)
barre_t1= Slider(ax=axe_t1, label='t1', valmin=0, valmax=10000, valinit=100)

# De même avec le paramètre N (il est en échelle logarithmique de base 10)
axe_N = plt.axes([0.1, 0.04, 0.65, 0.03])
barre_N= Slider(ax=axe_N, label='N', valmin=2, valmax=6, valinit=2)  # Attention à ne pas choisir N trop grand !

# Dessin d'un rectangle qui sera la bouton action.
cadre_raz=plt.axes([0.85, 0.05, 0.1, 0.03])
# Widget de type bouton 
bouton_raz=Button(cadre_raz,'R. à 0')

# Dessin d'un rectangle qui sera la bouton Fin
cadre_fin = plt.axes([0.85, 0.01, 0.1, 0.03])
# Widget de type bouton 
bouton_fin=Button(cadre_fin,'Fin')

# Dessin d'un rectangle qui sera la bouton +.
cadre_plus=plt.axes([0.85, 0.09, 0.1, 0.03])
# Widget de type bouton 
bouton_plus=Button(cadre_plus,'+')

# Dessin d'un rectangle qui sera la bouton +.
cadre_moins=plt.axes([0.85, 0.13, 0.1, 0.03])
# Widget de type bouton 
bouton_moins=Button(cadre_moins,'-')


# Les widgets sont créés, mais il faut maintenant associer 
# ce qu'il se passe quand on les utilise.

# ACTIVATION DES WIDGETS 
# Lorsque l'on change la valeur de la barre (on_changed) l'action update
# est exécutée.
barre_c.on_changed(update)
# Même chose pour la barre de t0 et de N
barre_t0.on_changed(update)
barre_t1.on_changed(update)
barre_N.on_changed(update)
# Lorsque l'on clique sur remise à zéro, on réinitialise
bouton_raz.on_clicked(reset)
# Lorsque l'on clique sur Fin, on ferme la fenêtre
bouton_fin.on_clicked(quitter)
# Lorsque l'on clique sur +, on ajoute 0.01 à c
bouton_plus.on_clicked(plus)
# Lorsque l'on clique sur -, on retire 0.01 à c
bouton_moins.on_clicked(moins)

# Initialisation du programme
# On fait une mise à jour avec update et un paramètre (n'importe lequel).
# Cela permet de mettre les valeurs des paramètres a,b,c,t0 à jour,
# en accord avec les valeurs indiquées sur les glisseurs.
update(0)

# AFFICHAGE DE LA FENÊTRE
plt.show(block=True)
# Le programme "tourne" maintenant dans la fenêtre. Il attend
# que les widgets soient utilisés pour effectuer les actions
# demandées. Tant que la fenêtre est ouverte, le programme continue 
# d'attendre et exécuter les opérations controlées par les widgets.
# Si on ferme la fenêtre, comme il n'y a plus rien en dessous de ce code,
# python a réalisé toutes les opérations demandées et termine le programme.

#CREATION DES FONCTIONS ET BOUTONS UTILE A LA FONCTION SECTION_POINCARRE
# Ces fonctions sont appelées automatiquement lorsque l'utilisateur clique
# sur un bouton. Chaque fonction appelle section_poincarre() avec un plan
# de section différent.

def xOy(_): 
    """
    Fonction de rappel pour le bouton 'xOy'.
    Trace la section de Poincaré dans le plan XOY (axe Z normal à la section) et 
    l'application de Poincaré (réstriction sur X)
    """
    X = section_poincarre(R_in, (a, b, c), 100, 500, 50000, 'xOy')
    application_poincarre(X, G = 'xOy')  


def xOz(_): 
    """
    Fonction de rappel pour le bouton 'xOz'.
    Trace la section de Poincaré dans le plan XOZ (axe Y normal à la section) et 
    l'application de Poincaré (réstriction sur X)
    """
    X = section_poincarre(R_in, (a, b, c), 100, 500, 50000, 'xOz')
    application_poincarre(X, G = 'xOz')


def yOz(_): 
    """
    Fonction de rappel pour le bouton 'yOz'.
    Trace la section de Poincaré dans le plan YOZ (axe X normal à la section) et 
    l'application de Poincaré (réstriction sur Y)
    """
    Y = section_poincarre(R_in, (a, b, c), 100, 500, 50000, 'yOz')
    application_poincarre(Y, G = 'yOz')



# CRÉATION DES ZONES D'ANCRAGE (AXES) POUR LES BOUTONS
# Chaque bouton a besoin d'un cadre (axes) pour se positionner dans la figure.
# Les coordonnées sont données au format : [x, y, largeur, hauteur] en proportion
# de la figure (0 = bas/gauche, 1 = haut/droite).

# Cadre pour le bouton xOy (plan XOY)
# Position : à gauche, légèrement au-dessus du bord inférieur
cadre_xOy = plt.axes([0.15, 0.08, 0.15, 0.06])

# Cadre pour le bouton xOz (plan XOZ)
# Position : au centre, légèrement au-dessus du bord inférieur
cadre_xOz = plt.axes([0.425, 0.08, 0.15, 0.06])

# Cadre pour le bouton yOz (plan YOZ)
# Position : à droite, légèrement au-dessus du bord inférieur
cadre_yOz = plt.axes([0.70, 0.08, 0.15, 0.06])


# INSTANCIATION DES BOUTONS
# Création des objets Button à partir des cadres définis ci-dessus

bouton_xOy = Button(cadre_xOy, 'xOy')  # Bouton pour la section XOY
bouton_xOz = Button(cadre_xOz, 'xOz')  # Bouton pour la section XOZ
bouton_yOz = Button(cadre_yOz, 'yOz')  # Bouton pour la section YOZ

# CONNEXION DES BOUTONS À LEURS FONCTIONS DE RAPPEL
# La méthode on_clicked() associe un bouton à sa fonction callback.
# Lorsque l'utilisateur clique sur le bouton, la fonction est exécutée.

bouton_xOy.on_clicked(xOy)  # Clique sur 'xOy' → trace section XOY
bouton_xOz.on_clicked(xOz)  # Clique sur 'xOz' → trace section XOZ
bouton_yOz.on_clicked(yOz)  # Clique sur 'yOz' → trace section YOZ



plt.show(block=True)

plt.show(block=True)
# c_2=3.83 premiere dedoublement,
# c_4=4.57 duexieme dedoubelemt,
# c_5=4.81 chaotique,
# c_6=5.57 cycle limite de 3 tours,
# c_7=6.38 cycle limits de 4 tours,
# c_8=7.19 cycle limits de 5 tours,
# c_8=8.83 chaotique,
# c_9=10 cycles limites

