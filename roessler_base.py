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
  
def section_carre(r0, parametres, t0, t1, npoints=N) :
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
    
    # ============================================================================
    # COLLECTE DES POINTS DE LA SECTION DE POINCARÉ (X = 0)
    # ============================================================================
    y_pointcarre = []
    z_pointcarre = []
    
    # Parcours de tous les segments de la trajectoire (entre deux points consécutifs)
    for k in range(len(X)-1):
    
        # Détection d'un passage de la section X = 0 dans le sens décroissant
        # On sélectionne les segments où X passe de positif (avant section) à négatif (après section)
        # Cette condition de sens est cruciale : elle ne capture qu'UNE SEULE branche de l'attracteur
        if X[k] > 0 and X[k+1] < 0:
            
            # Calcul du coefficient de pondération (interpolation linéaire)
            # Ce poids détermine la position exacte de l'intersection sur le segment
            poid = -X[k] / (X[k+1] - X[k])
            
            # Interpolation pour obtenir les coordonnées Y et Z au point d'intersection
            y_points = Y[k] + poid * (Y[k+1] - Y[k])
            z_points = Z[k] + poid * (Z[k+1] - Z[k])
            
            # Stockage des points d'intersection
            # NOTE IMPORTANTE : Aucun filtre explicite sur Y > 0 n'est nécessaire car :
            # - La condition de capture (X > 0 → X < 0) sélectionne naturellement une branche spécifique de l'attracteur
            # - Sur cette branche, du fait de la dynamique du système (ex: attracteur de Lorenz), la coordonnée Y est TOUJOURS positive
            # - Donc tous les points collectés vérifient implicitement Y > 0
            y_pointcarre.append(y_points)
            z_pointcarre.append(z_points)

            max_idx = np.argmax(y_pointcarre)

            critical_y = y_pointcarre[max_idx]

            critical_z = z_pointcarre[max_idx]

    # Création d'une nouvelle figure dédiée à la section de Poincaré (fenêtre graphique de 6x6 pouces)
    fig2 = plt.figure(figsize=(6, 6))
    
    # Ajout d'un axe 2D à la figure (subplot 1x1, premier et unique graphique)
    # Le paramètre '111' signifie : 1 ligne, 1 colonne, 1er sous-graphique
    ax2 = fig2.add_subplot(111)  # Axe 2D pour représenter le plan (Y, Z)

    # Affichage des points d'intersection avec la section X=0
    # On utilise des points ('.') et non des lignes car la section de Poincaré est un ensemble discret de points
    # L'option 'color' définit la couleur des points (bleu)
    # L'option 'markersize' contrôle la taille des points (2 pixels)
    ax2.plot(y_pointcarre, z_pointcarre, '.', color='blue', markersize=2)
    
    # Ajout du label pour l'axe des abscisses (Y)
    ax2.set_xlabel('Y')

    # Ajout du label pour l'axe des ordonnées (Z)
    ax2.set_ylabel('Z')

    # Ajout d'un titre personnalisé qui inclut la valeur du paramètre c (troisième paramètre de la liste 'parametres')
    # La chaîne 'f' permet d'insérer directement la valeur de parametres[2] dans le texte
    ax2.set_title(f'Section de Poincaré (X=0) pour c={parametres[2]}')

    # Activation de la grille de fond pour faciliter la lecture des coordonnées
    ax2.grid(True)

    plt.scatter(critical_y, critical_z,

                s=50,
                marker='.',
                edgecolors='black',
                linewidth=1,
                zorder=10,
                label='x_c')
    plt.legend(loc='best', fontsize=10)

    # Affichage de la figure à l'écran (attention : cette méthode peut être obsolète selon la version de matplotlib)
    # Dans les versions récentes, on préférera plt.show() ou fig2.canvas.draw()
    fig2.show()
    
    # Conversion en tableau NumPy pour bénéficier des opérations vectorisées
    y_array = np.array(y_pointcarre)
                
    # Récupération des valeurs extrêmes (tous les Y sont > 0 par construction)
    y_max = np.max(y_array)
    y_min = np.min(y_array)

    # Normalisation sur l'intervalle [0, 1]
    # Cette opération est valide car y_max > y_min (points non tous identiques)
    if y_max > y_min:
        y_norm = (y_array - y_min) / (y_max - y_min)

    # y_k représente les états successifs : y_n
    # On exclut le dernier point car il n'a pas de successeur
    yk = y_norm[:-1]
    
    # y_{k+1} représente l'image par f : f(y_n) = y_{n+1}
    # On exclut le premier point pour créer l'appariement (y_n, y_{n+1})
    yk1 = y_norm[1:]

    fig3 = plt.figure(figsize=(6, 6))
    ax3 = fig3.add_subplot(111)

    # Tracé des couples (y_n, y_{n+1}) pour visualiser la dynamique
    # La restriction est implicitement sur Y > 0 grâce à la condition de capture initiale
    ax3.plot(yk, yk1, '.', color='blue', markersize=2)

    # Étiquettes des axes
    ax3.set_xlabel('y_n (Y normalisé)')
    ax3.set_ylabel('y_{n+1} = f(y_n) (Y normalisé)')

    # Titre indiquant la restriction implicite à Y > 0
    ax3.set_title('Application du retour de Poincaré (restriction implicite à Y > 0)')

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
 
section_carre(R_in, (a,b,c), 100, 500, 50000)
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

