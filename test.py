# -*- coding: utf-8 -*-
# Ce code est destiné au TP numérique de Chaos en M1 Physique de l'Université Grenoble Alpes.
# Basé sur le code de Vincent Rossetto (2026-04-03)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- 修复报错：注释掉不兼容的后端切换 ---
plt.switch_backend('Qt5Agg')
# plt.ion()

# PARAMÈTRES
N = 100000
c0 = 5.5
(a, b, c) = (0.25, 1, c0)
R_in = [0, 1, 0.5]


def Roessler(R, t, a, b, c):
    """Les équations de Rössler (Eq. 1)"""
    return [-R[1] - R[2],
            R[0] + a * R[1],
            b + (R[0] - c) * R[2]]


def solve_Roessler(r0, parametres, duree, npoints=N):
    t = np.linspace(0, duree, npoints)
    R = odeint(Roessler, r0, t, args=parametres)
    return t, R


def Roessler_fixed_point(parametres, ax):
    a_p = parametres[0]
    c_p = parametres[2]
    D = c_p ** 2 - 4 * a_p * parametres[1]
    xp0 = (c_p - np.sqrt(D)) / 2
    xp1 = -xp0 / a_p
    xp2 = xp0 / a_p
    ax.plot3D([xp0], [xp1], [xp2], marker='.', linestyle='none', color='red')


def trace_Roessler(r0, parametres, t0, t1, npoints=N):
    n_warm = int(t0 / t1 * npoints) + 1
    _, r = solve_Roessler(r0, parametres, t0, n_warm)
    r1 = r[-1]
    _, R = solve_Roessler(r1, parametres, t1, npoints)
    [X, Y, Z] = R.T

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot3D(X, Y, Z, 'blue')
    Roessler_fixed_point(parametres, ax)
    plt.show()


def section_carre(r0, parametres, t0, t1, npoints=N):
    """Trace la section de Poincaré dans le plan (yOz)"""
    _, r = solve_Roessler(r0, parametres, t0, int(t0 / t1 * npoints) + 1)
    _, R = solve_Roessler(r[-1], parametres, t1, npoints)
    [X, Y, Z] = R.T

    y_p, z_p = [], []
    for k in range(len(X) - 1):
        if X[k] < 0 and X[k + 1] > 0:  # Passage par X=0
            poid = -X[k] / (X[k + 1] - X[k])
            y_p.append(Y[k] + poid * (Y[k + 1] - Y[k]))
            z_p.append(Z[k] + poid * (Z[k + 1] - Z[k]))

    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(y_p, z_p, '.', color='blue', markersize=2)
    ax2.set_title(f'Section de Poincaré (X=0) pour c={parametres[2]}')
    plt.show()


def restriction(r0, parametres, t0, t1, npoints=N, mode='y'):
    """Calcule la restriction fy et la pente au point fixe"""
    _, r = solve_Roessler(r0, parametres, t0, int(t0 / t1 * npoints) + 1)
    _, R = solve_Roessler(r[-1], parametres, t1, npoints)
    X, Y, Z = R.T

    vals = []
    for k in range(len(X) - 1):
        if mode == 'y':  # Plan (yOz), X=0
            if X[k] > 0 and X[k + 1] < 0:
                p = -X[k] / (X[k + 1] - X[k])
                y_val = Y[k] + p * (Y[k + 1] - Y[k])
                if y_val > 0: vals.append(y_val)
        else:  # Plan (xOz), Y=0
            if Y[k] < 0 and Y[k + 1] > 0:
                p = -Y[k] / (Y[k + 1] - Y[k])
                vals.append(X[k] + p * (X[k + 1] - X[k]))

    v_arr = np.array(vals)
    v_norm = (v_arr - v_arr.min()) / (v_arr.max() - v_arr.min())
    vk, vk1 = v_norm[:-1], v_norm[1:]

    # CALCUL DE LA PENTE f'
    idx = np.argmin(np.abs(vk - vk1))  # Point fixe là où vk+1 = vk
    slope = (vk1[idx + 1] - vk1[idx]) / (vk[idx + 1] - vk[idx])
    print(f"Pente f'_{mode} au point fixe: {slope:.4f}")

    fig3 = plt.figure(figsize=(6, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(vk, vk1, '.', color='blue', markersize=2)
    ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax3.set_title(f'Restriction f_{mode} (c={parametres[2]})')
    plt.show()


def verifier_TCL(parametres, t_long=5000, t_tcl=500, n_sims=50):
    """Vérification du Théorème Central Limite (Section 4)"""
    print("Calcul de la mesure de SRB...")
    # Estimation de mu_SRB (Eq. 2)
    _, R_long = solve_Roessler([0.1, 0.5, 0.1], parametres, t_long, 150000)
    mu_srb = np.mean(R_long[50000:, 0])  # Observable phi(x) = x1

    dt_values = []
    for i in range(n_sims):
        r_rand = np.random.uniform(-2, 2, 3)  # Indépendance par rapport à X
        t_v, R_v = solve_Roessler(r_rand, parametres, t_tcl, 10000)
        b_phi = np.trapz(R_v[:, 0], t_v)  # Intégrale (B phi)_t
        d_t = (1 / np.sqrt(t_tcl)) * (mu_srb * t_tcl - b_phi)  # Calcul de Dt (Eq. 62)
        dt_values.append(d_t)

    plt.figure()
    plt.hist(dt_values, bins=15, density=True, color='green', alpha=0.6)
    plt.title("Distribution de D_t (Théorème Central Limite)")
    plt.show()


# --- APPELS POUR RÉPONDRE AUX QUESTIONS ---
if __name__ == "__main__":
    params = (a, b, c)

    # Q3: Section de Poincaré et Pente
    restriction(R_in, params, 200, 1000, mode='y')

    # Q3: Étude dans le plan (xOz)
    # restriction(R_in, params, 200, 1000, mode='x')

    # Q4: Théorème Central Limite
    verifier_TCL(params)