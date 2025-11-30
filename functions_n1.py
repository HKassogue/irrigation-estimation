# %% [markdown]
# ## Méthode de tir pour résoudre le système d’optimalité (Eq. 10), en dimension 1
# 
# **But.** Pour $A\in\mathbb{R}$, $G\in\mathbb{R}$, $C\in\mathbb{R}$, et $q_1,q_2>0$ des pondérations, construire le couple optimal
# \begin{aligned}
# h^\star(t) &= -\tfrac{1}{q_2} G_2^\top p(t),\\
# z(0) &= -\tfrac{1}{q_1} p(0)
# \end{aligned}
# où $(z,p)$ est solution du système d’optimalité donné par le TPBVP:
# \begin{aligned}
# z'(t) &= A z(t) + G h^\star(t) = A z(t) - \tfrac{1}{q_2} G G^\top p(t), \quad z(0) = -\tfrac{1}{q_1} s,\\
# p'(t) &= -A^\top p(t) - C^\top\!\big(C z(t) - y^{\mathrm m}(t)\big), \quad p(T)=0,
# \end{aligned}
# et $s := p(0)\in\mathbb{R}^n$ est **l’inconnue de tir**. 
# 
# ---
# 
# ### Principe de résolution par tir exact (pseudo-code)
# 1. Give input data 
#     - System: A, G2, C
#     - Objective: z0, h as known
#     - Time discretization: T, N
#     - Regulations: q1, q2
# 2. Forward z_true and y_m from known z0 and h
#     - Integrate $z'(t) = A z(t) + G2 h(t)$ using RK4 over grid t
#     - Then compute $Y(t) = C z(t)$
# 3. Backward p and z from y_m with shooting method on p(0)
#     - Define a function that integrate $(z'(t), p'(t))$ with $z(0) = -s/q1$, $p(T) = 0$ under the shoot $s=p(0)$.
#     - Define the shooting function $F(s) := p(T; s)$ and find the good $s^\star$ such that $F(s^\star)=0$ by multiple integrations with $s$.
#     - Integrate final (z, p) with the good $s^\star$.
# 4. Reconstruction of the irrigation h
#     - $h(t) = -(1/q2) * G2^T p(t)$
#     - $z(0) = -(1/q1) * p(0)$
# 5. Make plots and metrics
#     - Metrics: 
#         - $z_0 - z_0estimated$
#         - $RMSE_z = \sqrt{mean((z_{true} - z_{estimated})^2)}$
#         - $RMSE_h = \sqrt{mean((z_{true} - z_{estimated})^2)}$
#         - $Y Misfit = \int_0^T \|Cz(t)-y^m(t)\|^2 dt$
#         - $H Energy = \int_0^T \|h(t)\|^2 dt$
#     - Plots:
#         - $z_{true}$ vs $z_{estimated}$
#         - $h_{true}$ vs $h_{estimated}$
#         - $z_{estimated}$ as function of q2
#         - $h_{estimated}$ as function of q2
#         - $RMSE_z$ as function of q2
#         - $RMSE_h$ as function of q2
#         - $Y Misfit$ as function of q2
#         - $H Energy$ as function of q2
# 
# ---
# 
# ### Input example
# - System: A=2, G2=3, C=4.9
# - Objective: z0=4, h(t)=t(1 - t)
# - Time discretization: T=1, N=200
# - Regulations: q1=1, q2=1
# 
# ---
# 
# ### Ouput example (metrics)
# z0 true = 4, z0 estimated = 2.617, error = 1.383<br>
# RMSE(z) = 10.881<br>
# RMSE(h) = 1.430<br>
# Y Misfit = 1.466<br>
# h Energy = 2.108


# %%
# Modules importation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D surface
import pandas as pd
import os


# %%
# Various functions definition
def forward_z_from_h(A, G2, z0, h, N, dt):
    """Integrate z'(t) = A z(t) + G2(t) h(t) from known z0 and h using RK4 over grid t
    Then compute Y(t) = C z(t)"""
    z = np.zeros(N+1); z[0] = z0
    for k in range(N):
        zk = z[k]; hk, hk_1 = h[k], h[k+1]
        h_mid = (hk + hk_1) / 2.0
        fz = lambda zz, hh : A*zz + G2*hh
        k1 = fz(zk,                hk)
        k2 = fz(zk+dt*k1/2.0,      h_mid)
        k3 = fz(zk+dt*k2/2.0,      h_mid)
        k4 = fz(zk+dt*k3,          hk_1)
        z[k+1] = zk + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return z


def backward_z_p_by_shooting(A, G2, C, N, dt, y_meas, q1, q2, s):
    """Integrate from y_m by shooting on p(0)
        z'(t) = A z(t) - (1/q2) G2 G2^T p(t)
        p'(t) = -A^T p(t) - C^T (C z(t) - y^m(t))
        z(0) = -s/q1, p(T) = 0 with s=p(0)
        s : vector (m,)
        y_meas : numpy array (N+1, m) containing y^m(t_k)
    """
    z = - (1.0 / q1) * s.copy()
    p = s.copy()

    traj_z = np.zeros((N+1, 1))
    traj_p = np.zeros((N+1, 1))
    traj_z[0] = z
    traj_p[0] = p

    G2G2T = G2 * G2 # G2T=G2 in n=1
    AT = A          # AT=A in n=1
    CT = C          # CT=C in n=1

    for k in range(N):
        yk = y_meas[k]  # y^m(t_k)

        # z' = A z - (1/q2) G2 G2^T p
        zdot = A * z - (1.0 / q2) * (G2G2T * p)

        # p' = -A^T p - C^T (C z - y^m)
        Cz_minus_y = C * z - yk
        pdot = - AT * p - CT * Cz_minus_y

        # Euler
        z = z + dt * zdot
        p = p + dt * pdot

        traj_z[k+1] = z
        traj_p[k+1] = p

    pT = p  # p at final time, i.e. p(T; s)
    return pT, traj_z, traj_p


def F_of_s(A, G2, C, N, dt, y_meas, q1, q2, s):
    """Return F(s) = P(T; s)"""
    pT, _, _ = backward_z_p_by_shooting(A, G2, C, N, dt, y_meas, q1, q2, s)
    return pT


def find_s_star(A, G2, C, N, dt, y_meas, q1, q2, s0=0.0, s1=1.0, tol=1e-6, maxit=100):
    """Resolve F(s)=P(T;s)=0 by secant method.
    s0, s1 : initial points for s"""
    F0 = F_of_s(A, G2, C, N, dt, y_meas, q1, q2, np.array([s0]))[0]
    F1 = F_of_s(A, G2, C, N, dt, y_meas, q1, q2, np.array([s1]))[0]

    for _ in range(maxit):
        if abs(F1 - F0) < 1e-12:
            break
        s2 = s1 - F1 * (s1 - s0) / (F1 - F0)
        F2 = F_of_s(A, G2, C, N, dt, y_meas, q1, q2, np.array([s2]))[0]
        if abs(F2) < tol:
            return s2
        s0, F0 = s1, F1
        s1, F1 = s2, F2

    return s1  # approx


def find_z_p(A, G2, C, N, dt, y_meas, q1, q2):
    """Finding final (z,p) from shooting on p(0)"""
    s_star = find_s_star(A, G2, C, N, dt, y_meas, q1, q2)
    return backward_z_p_by_shooting(A, G2, C, N, dt, y_meas, q1, q2, s_star)


def reconstruction(G2, traj_p, N, q1, q2):
    """Return (z0*, h*) from (., p), solution of the TPBVP:
        h(t) = -(1/q2) * G2^T p(t)
        z(0) = -(1/q1) * p(0) already in traj_z
    """
    z0 = -traj_p[0][0] / q1 # already in traj_z

    traj_h = np.zeros((N+1, 1))
    for k in range(N+1):
        traj_h[k] = - (1.0 / q2) * (G2 * traj_p[k])  # G2T=G2 in n=1
    
    return z0, traj_h


def simulate_n1(A, G2, C, z0, h, T=1, N=200, q1=1, q2=1):
    """Simulate the irrigation water command estimation for n=1
    - Input data are like:
      - System: A, G2, C
      - Objective: z0, h as known
      - Time discretization: T, N
      - Regulations: q1, q2
    - Simulation steps are like:
      - Forward z_true and y_m from known z0 and h
        - Integrate z'(t) = A z(t) + G2 h(t) using RK4 over grid t
        - Then compute Y(t) = C z(t)
      - Backward p and z from y_m with shooting method on p(0)
        - Define a function that integrate (z'(t), p'(t)) with z(0)=-s/q1, p(T)=0 under the shoot s=p(0)
        - Define the shooting function F(s) := p(T; s) and find the good s^* such that F(s^*)=0 by multiple integrations with s
        - Integrate final (z, p) with the good s^*
      - Reconstruction of the irrigation h
        - h(t) = -(1/q2) * G2^T p(t)
        - z(0) = -(1/q1) * p(0)
     - Outputs are:
        z0_true, h_true, z_true, y_m, p, z_est, h_est, z0_est"""

    # Step 1 : Data input (completion)
    dt = T / N
    t  = np.linspace(0, T, N+1)
    h_true = h(t)

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = C * z_true

    # Step 3: Backward z and p from y_m with shooting method on p(0)
    _, traj_z, traj_p = find_z_p(A, G2, C, N, dt, y_m, q1, q2)
    z_est = traj_z

    # Step 4: Reconstruction of the irrigation h
    z0_est, h_est = reconstruction(G2, traj_p, N, q1, q2)

    # Outputs
    return h_true, z_true, y_m, z0_est, z_est, h_est


def compute_metrics(C, z0_true, h_true, z_true, y_m, z0_est, z_est, h_est, t):
    """Compute the metrics"""
    z0_error = np.abs(z0_est - z0_true)
    z0_energy = np.linalg.norm(z0_est)
    z_rmse = np.sqrt(np.mean((z_est - z_true)**2))
    h_rmse = np.sqrt(np.mean((h_est - h_true)**2))
    misfit = np.trapz((C*z_est[:, 0] - y_m)**2, t)  #because z_est is (N+1, 1)
    energy = np.trapz(h_est[:, 0]**2, t)            #because h_est is (N+1, 1)
    return z0_error, z0_energy, z_rmse, h_rmse, misfit, energy


def plots( h_true, z_true, y_m, z0_est, z_est, h_est, t, save=False, outdir = "figs"):
    """Plots the outputs, and/or save them optionnaly in a created directory if it does not exist."""
    if not save:
        plt.figure(figsize=(8, 5)); plt.plot(t, h_true, label="h true"); plt.plot(t, h_est, "--", label="h estimated"); plt.legend(); plt.xlabel("t"); plt.ylabel("h(t)"); plt.grid(alpha=0.7, linewidth=0.5); plt.tight_layout()
        plt.figure(figsize=(8, 5)); plt.plot(t, z_true, label="z true"); plt.plot(t, z_est, "--", label="z estimated"); plt.legend(); plt.xlabel("t"); plt.ylabel("z(t)"); plt.grid(alpha=0.7, linewidth=0.5); plt.tight_layout()
        plt.show()
    else:
        outdir = 'outputs/' + outdir
        os.makedirs(outdir, exist_ok=True)
        plt.figure(figsize=(8, 5)); plt.plot(t, h_true, label="h true"); plt.plot(t, h_est, "--", label="h estimated"); plt.legend(); plt.xlabel("t"); plt.ylabel("h(t)"); plt.grid(alpha=0.7, linewidth=0.5); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_h_true_vs_h_estimated.png"), dpi=200); plt.close()
        plt.figure(figsize=(8, 5)); plt.plot(t, z_true, label="z true"); plt.plot(t, z_est, "--", label="z estimated"); plt.legend(); plt.xlabel("t"); plt.ylabel("z(t)"); plt.grid(alpha=0.7, linewidth=0.5); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z_true_vs_z_estimated.png"), dpi=200); plt.close()
        print("\nAll done. Outputs figures are saved under:", outdir)


def q2_sensibility(A, G2, C, z0, h, T=1, N=200, q1=1, q2_list=[0.25, 0.5, 1], save=False, outdir = "figs"):
    """Sensibility of q2: sweep q2, compute solutions & metrics"""
    # Step 1 : Data input (completion)
    dt = T / N
    t  = np.linspace(0, T, N+1)
    h_true = h(t)

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = C * z_true
    
    curves_z, curves_h, metrics = [], [], []
    for q2 in q2_list:
        # Step 3: Backward z and p from y_m with shooting method on p(0)
        _, traj_z, traj_p = find_z_p(A, G2, C, N, dt, y_m, q1, q2)
        z_est = traj_z

        # Step 4: Reconstruction of the irrigation h
        z0_est, h_est = reconstruction(G2, traj_p, N, q1, q2) 
        curves_z.append(z_est)
        curves_h.append(h_est)

        # Metrics
        z0_error, z0_energy, z_rmse, h_rmse, misfit, energy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
        metrics.append({
            "q2": q2,
            "z0_error": z0_error,
            "z0_energy": z0_energy,
            "z_rmse": z_rmse,
            "h_rmse": h_rmse,
            "misfit": misfit,
            "energy": energy,
        })
    metrics = pd.DataFrame(metrics).sort_values("q2").reset_index(drop=True)

    # Figures (overlays & curves)
    if not save:
        plt.figure(figsize=(8, 5))
        plt.plot(t, h_true, "k-", label=f"h true")
        for i, q2 in enumerate(q2_list):
            plt.plot(t, curves_h[i], "--", label=f"q2={q2}")
        plt.xlabel("t"); plt.ylabel("h(t)")
        # plt.title("Sensibility in q2 of h")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(t, z_true, "k-", label=f"z true")
        for i, q2 in enumerate(q2_list):
            plt.plot(t, curves_z[i], "--", label=f"q2={q2}")
        plt.xlabel("t"); plt.ylabel("z(t)")
        # plt.title("Sensibility in q2 of z")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["z0_error"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["z0_error"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$z_{0Energy}=|z_0 - z_0^\star|$")
        # plt.title(r"Error of $z_0$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["z0_energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["z0_energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$|z_0^\star|$")
        # plt.title(r"Energy of $z_0$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["z_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["z_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$RMSE_z$")
        # plt.title(r"$RMSE_z$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["h_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["h_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$RMSE_h$")
        # plt.title(r"$RMSE_h$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["misfit"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["misfit"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        # plt.title("Y Misfit as function of q2")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("H energy as function of q2")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["misfit"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["misfit"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("Compromis YMisfit/HEnergy")
        plt.tight_layout()
        plt.show()

    else:
        
        outdir = 'outputs/' + outdir
        os.makedirs(outdir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(t, h_true, "k-", label=f"h true")
        for i, q2 in enumerate(q2_list):
            plt.plot(t, curves_h[i], "--", label=f"q2={q2}")
        plt.xlabel("t"); plt.ylabel("h(t)")
        # plt.title("Sensibility in q2 of h")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_h_q2_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(t, z_true, "k-", label=f"z true")
        for i, q2 in enumerate(q2_list):
            plt.plot(t, curves_z[i], "--", label=f"q2={q2}")
        plt.xlabel("t"); plt.ylabel("z(t)")
        # plt.title("Sensibility in q2 of z")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z_q2_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["z0_error"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["z0_error"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$z_{0Energy}=|z_0 - z_0^\star|$")
        # plt.title(r"Error of $z_0$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z0_error.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["z0_energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["z0_energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$|z_0^\star|$")
        # plt.title(r"Energy of $z_0$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z0_energy.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["z_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["z_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$RMSE_z$")
        # plt.title(r"$RMSE_z$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z_rmse.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["h_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["h_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$RMSE_h$")
        # plt.title(r"$RMSE_h$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_h_rmse.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["misfit"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["misfit"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        # plt.title("Y Misfit as function of q2")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_misfit_q2_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q2"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["q2"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q2"); plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("H energy as function of q2")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_energy_q2_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["misfit"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row["misfit"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("Compromis YMisfit/HEnergy")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_misfit_energy_q2_sensibility.png"), dpi=200); plt.close()

        print("\nAll done. Outputs figures are saved under:", outdir)

    return curves_z, curves_h, metrics


def q1_sensibility(A, G2, C, z0, h, T=1, N=200, q1_list=[0.25, 0.5, 1], q2=1, save=False, outdir = "figs"):
    """Sensibility of q1: sweep q1, compute solutions & metrics"""
    # Step 1 : Data input (completion)
    dt = T / N
    t  = np.linspace(0, T, N+1)
    h_true = h(t)

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = C * z_true
    
    curves_z, curves_h, metrics = [], [], []
    for q1 in q1_list:
        # Step 3: Backward z and p from y_m with shooting method on p(0)
        _, traj_z, traj_p = find_z_p(A, G2, C, N, dt, y_m, q1, q2)
        z_est = traj_z

        # Step 4: Reconstruction of the irrigation h
        z0_est, h_est = reconstruction(G2, traj_p, N, q1, q2) 
        curves_z.append(z_est)
        curves_h.append(h_est)

        # Metrics
        z0_error, z0_energy, z_rmse, h_rmse, misfit, energy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
        metrics.append({
            "q1": q1,
            "z0_error": z0_error,
            "z0_energy": z0_energy,
            "z_rmse": z_rmse,
            "h_rmse": h_rmse,
            "misfit": misfit,
            "energy": energy,
        })
    metrics = pd.DataFrame(metrics).sort_values("q1").reset_index(drop=True)

    # Figures (overlays & curves)
    if not save:
        plt.figure(figsize=(8, 5))
        plt.plot(t, h_true, "k-", label=f"h true")
        for i, q1 in enumerate(q1_list):
            plt.plot(t, curves_h[i], "--", label=f"q1={q1}")
        plt.xlabel("t"); plt.ylabel("h(t)")
        # plt.title("Sensibility in q1 of h")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(t, z_true, "k-", label=f"z true")
        for i, q1 in enumerate(q1_list):
            plt.plot(t, curves_z[i], "--", label=f"q1={q1}")
        plt.xlabel("t"); plt.ylabel("z(t)")
        # plt.title("Sensibility in q1 of z")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["z0_error"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["z0_error"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$z_{0Energy}=|z_0 - z_0^\star|$")
        # plt.title(r"Error of $z_0$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["z0_energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["z0_energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$|z_0^\star|$")
        # plt.title(r"Energy of $z_0$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["z_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["z_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$RMSE_z$")
        # plt.title(r"$RMSE_z$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["h_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["h_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$RMSE_h$")
        # plt.title(r"$RMSE_h$")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["misfit"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["misfit"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        # plt.title("Y Misfit as function of q1")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("H energy as function of q1")
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["misfit"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["misfit"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("Compromis YMisfit/HEnergy")
        plt.tight_layout()
        plt.show()

    else:
        
        outdir = 'outputs/' + outdir
        os.makedirs(outdir, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(t, h_true, "k-", label=f"h true")
        for i, q1 in enumerate(q1_list):
            plt.plot(t, curves_h[i], "--", label=f"q1={q1}")
        plt.xlabel("t"); plt.ylabel("h(t)")
        # plt.title("Sensibility in q1 of h")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_h_q1_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(t, z_true, "k-", label=f"z true")
        for i, q1 in enumerate(q1_list):
            plt.plot(t, curves_z[i], "--", label=f"q1={q1}")
        plt.xlabel("t"); plt.ylabel("z(t)")
        # plt.title("Sensibility in q1 of z")
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z_q1_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["z0_error"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["z0_error"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$z_{0Energy}=|z_0 - z_0^\star|$")
        # plt.title(r"Error of $z_0$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z0_error.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["z0_energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["z0_energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$|z_0^\star|$")
        # plt.title(r"Energy of $z_0$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z0_energy.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["z_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["z_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$RMSE_z$")
        # plt.title(r"$RMSE_z$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_z_rmse.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["h_rmse"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["h_rmse"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$RMSE_h$")
        # plt.title(r"$RMSE_h$")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_h_rmse.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["misfit"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["misfit"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        # plt.title("Y Misfit as function of q1")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_misfit_q1_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["q1"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["q1"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel("q1"); plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("H energy as function of q1")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_energy_q1_sensibility.png"), dpi=200); plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics["misfit"], metrics["energy"], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row["misfit"], row["energy"]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel(r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$")
        plt.ylabel(r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$")
        # plt.title("Compromis YMisfit/HEnergy")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_n1_misfit_energy_q1_sensibility.png"), dpi=200); plt.close()

        print("\nAll done. Outputs figures are saved under:", outdir)

    return curves_z, curves_h, metrics


def q12_sensibility_with_optimum(A, G2, C, z0, h, T=1, N=200, q1_list=(0.25, 0.5, 1), q2_list=(0.25, 0.5, 1), save=False, outdir="figs"):
    """
    Sensibility of (q1, q2): sweep q1 and q2, compute solutions & metrics,
    produce 3D surfaces + 2D contour plots of the metrics as functions of (q1, q2),
    and mark the optimal (q1*, q2*) minimizing J = misfit + energy.
    """
    # Step 1 : Data input (completion)
    dt = T / N
    t  = np.linspace(0, T, N+1)
    h_true = h(t)

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = C * z_true

    # Pour stocker les trajectoires (optionnel)
    q1_list = list(q1_list)
    q2_list = list(q2_list)
    curves_z = [[None for _ in q2_list] for _ in q1_list]
    curves_h = [[None for _ in q2_list] for _ in q1_list]

    metrics_list = []

    # Double balayage en q1 et q2
    for i, q1 in enumerate(q1_list):
        for j, q2 in enumerate(q2_list):
            # Step 3: Backward z and p from y_m avec tir sur p(0)
            _, traj_z, traj_p = find_z_p(A, G2, C, N, dt, y_m, q1, q2)
            z_est = traj_z

            # Step 4: Reconstruction de l'irrigation h
            z0_est, h_est = reconstruction(G2, traj_p, N, q1, q2)

            curves_z[i][j] = z_est
            curves_h[i][j] = h_est

            # Metrics
            z0_error, z0_energy, z_rmse, h_rmse, misfit, energy = compute_metrics(
                C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t
            )
            metrics_list.append({
                "q1": q1,
                "q2": q2,
                "z0_error": z0_error,
                "z0_energy": z0_energy,
                "z_rmse": z_rmse,
                "h_rmse": h_rmse,
                "misfit": misfit,
                "energy": energy,
            })

    metrics = pd.DataFrame(metrics_list).sort_values(["q1", "q2"]).reset_index(drop=True)

    # --- Critère de compromis J(q1, q2) = misfit + λ * energy ---
    metrics["objective"] = metrics["misfit"] + metrics["energy"]

    # --- Point optimal : (q1*, q2*) minimisant J ---
    idx_opt = metrics["objective"].idxmin()
    best = metrics.loc[idx_opt]
    q1_opt, q2_opt = best["q1"], best["q2"]

    print(f"Optimal compromise :")
    print(f"  q1* = {q1_opt}, q2* = {q2_opt}")
    print(f"  z0 error = {best['z0_error']:.3f}")
    print(f"  RMSE(z) = {best['z_rmse']:.3f}")
    print(f"  RMSE(h) = {best['h_rmse']:.3f}")
    print(f"  Misfit*   = {best['misfit']:.3f}")
    print(f"  Energy*   = {best['energy']:.3f}")
    print(f"  Objective = {best['objective']:.3f}")

    # --- Construction des grilles (q1, q2) pour surfaces/contours ---
    def make_grid(df, value_col):
        q1_vals = np.sort(df["q1"].unique())
        q2_vals = np.sort(df["q2"].unique())
        pivot = df.pivot(index="q1", columns="q2", values=value_col).loc[q1_vals, q2_vals]
        Q1, Q2 = np.meshgrid(q1_vals, q2_vals, indexing="ij")  # Q1(i,j)=q1_i, Q2(i,j)=q2_j
        Z = pivot.values
        return Q1, Q2, Z, q1_vals, q2_vals

    # --- Fonction utilitaire pour tracer/sauver une surface 3D ---
    def plot_surface_metric(value_col, zlabel, filename, mark_opt=True):
        Q1, Q2, Z, _, _ = make_grid(metrics, value_col)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(Q1, Q2, Z, cmap="viridis", edgecolor="none", alpha=0.9)

        ax.set_xlabel("q1")
        ax.set_ylabel("q2")
        ax.set_zlabel(zlabel)
        # ax.set_title(zlabel + " as function of (q1, q2)")
        fig.colorbar(surf, shrink=0.5, aspect=10)

        if mark_opt:
            # valeur de la métrique au point optimal
            z_best = metrics.loc[idx_opt, value_col]
            ax.scatter(q1_opt, q2_opt, z_best, color="red", s=60, label="optimum J")
            ax.legend()

        plt.tight_layout()

        if save:
            outdir = 'outputs/' + outdir
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, filename), dpi=200)
            plt.close()
        else:
            plt.show()

    # --- Fonction utilitaire pour tracer/sauver une carte de niveaux 2D ---
    def plot_contour_metric(value_col, clabel, filename, mark_opt=True):
        Q1, Q2, Z, _, _ = make_grid(metrics, value_col)

        fig, ax = plt.subplots(figsize=(7, 5))
        # Contours remplis
        cf = ax.contourf(Q1, Q2, Z, levels=20, cmap="viridis")
        # Courbes d'iso-valeurs
        cs = ax.contour(Q1, Q2, Z, levels=10, colors="k", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8)

        if mark_opt:
            ax.plot(q1_opt, q2_opt, "ro", markersize=6, label="optimum J")
            ax.legend(loc="best")

        ax.set_xlabel("q1")
        ax.set_ylabel("q2")
        # ax.set_title(clabel + " (contours in (q1,q2)-plane)")
        cbar = fig.colorbar(cf)
        cbar.set_label(clabel)
        plt.tight_layout()

        if save:
            outdir = 'outputs/' + outdir
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, filename), dpi=200)
            plt.close()
        else:
            plt.show()

    # --- Surfaces 3D pour chaque métrique (avec point optimal) ---
    plot_surface_metric("z0_error",  r"$z_{0Error}=|z_0 - z_0^\star|$",   "surf_z0_error_q1_q2.png")
    plot_surface_metric("z0_energy", r"$|z_0^\star|$",                    "surf_z0_energy_q1_q2.png")
    plot_surface_metric("z_rmse",    r"$RMSE_z$",                         "surf_z_rmse_q1_q2.png")
    plot_surface_metric("h_rmse",    r"$RMSE_h$",                         "surf_h_rmse_q1_q2.png")
    plot_surface_metric("misfit",    r"$\mathrm{YMisfit}=\int_0^T\|Cz-y^m\|^2 dt$", "surf_misfit_q1_q2.png")
    plot_surface_metric("energy",    r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$",  "surf_energy_q1_q2.png")
    # Surface 3D du critère J
    plot_surface_metric("objective", r"$J=\mathrm{YMisfit}+\mathrm{HEnergy}$", "surf_objective_q1_q2.png")

    # --- Contours 2D pour chaque métrique (avec point optimal) ---
    plot_contour_metric("z0_error",  r"$z_{0Error}=|z_0 - z_0^\star|$",   "contour_z0_error_q1_q2.png")
    plot_contour_metric("z0_energy", r"$|z_0^\star|$",                    "contour_z0_energy_q1_q2.png")
    plot_contour_metric("z_rmse",    r"$RMSE_z$",                         "contour_z_rmse_q1_q2.png")
    plot_contour_metric("h_rmse",    r"$RMSE_h$",                         "contour_h_rmse_q1_q2.png")
    plot_contour_metric("misfit",    r"$\mathrm{YMisfit}=\int_0^T\|Cz-y^m\|^2 dt$", "contour_misfit_q1_q2.png")
    plot_contour_metric("energy",    r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$",  "contour_energy_q1_q2.png")
    # Contours 2D du critère J
    plot_contour_metric("objective", r"$J=\mathrm{YMisfit}+\mathrm{HEnergy}$", "contour_objective_q1_q2.png")

    if save:
        print("\nAll done. Surfaces, contours and optimal point are saved under:", outdir)

    return curves_z, curves_h, metrics, q1_opt, q2_opt
