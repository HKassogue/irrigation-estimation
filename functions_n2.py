# %% [markdown]
# ## Problem of irrigation water estimation using the shooting method in dimension 2 -- Various functions needed
# **But.** 

# For given $A\in\mathcal{M}_2(\mathbb{R})$, $G_2\in\mathcal{M}_2(\mathbb{R})$, $C\in\mathcal{M}_2(\mathbb{R})$, and $q_1,q_2>0$ considered ponderation weights, we want to construct the optimal pair
# ```math
# \begin{aligned}
# h^\star(t) &= -\tfrac{1}{q_2} G_2^\top p(t),\\
# z(0) &= -\tfrac{1}{q_1} p(0)
# \end{aligned}
# ```
# where $(z,p)$ is solution of the optimality system  given by the TPBVP:
# ```math
# \begin{aligned}
# z'(t) &= A z(t) + G_2 h^\star(t) = A z(t) - \tfrac{1}{q_2} G_2 G_2^\top p(t), \quad z(0) = -\tfrac{1}{q_1} s,\\
# p'(t) &= -A^\top p(t) - C^\top\!\big(C z(t) - y^{\mathrm m}(t)\big), \quad p(T)=0,
# \end{aligned}
# ```
# and $s := p(0)\in\mathbb{R}^2$ is **the shoot unknown**.  
# 
# 
# **The functions concern the implementation scheme steps 2 to 5 **:
# 1. Give input data 
#     - System: A, $G_2$, C
#     - Objective: z0, h as known
#     - Time discretization: T, N
#     - Regulations: q1, q2
# 2. Forward z_true and y_m from known z0 and h
#     - Integrate $z'(t) = A z(t) + G_2 h(t)$ using RK4 over grid t
#     - Then compute $y(t) = C z(t)$
# 3. Backward p and $z_{est}$ from y_m with shooting method on p(0)
#     - Define a function that integrate $(z'(t), p'(t))$ with $z(0) = -s/q1$, $p(T) = 0$ under the shoot $s=p(0)$.
#     - Define the shooting function $F(s) := p(T; s)$ and find the good $s^\star$ such that $F(s^\star)=0$ by multiple integrations with $s$.
#     - Integrate final (z, p) with the good $s^\star$.
# 4. Reconstruction of the irrigation $h_{est}$
#     - $h(t) = -(1/q2) * G_2^T p(t)$
#     - $z(0) = -(1/q1) * p(0)$
# 5. Make plots and metrics
#     - Metrics: 
#         - $Z0Error = z_0 - z_{0est}$
#         - $RMSE_z = \sqrt{mean((z_{true} - z_{est})^2)}$
#         - $RMSE_h = \sqrt{mean((z_{true} - z_{est})^2)}$
#         - $Y Misfit = \int_0^T \|Cz(t)-y^m(t)\|^2 dt$
#         - $H Energy = \int_0^T \|h(t)\|^2 dt$
#     - Plots:
#         - $z_{true}$ vs $z_{est}$
#         - $h_{true}$ vs $h_{est}$
#         - $z_{est}$ as function of q1 and/or q2
#         - $h_{est}$ as function of q1 and/or q2
#         - $RMSE_z$ as function of q1 and/or q2
#         - $RMSE_h$ as function of q1 and/or q2
#         - $Y Misfit$ as function of q1 and/or q2
#         - $H Energy$ as function of q1 and/or q2

# %%
# Modules importation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D surface
import pandas as pd
import os


# %%
# Step 1 functions
def forward_z_from_h(A, G2, z0, h, N, dt):
    """Integrate z'(t) = A z(t) + G2(t) h(t) from known z0 and h using RK4 over grid t
    Then compute Y(t) = C z(t)
    - z0 numpy array (2, )
    - h numpy array (N+1, 2)"""
    z = np.zeros((N+1, 2)); z[0] = z0
    for k in range(N):
        zk = z[k]; hk, hk_1 = h[k, :], h[k+1, :]
        h_mid = (hk + hk_1) / 2.0
        fz = lambda zz, hh : A @ zz + G2 @ hh
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
        y_meas : numpy array (N+1, 2) containing y^m(t_k)
        q1, q2: scalar
    """
    z = - (1.0 / q1) * s.copy()
    p = s.copy()

    traj_z = np.zeros((N+1, 2))
    traj_p = np.zeros((N+1, 2))
    traj_z[0] = z
    traj_p[0] = p

    G2G2T = G2 @ G2.T
    AT = A.T
    CT = C.T

    for k in range(N):
        yk = y_meas[k]  # y^m(t_k)

        # z' = A z - (1/q2) G2 G2^T p
        zdot = A @ z - (1.0 / q2) * (G2G2T @ p)

        # p' = -A^T p - C^T (C z - y^m)
        Cz_minus_y = C @ z - yk
        pdot = - AT @ p - CT @ Cz_minus_y

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


def find_s_star(A, G2, C, N, dt, y_meas, q1, q2, s_init=np.zeros((2,)), tol=1e-6, maxit=100, fd_eps=1e-5):
    """Resolve F(s)=P(T;s)=0 by Newton method with approached jacobian.
    s_init : np.array(2,) initial points for s"""
    s = s_init.copy()
    for it in range(maxit):
        # First evaluation of F(s)
        F0 = F_of_s(A, G2, C, N, dt, y_meas, q1, q2, s)  # (n,)

        # Stopping criteria
        if np.linalg.norm(F0, ord=2) < tol:
            # Good s found
            return s

        # Computation of Jacobian J ~ dF/ds by finite differences
        J = np.zeros((2, 2))
        for j in range(2):
            s_pert = s.copy()
            s_pert[j] += fd_eps
            Fj = F_of_s(A, G2, C, N, dt, y_meas, q1, q2, s_pert)
            # Column j = (F(s+eps e_j) - F(s)) / eps
            J[:, j] = (Fj - F0) / fd_eps

        # Resolve J * delta = F0   (we want in fact s_{k+1} = s_k - delta)
        try:
            delta = np.linalg.solve(J, F0)
        except np.linalg.LinAlgError:
            # Unconditionned Jacobian, we make a pseudo-inverse
            delta = np.linalg.pinv(J) @ F0

        # Update of s
        s = s - delta

    # If we exit the loop without converging, we return the last s
    return s


def find_z_p(A, G2, C, N, dt, y_meas, q1, q2):
    """Finding final (z,p) from shooting on p(0)"""
    s_star = find_s_star(A, G2, C, N, dt, y_meas, q1, q2)
    return backward_z_p_by_shooting(A, G2, C, N, dt, y_meas, q1, q2, s_star)


def reconstruction(G2, traj_p, N, q1, q2):
    """Return (z0*, h*) from (., p), solution of the TPBVP:
        h(t) = -(1/q2) * G2^T p(t)
        z(0) = -(1/q1) * p(0) already in traj_z
    """
    z0 = -traj_p[0] / q1 # already in traj_z

    traj_h = np.zeros((N+1, 2))
    for k in range(N+1):
        traj_h[k] = - (1.0 / q2) * (G2.T @ traj_p[k])
    
    return z0, traj_h


def simulate_n2(A, G2, C, z0, h1, h2, T=1, N=200, q1=1, q2=1):
    """Simulate the irrigation water command estimation for n=2
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
    h_true = np.column_stack([h1(t), h2(t)])

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = (C @ z_true.T).T

    # Step 3: Backward z and p from y_m with shooting method on p(0)
    _, traj_z, traj_p = find_z_p(A, G2, C, N, dt, y_m, q1, q2)
    z_est = traj_z

    # Step 4: Reconstruction of the irrigation h
    z0_est, h_est = reconstruction(G2, traj_p, N, q1, q2)

    # Outputs
    return h_true, z_true, y_m, z0_est, z_est, h_est


def compute_metrics(C, z0_true, h_true, z_true, y_m, z0_est, z_est, h_est, t):
    """Compute the metrics"""
    z0_error = np.linalg.norm(z0_est - z0_true)
    z0_energy = np.linalg.norm(z0_est)
    z_rmse = np.sqrt(np.mean((z_est - z_true)**2))
    h_rmse = np.sqrt(np.mean((h_est - h_true)**2))
    misfit = np.trapz(np.sum((C @ z_est.T - y_m.T)**2, axis=0), t)
    henergy = np.trapz(np.sum(h_est**2, axis=1), t)
    return z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy


def make_plots(h_true, z_true, z_est, h_est, t, show=True, save=False, file_end=None, outdir = "figs"):
    """Plots the outputs, and/or save them optionnaly in a created directory if it does not exist."""
    def plot(y1, y2, y1label, y2label, ylabel, filename):
        plt.figure(figsize=(8, 5)); plt.plot(t, y1, label=y1label); plt.plot(t, y2, "--", label=y2label); plt.legend(); plt.xlabel("t"); plt.ylabel(ylabel); plt.grid(alpha=0.7, linewidth=0.5); plt.tight_layout()
        if save and filename:
            plt.savefig(os.path.join(outdir, filename), dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    if save:
        outdir = 'outputs/' + outdir
        os.makedirs(outdir, exist_ok=True)

    plot(h_true[:, 0], h_est[:, 0], "h1 true", "h1 estimated", r"$h_1(t)$", f"fig_n2_h1_true_vs_h1_estimated_{file_end}.png")
    plot(h_true[:, 1], h_est[:, 1], "h2 true", "h2 estimated", r"$h_2(t)$", f"fig_n2_h2_true_vs_h2_estimated_{file_end}.png")
    plot(z_true[:, 0], z_est[:, 0], "z1 true", "z1 estimated", r"$z_1(t)$", f"fig_n2_z1_true_vs_z1_estimated_{file_end}.png")
    plot(z_true[:, 1], z_est[:, 1], "z2 true", "z2 estimated", r"$z_2(t)$", f"fig_n2_z2_true_vs_z2_estimated_{file_end}.png")

    if save:
        print("All done. Outputs figures are saved under:", outdir)


def q2_sensibility(A, G2, C, z0, h1, h2, T=1, N=200, q1=1, q2_list=[0.25, 0.5, 1], show=True, save=False, file_end=None, outdir = "figs"):
    """Sensibility of q2: sweep q2, compute solutions & metrics"""
    # Step 1 : Data input (completion)
    dt = T / N
    t  = np.linspace(0, T, N+1)
    h_true = np.column_stack([h1(t), h2(t)])

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = (C @ z_true.T).T
    
    curves_z1, curves_z2, curves_h1, curves_h2, metrics = [], [], [], [], []
    for q2 in q2_list:
        # Step 3: Backward z and p from y_m with shooting method on p(0)
        _, traj_z, traj_p = find_z_p(A, G2, C, N, dt, y_m, q1, q2)
        z_est = traj_z

        # Step 4: Reconstruction of the irrigation h
        z0_est, h_est = reconstruction(G2, traj_p, N, q1, q2) 
        curves_z1.append(z_est[:, 0]); curves_z2.append(z_est[:, 1])
        curves_h1.append(h_est[:, 0]); curves_h2.append(h_est[:, 1])

        # Metrics
        z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
        metrics.append({
            "q2": q2,
            "z0_error": z0_error,
            "z0_energy": z0_energy,
            "z_rmse": z_rmse,
            "h_rmse": h_rmse,
            "misfit": misfit,
            "henergy": henergy,
            "energy": z0_energy**2 + henergy,
        })
    metrics = pd.DataFrame(metrics).sort_values("q2").reset_index(drop=True)
    
    # --- Utilitary function to plot/save the h or z curve over t per q2 ---
    def make_plot_over_t_per_q2(values_true, values_est, label, ylabel, title=None, filename=None):
        plt.figure(figsize=(8, 5))
        plt.plot(t, values_true, "k-", label=label)
        for i, q2 in enumerate(q2_list):
            plt.plot(t, values_est[i], "--", label=f"q2={q2}")
        plt.xlabel("t"); plt.ylabel(ylabel)
        # if title: plt.title(title)
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()
        if save and filename:
            plt.savefig(os.path.join(outdir, filename), dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    # --- Utilitary function to plot/save the metrics over q2 and anotate it ---
    def make_plot_over_q2_and_anotate(x, y, xlabel, ylabel, title=None, filename=None):
        plt.figure(figsize=(8, 5))
        plt.plot(metrics[x], metrics[y], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q2={row['q2']}", (row[x], row[y]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        # if title: plt.title(title)
        plt.tight_layout()
        if save and filename:
            plt.savefig(os.path.join(outdir, filename), dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    # Figures (overlays & curves)
    if save:
        outdir = 'outputs/' + outdir
        os.makedirs(outdir, exist_ok=True)

    file_end = f"q2_sensibility_with_q1_{q1}_{file_end}" if file_end else f"q2_sensibility_with_q1_{q1}"

    make_plot_over_t_per_q2(h_true[:, 0], curves_h1, 
                            label=f"h1 true", 
                            ylabel=r"$h_1(t)$", 
                            title="Sensibility in q2 of h1", 
                            filename=f"fig_n2_h1_{file_end}.png")
    make_plot_over_t_per_q2(h_true[:, 1], curves_h2, 
                            label=f"h2 true", 
                            ylabel=r"$h_2(t)$", 
                            title="Sensibility in q2 of h2", 
                            filename=f"fig_n2_h2_{file_end}.png")
    make_plot_over_t_per_q2(z_true[:, 0], curves_z1, 
                            label=f"z1 true", 
                            ylabel=r"$z_1(t)$", 
                            title="Sensibility in q2 of z1", 
                            filename=f"fig_n2_z1_{file_end}.png")
    make_plot_over_t_per_q2(z_true[:, 1], curves_z2, 
                            label=f"z2 true", 
                            ylabel=r"$z_2(t)$", 
                            title="Sensibility in q2 of z2", 
                            filename=f"fig_n2_z2_{file_end}.png")
    make_plot_over_q2_and_anotate('q2', 'z0_error', 
                                    xlabel='q2', 
                                    ylabel=r"$z_{0Error}=|z_0 - z_0^\star|$", 
                                    title=r"Error of $z_0$", 
                                    filename=f"fig_n2_z0error_{file_end}.png")
    make_plot_over_q2_and_anotate('q2', 'z0_energy', 
                                    xlabel='q2', 
                                    ylabel=r"$z_{0Energy}=|z_0^\star|$", 
                                    title=r"Energy of $z_0$", 
                                    filename=f"fig_n2_z0energy_{file_end}.png")
    make_plot_over_q2_and_anotate('q2', 'z_rmse', 
                                    xlabel='q2', 
                                    ylabel=r"$RMSE_z$", 
                                    title=r"$RMSE_z$", 
                                    filename=f"fig_n2_z_rmse_{file_end}.png")
    make_plot_over_q2_and_anotate('q2', 'h_rmse', 
                                    xlabel='q2', 
                                    ylabel=r"$RMSE_h$", 
                                    title=r"$RMSE_h$", 
                                    filename=f"fig_n2_h_rmse_{file_end}.png")
    make_plot_over_q2_and_anotate('z_rmse', 'h_rmse', 
                                    xlabel=r"$RMSE_z$", 
                                    ylabel=r"$RMSE_h$", 
                                    title=r"Compromis $RMSE_z$/$RMSE_h$", 
                                    filename=f"fig_n2_z_h_rmse_{file_end}.png")
    make_plot_over_q2_and_anotate('q2', 'misfit', 
                                    xlabel='q2', 
                                    ylabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    title="Y Misfit as function of q2", 
                                    filename=f"fig_n2_misfit_{file_end}.png")
    make_plot_over_q2_and_anotate('q2', 'henergy', 
                                    xlabel='q2', 
                                    ylabel=r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$", 
                                    title="H energy as function of q2", 
                                    filename=f"fig_n2_henergy_{file_end}.png")
    make_plot_over_q2_and_anotate('q2', 'energy', 
                                    xlabel='q2', 
                                    ylabel=r"$z_0\mathrm{HEnergy}=\|z_0\|^2 + \int_0^T \|h(t)\|^2 dt$", 
                                    title="$z_0$HEnergy as function of q2", 
                                    filename=f"fig_n2_z0henergy_{file_end}.png")
    make_plot_over_q2_and_anotate('misfit', 'henergy', 
                                    xlabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    ylabel=r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$", 
                                    title="Compromis YMisfit/HEnergy", 
                                    filename=f"fig_n2_misfit_henergy_{file_end}.png")
    make_plot_over_q2_and_anotate('misfit', 'z0_energy', 
                                    xlabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    ylabel=r"$z_0\mathrm{Energy}=\|z_0\|$", 
                                    title=r"Compromis YMisfit/$z_0$Energy", 
                                    filename=f"fig_n2_misfit_z0energy_{file_end}.png")
    make_plot_over_q2_and_anotate('misfit', 'energy', 
                                    xlabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    ylabel=r"$z_0\mathrm{HEnergy}=\|z_0\|^2 + \int_0^T \|h(t)\|^2 dt$", 
                                    title=r"Compromis YMisfit/$z_0$HEnergy", 
                                    filename=f"fig_n2_misfit_z0henergy_{file_end}.png")
        
    if save:
        print("All done. Outputs figures are saved under:", outdir)

    return curves_z1, curves_z2, curves_h1, curves_h2, metrics


def q1_sensibility(A, G2, C, z0, h1, h2, T=1, N=200, q1_list=[0.25, 0.5, 1], q2=1, show=True, save=False, file_end=None, outdir = "figs"):
    """Sensibility of q1: sweep q1, compute solutions & metrics"""
    # Step 1 : Data input (completion)
    dt = T / N
    t  = np.linspace(0, T, N+1)
    h_true = np.column_stack([h1(t), h2(t)])

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = (C @ z_true.T).T
    
    curves_z1, curves_z2, curves_h1, curves_h2, metrics = [], [], [], [], []
    for q1 in q1_list:
        # Step 3: Backward z and p from y_m with shooting method on p(0)
        _, traj_z, traj_p = find_z_p(A, G2, C, N, dt, y_m, q1, q2)
        z_est = traj_z

        # Step 4: Reconstruction of the irrigation h
        z0_est, h_est = reconstruction(G2, traj_p, N, q1, q2) 
        curves_z1.append(z_est[:, 0]); curves_z2.append(z_est[:, 1])
        curves_h1.append(h_est[:, 0]); curves_h2.append(h_est[:, 1])

        # Metrics
        z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
        metrics.append({
            "q1": q1,
            "z0_error": z0_error,
            "z0_energy": z0_energy,
            "z_rmse": z_rmse,
            "h_rmse": h_rmse,
            "misfit": misfit,
            "henergy": henergy,
            "energy": z0_energy**2 + henergy,
        })
    metrics = pd.DataFrame(metrics).sort_values("q1").reset_index(drop=True)

    # --- Utilitary function to plot/save the h or z curve over t per q1 ---
    def make_plot_over_t_per_q1(values_true, values_est, label, ylabel, title=None, filename=None):
        plt.figure(figsize=(8, 5))
        plt.plot(t, values_true, "k-", label=label)
        for i, q1 in enumerate(q1_list):
            plt.plot(t, values_est[i], "--", label=f"q1={q1}")
        plt.xlabel("t"); plt.ylabel(ylabel)
        # if title: plt.title(title)
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.legend(); plt.tight_layout()
        if save and filename:
            plt.savefig(os.path.join(outdir, filename), dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    # --- Utilitary function to plot/save the metrics over q1 and anotate it ---
    def make_plot_over_q1_and_anotate(x, y, xlabel, ylabel, title=None, filename=None):
        plt.figure(figsize=(8, 5))
        plt.plot(metrics[x], metrics[y], marker="o")
        for i, row in metrics.iterrows():
            plt.annotate(f"q1={row['q1']}", (row[x], row[y]))
        plt.grid(alpha=0.7, linewidth=0.5)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        # if title: plt.title(title)
        plt.tight_layout()
        if save and filename:
            plt.savefig(os.path.join(outdir, filename), dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    # Figures (overlays & curves)
    if save:
        outdir = 'outputs/' + outdir
        os.makedirs(outdir, exist_ok=True)

    file_end = f"q1_sensibility_with_q2_{q2}_{file_end}" if file_end else f"q1_sensibility_with_q2_{q2}"

    make_plot_over_t_per_q1(h_true[:, 0], curves_h1, 
                            label=f"h1 true", 
                            ylabel=r"$h_1(t)$", 
                            title="Sensibility in q1 of h1", 
                            filename=f"fig_n2_h1_{file_end}.png")
    make_plot_over_t_per_q1(h_true[:, 1], curves_h2, 
                            label=f"h2 true", 
                            ylabel=r"$h_2(t)$", 
                            title="Sensibility in q1 of h2", 
                            filename=f"fig_n2_h2_{file_end}.png")
    make_plot_over_t_per_q1(z_true[:, 0], curves_z1, 
                            label=f"z1 true", 
                            ylabel=r"$z_1(t)$", 
                            title="Sensibility in q1 of z1", 
                            filename=f"fig_n2_z1_{file_end}.png")
    make_plot_over_t_per_q1(z_true[:, 1], curves_z2, 
                            label=f"z2 true", 
                            ylabel=r"$z_2(t)$", 
                            title="Sensibility in q1 of z2", 
                            filename=f"fig_n2_z2_{file_end}.png")
    make_plot_over_q1_and_anotate('q1', 'z0_error', 
                                    xlabel='q1', 
                                    ylabel=r"$z_{0Error}=|z_0 - z_0^\star|$", 
                                    title=r"Error of $z_0$", 
                                    filename=f"fig_n2_z0error_{file_end}.png")
    make_plot_over_q1_and_anotate('q1', 'z0_energy', 
                                    xlabel='q1', 
                                    ylabel=r"$z_{0Energy}=|z_0^\star|$", 
                                    title=r"Energy of $z_0$", 
                                    filename=f"fig_n2_z0energy_{file_end}.png")
    make_plot_over_q1_and_anotate('q1', 'z_rmse', 
                                    xlabel='q1', 
                                    ylabel=r"$RMSE_z$", 
                                    title=r"$RMSE_z$", 
                                    filename=f"fig_n2_z_rmse_{file_end}.png")
    make_plot_over_q1_and_anotate('q1', 'h_rmse', 
                                    xlabel='q1', 
                                    ylabel=r"$RMSE_h$", 
                                    title=r"$RMSE_h$", 
                                    filename=f"fig_n2_h_rmse_{file_end}.png")
    make_plot_over_q1_and_anotate('z_rmse', 'h_rmse', 
                                    xlabel=r"$RMSE_z$", 
                                    ylabel=r"$RMSE_h$", 
                                    title=r"Compromis $RMSE_z$/$RMSE_h$", 
                                    filename=f"fig_n2_z_h_rmse_{file_end}.png")
    make_plot_over_q1_and_anotate('q1', 'misfit', 
                                    xlabel='q1', 
                                    ylabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    title="Y Misfit as function of q1", 
                                    filename=f"fig_n2_misfit_{file_end}.png")
    make_plot_over_q1_and_anotate('q1', 'henergy', 
                                    xlabel='q1', 
                                    ylabel=r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$", 
                                    title="H energy as function of q1", 
                                    filename=f"fig_n2_henergy_{file_end}.png")
    make_plot_over_q1_and_anotate('q1', 'energy', 
                                    xlabel='q1', 
                                    ylabel=r"$z_0\mathrm{HEnergy}=\|z_0\|^2 + \int_0^T \|h(t)\|^2 dt$", 
                                    title=r"$z_0$HEnergy as function of q1", 
                                    filename=f"fig_n2_z0henergy_{file_end}.png")
    make_plot_over_q1_and_anotate('misfit', 'henergy', 
                                    xlabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    ylabel=r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$", 
                                    title="Compromis YMisfit/HEnergy", 
                                    filename=f"fig_n2_misfit_henergy_{file_end}.png")
    make_plot_over_q1_and_anotate('misfit', 'z0_energy', 
                                    xlabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    ylabel=r"$z_0\mathrm{Energy}=\|z_0\|$", 
                                    title=r"Compromis YMisfit/$z_0$Energy", 
                                    filename=f"fig_n2_misfit_z0energy_{file_end}.png")
    make_plot_over_q1_and_anotate('misfit', 'energy', 
                                    xlabel=r"$\mathrm{YMisfit}=\int_0^T\|Cz(t)-y^m(t)\|^2 dt$", 
                                    ylabel=r"$z_0\mathrm{HEnergy}=\|z_0\|^2 + \int_0^T \|h(t)\|^2 dt$", 
                                    title=r"Compromis YMisfit/$z_0$HEnergy", 
                                    filename=f"fig_n2_misfit_z0henergy_{file_end}.png")
        
    if save:
        print("All done. Outputs figures are saved under:", outdir)

    return curves_z1, curves_z2, curves_h1, curves_h2, metrics


def find_Lcurve_elbow(metrics, x_col="z_rmse", y_col="h_rmse", use_log=True, sort_on='q1'):
    """
    Trouve automatiquement un 'coude' sur la L-curve (x_col, y_col)
    par la méthode de la distance maximale à la corde
    reliant les deux extrémités de la courbe.

    metrics : DataFrame contenant x_col et y_col (ex: 'z_rmse', 'h_rmse')
    x_col   : nom de la colonne utilisée comme abscisse
    y_col   : nom de la colonne utilisée comme ordonnée
    use_log : si True, on travaille sur log10(x), log10(y)

    Retour
    ------
    idx_elbow : index (ligne) dans metrics correspondant au coude.
    """
    # On fait une copie triée par q1 ou q2 (paramètre de la courbe)
    df = metrics.sort_values(sort_on).reset_index()
    idx_original = df["index"].values  # indices d'origine dans metrics

    # Extraction de x et y
    x = df[x_col].values
    y = df[y_col].values

    # Passage en log si souhaité (classique pour les L-curves)
    if use_log:
        x = np.log10(x)
        y = np.log10(y)

    # Point initial et final de la cordes
    x1, y1 = x[0],     y[0]
    x2, y2 = x[-1],    y[-1]

    # Vecteur de la corde
    line_vec = np.array([x2 - x1, y2 - y1])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        # cas dégénéré : tous les points sont identiques
        return idx_original[0]

    # Vecteur unitaire
    line_unit = line_vec / line_len

    # Distance de chaque point à la droite (x1,y1)-(x2,y2)
    # dist = || p - proj_ligne(p) ||
    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi - x1, yi - y1])
        proj_len = np.dot(p, line_unit)
        proj = proj_len * line_unit
        dist = np.linalg.norm(p - proj)
        distances.append(dist)

    distances = np.array(distances)
    i_max = np.argmax(distances)        # indice dans le tableau trié
    idx_elbow = idx_original[i_max]       # index réel dans metrics

    return idx_elbow


def find_compromise_point(metrics, x_col="z_rmse", y_col="h_rmse"):
    # On normalise pour que les deux axes soient comparables
    x_norm = (metrics[x_col] - metrics[x_col].min()) / (metrics[x_col].max() - metrics[x_col].min())
    y_norm = (metrics[y_col] - metrics[y_col].min()) / (metrics[y_col].max() - metrics[y_col].min())
    dist2 = x_norm**2 + y_norm**2
    idx = dist2.idxmin()
    return idx


def q12_sensibility_with_optimum(A, G2, C, z0, h1, h2, T=1, N=200, q1_list=(0.25, 0.5, 1), q2_list=(0.25, 0.5, 1), elbow_on='rmse', show=True, save=False, file_end=None, outdir="figs"):
    """
    Sensibility of (q1, q2): sweep q1 and q2, compute solutions & metrics,
    produce 3D surfaces + 2D contour plots of the metrics as functions of (q1, q2),
    and mark the optimal (q1*, q2*) as Lcurve compromise on (RMSE_z, RMSE_h) or (YMisfit, z0HEnergy).
    """
    # Step 1 : Data input (completion)
    dt = T / N
    t  = np.linspace(0, T, N+1)
    h_true = np.column_stack([h1(t), h2(t)])

    # Step 2: Forward z_true and y_m from z0 and h known
    z_true = forward_z_from_h(A, G2, z0, h_true, N, dt)
    y_m = (C @ z_true.T).T

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
            z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy = compute_metrics(
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
                "henergy": henergy,
                "energy": z0_energy**2 + henergy,
                "objective": misfit + q1 * z0_energy**2 + q2 * henergy
            })

    metrics = pd.DataFrame(metrics_list).sort_values(["q1", "q2"]).reset_index(drop=True)

    # --- Optimization criterion J(z0, h; q1, q2) = yMisfit + q1 * z0Energy + q2 * hEnergy ---
    # idx_opt = metrics["objective"].idxmin()

    # --- Optimum comprise on (RMSE_z, RMSE_h) or (YMisfit, Energy) chosing (q1*, q2*) ---
    # idx_opt = find_compromise_point(metrics, x_col="z_rmse", y_col="h_rmse") if elbow_on == 'rmse' else find_compromise_point(metrics, x_col="misfit", y_col="energy")

    # --- Elbow-based comprise on (RMSE_z, RMSE_h) or (YMisfit, Energy) chosing (q1*, q2*) ---
    idx_opt = find_Lcurve_elbow(metrics, x_col="z_rmse", y_col="h_rmse", use_log=True) if elbow_on == 'rmse' else find_Lcurve_elbow(metrics, x_col="misfit", y_col="energy", use_log=True)

    best = metrics.loc[idx_opt]
    q1_opt, q2_opt = best["q1"], best["q2"]

    print("Elbow-based compromise (L-curve on", "(RMSE_z, RMSE_h)" if elbow_on == 'rmse' else "(YMisfit, z0HEnergy)" + ") :")
    print(f"  q1* = {q1_opt}, q2* = {q2_opt}")
    print(f"  z0Error*     = {best['z0_error']}")
    print(f"  RMSE_z*      = {best['z_rmse']}")
    print(f"  RMSE_h*      = {best['h_rmse']}")
    print(f"  YMisfit*     = {best['misfit']}")
    print(f"  z0HEnergy*   = {best['energy']}")
    print(f"  z0Energy*    = {best['z0_energy']}")
    print(f"  HEnergy*     = {best['henergy']}")
    print(f"  J*           = {best['objective']}")

    # --- Construction des grilles (q1, q2) pour surfaces/contours ---
    def make_grid(df, value_col):
        q1_vals = np.sort(df["q1"].unique())
        q2_vals = np.sort(df["q2"].unique())
        pivot = df.pivot(index="q1", columns="q2", values=value_col).loc[q1_vals, q2_vals]
        Q1, Q2 = np.meshgrid(q1_vals, q2_vals, indexing="ij")  # Q1(i,j)=q1_i, Q2(i,j)=q2_j
        Z = pivot.values
        return Q1, Q2, Z
    
    if save:
        outdir = 'outputs/' + outdir
        os.makedirs(outdir, exist_ok=True)

    label = 'Compromise (RMSE_z/RMSE_h)' if elbow_on == 'rmse' else 'Compromise (YMisfit/z0HEnergy)'

    # --- Fonction utilitaire pour tracer/sauver une surface 3D ---
    def plot_surface_metric(value_col, zlabel, filename, mark_opt=True):
        Q1, Q2, Z = make_grid(metrics, value_col)

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
            ax.scatter(q1_opt, q2_opt, z_best, color="red", s=60, label=label)
            ax.legend(loc="best")

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(outdir, filename), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    # --- Fonction utilitaire pour tracer/sauver une carte de niveaux 2D ---
    def plot_contour_metric(value_col, clabel, filename, mark_opt=True):
        Q1, Q2, Z = make_grid(metrics, value_col)

        fig, ax = plt.subplots(figsize=(7, 5))
        # Contours remplis
        cf = ax.contourf(Q1, Q2, Z, levels=20, cmap="viridis")
        # Courbes d'iso-valeurs
        cs = ax.contour(Q1, Q2, Z, levels=10, colors="k", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8)

        if mark_opt:
            ax.plot(q1_opt, q2_opt, "ro", markersize=6, label=label)
            ax.legend(loc="best")

        ax.set_xlabel("q1")
        ax.set_ylabel("q2")
        # ax.set_title(clabel + " (contours in (q1,q2)-plane)")
        cbar = fig.colorbar(cf)
        cbar.set_label(clabel)
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(outdir, filename), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    file_end = f"q1_q2_sensibility_{file_end}" if file_end else "q1_q2_sensibility"

    print('-'*70,'\nPlot process: ...')
    # --- Surfaces 3D pour chaque métrique (avec point optimal) ---
    plot_surface_metric("z0_error",  r"$z_{0Error}=|z_0 - z_0^\star|$",             f"surf_n2_z0_error_{file_end}.png")
    plot_surface_metric("z0_energy", r"$|z_0^\star|$",                              f"surf_n2_z0_energy_{file_end}.png")
    plot_surface_metric("z_rmse",    r"$RMSE_z$",                                   f"surf_n2_z_rmse_{file_end}.png")
    plot_surface_metric("h_rmse",    r"$RMSE_h$",                                   f"surf_n2_h_rmse_{file_end}.png")
    plot_surface_metric("misfit",    r"$\mathrm{YMisfit}=\int_0^T\|Cz-y^m\|^2 dt$", f"surf_n2_misfit_{file_end}.png")
    plot_surface_metric("henergy",   r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$",  f"surf_n2_henergy_{file_end}.png")
    plot_surface_metric("energy",    r"$z_0\mathrm{HEnergy}=\|z_0\|^2+\int_0^T \|h(t)\|^2 dt$", f"surf_n2_z0henergy_{file_end}.png")
    plot_surface_metric("objective", r"$J(z_0^*,h^*)$ as function of $(q_1, q_2)$",             f"surf_n2_objective_{file_end}.png")

    # --- Contours 2D pour chaque métrique (avec point optimal) ---
    plot_contour_metric("z0_error",  r"$z_{0Error}=|z_0 - z_0^\star|$",             f"contour_n2_z0_error_{file_end}.png")
    plot_contour_metric("z0_energy", r"$|z_0^\star|$",                              f"contour_n2_z0_energy_{file_end}.png")
    plot_contour_metric("z_rmse",    r"$RMSE_z$",                                   f"contour_n2_z_rmse_{file_end}.png")
    plot_contour_metric("h_rmse",    r"$RMSE_h$",                                   f"contour_n2_h_rmse_{file_end}.png")
    plot_contour_metric("misfit",    r"$\mathrm{YMisfit}=\int_0^T\|Cz-y^m\|^2 dt$", f"contour_n2_misfit_{file_end}.png")
    plot_contour_metric("henergy",   r"$\mathrm{HEnergy}=\int_0^T \|h(t)\|^2 dt$",  f"contour_n2_henergy_{file_end}.png")
    plot_contour_metric("energy",    r"$z_0\mathrm{HEnergy}=\|z_0\|^2+\int_0^T \|h(t)\|^2 dt$", f"contour_n2_z0henergy_{file_end}.png")
    plot_contour_metric("objective", r"$J(z_0^*,h^*)$ as function of $(q_1, q_2)$",             f"contour_n2_objective_{file_end}.png")

    if save:
        print("All done. Surfaces, contours and optimal point are saved under:", outdir)

    return curves_z, curves_h, metrics, q1_opt, q2_opt
