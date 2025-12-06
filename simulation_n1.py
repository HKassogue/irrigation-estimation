# %% [markdown]
## Resolution of a problem of irrigation water estimation using the shooting method in dimension 1

# **But.** 

# For given $A\in\mathbb{R}$, $G_2\in\mathbb{R}$, $C\in\mathbb{R}$, and $q_1,q_2>0$ considered ponderation weights, we want to construct the optimal pair
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
# and $s := p(0)\in\mathbb{R}$ is **the shoot unknown**. 

# ---

# ### Implementation scheme (pseudo-code)
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

# ---

# ### Input example
# - System: $A=2, G_2=3, C=4.9$
# - Objective: $z_{0true}=4, h_{\mathrm{true}}(t)=t(1 - t)$
# - Time discretization: $T=1, N=200$
# - Regulation: $q_1=0.001, q_2=0.25$

# ---

# ### Ouput example (metrics)
# z0 true = 4, z0 estimated = 4.000, error = 0.000<br>
# RMSE(z) = 0.003<br>
# RMSE(h) = 0.044<br>
# Y Misfit = 0.000<br>
# h Energy = 0.048

# %%
# Importations des modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
# Step 1 : Data input
# System parameters
n = 1
A = 2
G2 = 3
C = 4.9

# Objective
z0 = 4                      # supposed known
h = lambda t : t * (1 - t)  # supposed known

# Regulation parameters
q1 = 1
q2 = 1

# Time discretization
T = 1
N = 200
dt = T / N
t  = np.linspace(0, T, N+1)

# %%
# Import steps 1-5 functions
from functions_n1 import *

# %%
# Simulation for q1 and q2 (step 1 to 4)
print('='*20, f'Simulation for q1={q1} and q2={q2}', '='*20)
print('Proceding ...')
h_true, z_true, y_m, z0_est, z_est, h_est = simulate_n1(A, G2, C, z0, h, T, N, q1, q2)

# %%
# showing true vs estimated
df = pd.DataFrame({"t": t, 'h_true': h_true, 'h_est': h_est[:, 0], 'z_true': z_true, 'z_est': z_est[:, 0], 'y_m': y_m})
print('DF Estimated vs True:')
print(df)

# %%
# Step 5: quick plots
print('-'*70,'\nPlot process: ...')
make_plots(h_true, z_true, z_est, h_est, t, save=True, outdir = "figs")

# %%
# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, energy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
print('-'*70, '\nMetrics:')
print(f"z0 true = {z0}, z0 estimated = {z0_est:.3f}, error = {z0_error:.3f}")
print(f"RMSE(z) = {z_rmse:.3f}")
print(f"RMSE(h) = {h_rmse:.3f}")
print(f"Misfit = {misfit:.3f}")
print(f"Energy = {energy:.3f}")

# %%
# Sensibility of q2: sweep q2, compute solutions & metrics
q1 = 0.001
q2_list = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]     # we start by 0.25 because q2<=0.1 flatten the curves
print('='*19, f'Sensibility of q2 with q1={q1}', '='*19)
print('Proceding ...')
curves_z, curves_h, metrics = q2_sensibility(A, G2, C, z0, h, T, N, q1, q2_list, save=True, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# %%
# Sensibility of q1: sweep q1, compute solutions & metrics
q1_list = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
q2 = 0.25
print('='*19, f'Sensibility of q1 with q2={q2}', '='*19)
print('Proceding ...')
curves_z, curves_h, metrics = q1_sensibility(A, G2, C, z0, h, T, N, q1_list, q2, save=True, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# %%
# Sensibility of q1 and q2: sweep q1 and q2, compute solutions & metrics, and search optimum q1 and q2
# q1_vals = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
# q2_vals = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]

# q1_vals = np.linspace(0.001, 1, 10)
# q2_vals = np.linspace(0.001, 1, 10)

q1_vals = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5]
q2_vals = [0.01, 0.1, 0.25, 0.5, 1, 2, 5]

print('='*24, f'Sensibility of q1 and q2', '='*24)
print('Proceding ...')
curves_z, curves_h, metrics, q1_opt, q2_opt = q12_sensibility_with_optimum(A, G2, C, z0, h, T=1, N=200, 
                                                                           q1_list=q1_vals, q2_list=q2_vals, save=True, outdir="figs")
print('-'*105, '\nMetrics:')
print(metrics)

# %%
# Simulation for q1_opt and q2_opt (step 1 to 4)
print('='*33, f'Simulation for q1*={q1_opt} and q2*={q2_opt}', '='*33)
print('Proceding ...')
h_true, z_true, y_m, z0_est, z_est, h_est = simulate_n1(A, G2, C, z0, h, T, N, q1_opt, q2_opt)

# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, energy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
print('-'*105, '\nMetrics:')
print(f"z0 true = {z0}, z0 estimated = {z0_est:.3f}, error = {z0_error:.3f}")
print(f"RMSE(z) = {z_rmse:.3f}")
print(f"RMSE(h) = {h_rmse:.3f}")
print(f"Misfit = {misfit:.3f}")
print(f"Energy = {energy:.3f}")

# Step 5: quick plots
print('-'*105,'\nPlot process: ...')
make_plots(h_true, z_true, z_est, h_est, t, save=True, outdir = "figs")