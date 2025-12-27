# %% [markdown]
# ## Resolution of a problem of irrigation water estimation using the shooting method in dimension 2

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
#         - $z0Error = z_0 - z_{0est}$
#         - $RMSE_z = \sqrt{mean((z_{true} - z_{est})^2)}$
#         - $RMSE_h = \sqrt{mean((z_{true} - z_{est})^2)}$
#         - $YMisfit = \int_0^T \|Cz(t)-y^m(t)\|^2 dt$
#         - $z_0Energy = \|z_0\|^2$
#         - $HEnergy = \int_0^T \|h(t)\|^2 dt$
#         - $z_0HEnergy = z_0Energy^2 + HEnergy$
#     - Plots:
#         - $z_{true}$ vs $z_{est}$
#         - $h_{true}$ vs $h_{est}$
#         - $z_{est}$ as function of q1 and/or q2
#         - $h_{est}$ as function of q1 and/or q2
#         - $z0Error$ as function of q1 and/or q2
#         - $RMSE_z$ as function of q1 and/or q2
#         - $RMSE_h$ as function of q1 and/or q2
#         - $YMisfit$ as function of q1 and/or q2
#         - $z_0Energy$ as function of q1 and/or q2
#         - $HEnergy$ as function of q1 and/or q2
#         - $z_0HEnergy$ as function of q1 and/or q2

# ---

# ### Input example
# - System: $A = \begin{bmatrix}-0.06 & 0.01 \\[3pt] 0.01  & -0.06 \end{bmatrix},\;G_2 = \begin{bmatrix}0.1 & 0 \\ 0 & 0.1\end{bmatrix}$ and $C = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$
# - Objective: $z_{0true} = \begin{bmatrix}0.4 \\ 0.7\end{bmatrix}$, $h_{\mathrm{true}}(t) = \begin{bmatrix}t(1-t) \\ 0.5\,t(1-t)\end{bmatrix}$
# - Time discretization: $T=1, N=200$
# - Regulation: $q_1=10^{-5}, q_2=10^{-5}$

# ---

# ### Ouput example (metrics)
# z0 true = [0.4 0.7], z0 estimated = [0.39997393 0.69982875], z0 error = 0.00017322682302418966<br>
# RMSE_z    = 4.7125821331296974e-05<br>
# RMSE_h    = 0.00826870734086433<br>
# YMisfit   = 4.3127306750436686e-09<br>
# z0Energy  = 0.806064151243763<br>
# HEnergy   = 0.04102883167179921<br>
# z0HEnergy = 0.6857054886574083

# ---


# %%
# Importations des modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# %%
# Step 1: Data input
# System parameters
n = 2
A  = np.array([
    [-0.06, 0.01],
    [0.01, -0.06]])
G2 = np.diag([0.1, 0.1])
C  = np.array([
    [1.0, 0],
    [ 0, 1.0]])

# Objective
z0 = np.array([0.4, 0.7])           # supposed known
h1 = lambda t : t * (1 - t)         # supposed known
h2 = lambda t : 0.5 * t * (1 - t)   # supposed known
# h1 = lambda t : 0.8 * np.exp(-((t - 0.2) / 0.06)**2) + 0.9 * np.exp(-((t - 0.6) / 0.08)**2)         # supposed known
# h2 = lambda t : 0.5 * np.exp(-((t - 0.3) / 0.07)**2) + 0.7 * np.exp(-((t - 0.75) / 0.06)**2)        # supposed known

# Time discretization
T = 1
N = 200
dt = T / N
t  = np.linspace(0, T, N+1)

# %%
# Import steps 1-5 functions
from functions_n2 import *


# %%
#  Baseline simulation: $q_1=1$ and $q_2=1$

# Regulation parameters
q1 = 1
q2 = 1 

# Simulation for q1 and q2 (step 1 to 4)
print('='*50, f'Simulation for q1={q1} and q2={q2}', '='*50)
print('Proceding ...')
h_true, z_true, y_m, z0_est, z_est, h_est = simulate_n2(A, G2, C, z0, h1, h2, T, N, q1, q2)


# %%
# showing true vs estimated
df = pd.DataFrame({"t": t})
df[["h1_true", "h2_true"]] = h_true
df[["h1_est", "h2_est"]] = h_est
df[["z1_true", "z2_true"]] = z_true
df[["z1_est", "z2_est"]] = z_est
df[["y1", "y2"]] = y_m
print('DF Estimated vs True:')
print(df)


# %%
# Step 5: quick plots
print('-'*100,'\nPlot process: ...')
make_plots(h_true, z_true, z_est, h_est, t, q1, q2, save=True, outdir = "figs")


# %%
# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
print('-'*100, '\nMetrics:')
print(f"z0 true = {z0}, z0 estimated = {z0_est}, error = {z0_error}")
print(f"RMSE_z    = {z_rmse}")
print(f"RMSE_h    = {h_rmse}")
print(f"YMisfit   = {misfit}")
print(f"z0HEnergy = {z0_energy**2 + henergy}")
print(f"z0Energy  = {z0_energy}")
print(f"HEnergy   = {henergy}")


# %%
# Sensibility of q2: sweep q2, compute solutions & metrics
q1 = 1e-5   #0.01
q2_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
q2_uniq_vals = q2_list.copy()   # needed for 3D plots
print('='*20, f'Sensibility of q2 with q1={q1}', '='*20)
print('Proceding ...')
curves_z1, curves_z2, curves_h1, curves_h2, metrics = q2_sensibility(A, G2, C, z0, h1, h2, T, N, q1, q2_list, show=False, save=True, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# Redefining q2 values for elbow detection
q2_list = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,  0.0001]    # refined and zoomed
q2_uniq_vals += q2_list     # needed for 3D plots

print('-'*70, "\nElbow-based compromise (L-curve on (RMSE_z, RMSE_h)):\nIn progress...")
curves_z1, curves_z2, curves_h1, curves_h2, metrics = q2_sensibility(A, G2, C, z0, h1, h2, T, N, q1, q2_list, show=False, save=False, outdir = "figs")
idx_elbow_rmse = find_Lcurve_elbow(metrics, x_col="z_rmse", y_col="h_rmse", use_log=True, sort_on='q2')
row_star_rmse = metrics.loc[idx_elbow_rmse]
print(row_star_rmse)

print('-'*70, "\nElbow-based compromise (L-curve on (YMisfit, Energy)):\nIn progress...")
curves_z1, curves_z2, curves_h1, curves_h2, metrics = q2_sensibility(A, G2, C, z0, h1, h2, T, N, q1, q2_list, show=False, save=False, outdir = "figs")
idx_elbow_misfit = find_Lcurve_elbow(metrics, x_col="misfit", y_col="energy", use_log=True, sort_on='q2')
row_star_misfit = metrics.loc[idx_elbow_misfit]
print(row_star_misfit)


# %%
# Sensibility of q1: sweep q1, compute solutions & metrics
# q1_list = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
q1_list = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
q1_uniq_vals = q1_list.copy()     # needed for 3D plots
q2 = 3e-5     # choice from L-curve on (RMSE_z, RMSE_h)
print('='*19, f'Sensibility of q1 with q2={q2}', '='*19)
print('Proceding ...')
curves_z1, curves_z2, curves_h1, curves_h2, metrics = q1_sensibility(A, G2, C, z0, h1, h2, T, N, q1_list, q2, show=False, save=True, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# Redefining q1 values for elbow detection
# q1_list = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
q1_list = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
q1_uniq_vals += q1_list     # needed for 3D plots

print('-'*70, "\nElbow-based compromise (L-curve on (RMSE_z, RMSE_h)):\nIn progress...")
curves_z1, curves_z2, curves_h1, curves_h2, metrics = q1_sensibility(A, G2, C, z0, h1, h2, T, N, q1_list, q2, show=False, save=False, outdir = "figs")
idx_elbow_rmse = find_Lcurve_elbow(metrics, x_col="z_rmse", y_col="h_rmse", use_log=True, sort_on='q1')
row_star_rmse = metrics.loc[idx_elbow_rmse]
print(row_star_rmse)


# %%
# 2nd sensibility of q1: sweep q1, compute solutions & metrics
# q1_list = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
q1_list = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
q1_uniq_vals += q1_list     # needed for 3D plots
q2 = 4e-5  # choice from L-curve on (YMisfit, z0HEnergy)
print('='*19, f'Sensibility of q1 with q2={q2}', '='*19)
print('Proceding ...')
curves_z1, curves_z2, curves_h1, curves_h2, metrics = q1_sensibility(A, G2, C, z0, h1, h2, T, N, q1_list, q2, show=False, save=False, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# Redefining q1 values for elbow detection
# q1_list = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
q1_list = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
q1_uniq_vals += q1_list     # needed for 3D plots

print('-'*70, "\nElbow-based compromise (L-curve on (YMisfit, Energy)):\nIn progress...")
curves_z1, curves_z2, curves_h1, curves_h2, metrics = q1_sensibility(A, G2, C, z0, h1, h2, T, N, q1_list, q2, show=False, save=False, outdir = "figs")
idx_elbow_misfit = find_Lcurve_elbow(metrics, x_col="misfit", y_col="energy", use_log=True, sort_on='q1')
row_star_misfit = metrics.loc[idx_elbow_misfit]
print(row_star_misfit)


# %%
# Sensibility of q1 and q2: sweep q1 and q2, compute solutions & metrics, and search optimum q1 and q2 as L-curve compromise on RMSE_z/RMSE_h
q1_vals = list(np.unique(q1_uniq_vals))
q2_vals = list(np.unique(q2_uniq_vals))
print(q1_vals)
print(q2_vals)

print('='*24, f'Sensibility of q1 and q2 with L-curve compromise on RMSE_z/RMSE_h', '='*24)
print('Proceding ...')
curves_z, curves_h, metrics, q1_opt, q2_opt = q12_sensibility_with_optimum(A, G2, C, z0, h1, h2, T=1, N=200, 
                                                                           q1_list=q1_vals, q2_list=q2_vals, elbow_on='rmse', save=True, outdir="figs")
print('-'*105, '\nMetrics:')
print(metrics)


# %%
# Simulation for q1_opt and q2_opt (step 1 to 4)
print('='*33, f'Simulation for q1*={q1_opt} and q2*={q2_opt}', '='*33)
print('Proceding ...')
h_true, z_true, y_m, z0_est, z_est, h_est = simulate_n2(A, G2, C, z0, h1, h2, T, N, q1_opt, q2_opt)

# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, energy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
print('-'*105, '\nMetrics:')
print(f"z0 true = {z0}, z0 estimated = {z0_est}, error = {z0_error}")
print(f"RMSE_z    = {z_rmse}")
print(f"RMSE_h    = {h_rmse}")
print(f"YMisfit   = {misfit}")
print(f"z0HEnergy = {z0_energy**2 + henergy}")
print(f"z0Energy  = {z0_energy}")
print(f"HEnergy   = {henergy}")

# Step 5: quick plots
print('-'*105,'\nPlot process: ...')
make_plots(h_true, z_true, z_est, h_est, t, q1_opt, q2_opt, save=True, outdir = "figs")


# %%
# Sensibility of q1 and q2: sweep q1 and q2, compute solutions & metrics, and search optimum q1 and q2 as L-curve compromise on YMisfit/HEnergy
print('='*24, f'Sensibility of q1 and q2 with L-curve compromise on YMisfit/HEnergy', '='*24)
print('Proceding ...')
curves_z, curves_h, metrics, q1_opt, q2_opt = q12_sensibility_with_optimum(A, G2, C, z0, h1, h2, T=1, N=200, 
                                                                           q1_list=q1_vals, q2_list=q2_vals, elbow_on='misfit', save=True, outdir="figs")
print('-'*105, '\nMetrics:')
print(metrics)


# %%
# Simulation for q1_opt and q2_opt (step 1 to 4)
print('='*33, f'Simulation for q1*={q1_opt} and q2*={q2_opt}', '='*33)
print('Proceding ...')
h_true, z_true, y_m, z0_est, z_est, h_est = simulate_n2(A, G2, C, z0, h1, h2, T, N, q1_opt, q2_opt)

# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, energy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
print('-'*105, '\nMetrics:')
print(f"z0 true = {z0}, z0 estimated = {z0_est}, error = {z0_error}")
print(f"RMSE_z    = {z_rmse}")
print(f"RMSE_h    = {h_rmse}")
print(f"YMisfit   = {misfit}")
print(f"z0HEnergy = {z0_energy**2 + henergy}")
print(f"z0Energy  = {z0_energy}")
print(f"HEnergy   = {henergy}")

# Step 5: quick plots
print('-'*105,'\nPlot process: ...')
make_plots(h_true, z_true, z_est, h_est, t, q1_opt, q2_opt, save=True, outdir = "figs")