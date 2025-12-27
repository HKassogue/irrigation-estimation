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
# - System: $A=2, G_2=3, C=4.9$
# - Objective: $z_{0true}=4, h_{\mathrm{true}}(t)=t(1 - t)$
# - Time discretization: $T=1, N=200$
# - Regulation: $q_1=0.001, q_2=3.5$

# ---

# ### Ouput example (metrics)
# z0 true = 4, z0 estimated = 4.040606328767682, error = 0.04060632876768189<br>
# RMSE_z    = 0.03310815573827195<br>
# RMSE_h    = 0.010816453903076895<br>
# YMisfit   = 0.026098809107473274<br>
# z0Energy  = 4.040606328767682<br>
# hEnergy   = 0.0326768584236353<br>
# z0HEnergy = 16.359176362501078

# ---


# %%
# Importations des modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# %%
# Import steps 1-5 functions
from functions_n1 import *


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

# Time discretization
T = 1
N = 200
dt = T / N
t  = np.linspace(0, T, N+1)


# %% 
# Baseline simulation: $q_1=1$ and $q_2=1$

# Regulation parameters
q1 = 1
q2 = 1

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
make_plots(h_true, z_true, z_est, h_est, t,  show=False, save=True, file_end=f"q1_{q1}_q2_{q2}", outdir = "figs")


# %%
# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
print('-'*70, '\nMetrics:')
print(f"z0 true = {z0}, z0 estimated = {z0_est}, error = {z0_error}")
print(f"RMSE_z    = {z_rmse}")
print(f"RMSE_h    = {h_rmse}")
print(f"YMisfit   = {misfit}")
print(f"z0HEnergy = {z0_energy**2 + henergy}")
print(f"z0Energy  = {z0_energy}")
print(f"HEnergy   = {henergy}")


# %%
# Sensibility of q2: sweep q2, compute solutions & metrics
q1 = 0.001
q2_list = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]     # we start by 0.25 because q2<=0.1 flatten the curves
q2_uniq_vals = q2_list.copy()   # needed for 3D plots
print('='*19, f'Sensibility of q2 with q1={q1}', '='*19)
print('Proceding ...')
curves_z, curves_h, metrics = q2_sensibility(A, G2, C, z0, h, T, N, q1, q2_list, show=False, save=True, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# Redefining q2 values for elbow detection
print('-'*70, "\nRedefining q2 values for elbow detection:\nIn progress...")
# q2_list = [0.25, 0.5, 1, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 10, 20, 50, 100] #refined
q2_list = [2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]    # refined and zoomed
q2_uniq_vals += q2_list     # needed for 3D plots
curves_z, curves_h, metrics = q2_sensibility(A, G2, C, z0, h, T, N, q1, q2_list, show=False, save=True, file_end='red', outdir='figs')

print('-'*70, "\nElbow-based compromise (L-curve on (RMSE_z, RMSE_h)):")
idx_elbow_rmse = find_Lcurve_elbow(metrics, x_col="z_rmse", y_col="h_rmse", use_log=True, sort_on='q2')
row_star_rmse = metrics.loc[idx_elbow_rmse]
print(row_star_rmse)

print('-'*70, "\nElbow-based compromise (L-curve on (YMisfit, Energy)):")
idx_elbow_misfit = find_Lcurve_elbow(metrics, x_col="misfit", y_col="energy", use_log=True, sort_on='q2')
row_star_misfit = metrics.loc[idx_elbow_misfit]
print(row_star_misfit)


# %%
# Sensibility of q1: sweep q1, compute solutions & metrics
# q1_list = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10]
q1_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.01, 0.1]
q1_uniq_vals = q1_list.copy()     # needed for 3D plots
q2 = 3.5  # choice from L-curve on (RMSE_z, RMSE_h)
print('='*19, f'Sensibility of q1 with q2={q2}', '='*19)
print('Proceding ...')
curves_z, curves_h, metrics = q1_sensibility(A, G2, C, z0, h, T, N, q1_list, q2, show=False, save=True, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# Redefining q1 values for elbow detection
print('-'*70, "\nRedefining q1 values for elbow detection:\nIn progress...")
# q1_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.01, 0.1]    #refined
q1_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]      #refined and zoomed
q1_uniq_vals += q1_list     # needed for 3D plots
curves_z, curves_h, metrics = q1_sensibility(A, G2, C, z0, h, T, N, q1_list, q2, show=False, save=True, file_end="red", outdir = "figs")

print('-'*70, "\nElbow-based compromise (L-curve on (RMSE_z, RMSE_h)):")
idx_elbow_rmse = find_Lcurve_elbow(metrics, x_col="z_rmse", y_col="h_rmse", use_log=True, sort_on='q1')
row_star_rmse = metrics.loc[idx_elbow_rmse]
print(row_star_rmse)


# %%
# 2nd sensibility of q1: sweep q1, compute solutions & metrics
# q1_list = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10]
q1_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.01, 0.1]
q1_uniq_vals = q1_list.copy()     # needed for 3D plots
q2 = 3.25  # choice from L-curve on (YMisfit, z0HEnergy)
print('='*19, f'2nd sensibility of q1 with q2={q2}', '='*19)
print('Proceding ...')
curves_z, curves_h, metrics = q1_sensibility(A, G2, C, z0, h, T, N, q1_list, q2, show=False, save=True, outdir = "figs")
print('-'*70, '\nMetrics:')
print(metrics)

# Redefining q1 values for elbow detection
print('-'*70, "\nRedefining q1 values for elbow detection:\nIn progress...")
# q1_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.01, 0.1]    #refined
q1_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]      #refined and zoomed
q1_uniq_vals += q1_list     # needed for 3D plots
curves_z, curves_h, metrics = q1_sensibility(A, G2, C, z0, h, T, N, q1_list, q2, show=False, save=True, file_end="red", outdir = "figs")

print('-'*70, "\nElbow-based compromise (L-curve on (YMisfit, Energy)):")
idx_elbow_misfit = find_Lcurve_elbow(metrics, x_col="misfit", y_col="energy", use_log=True, sort_on='q1')
row_star_misfit = metrics.loc[idx_elbow_misfit]
print(row_star_misfit)


# %%
# Sensibility of q1 and q2: sweep q1 and q2, compute solutions & metrics, and search optimum q1 and q2 as L-curve compromise on RMSE_z/RMSE_h
# q1_vals = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 
#            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 
#            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
#            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#            1]
# q2_vals = [0.25, 0.5, 1.0, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]

q1_vals = list(np.unique(q1_uniq_vals))
q2_vals = list(np.unique(q2_uniq_vals))

print('='*24, f'Sensibility of q1 and q2 with elbow on RMSE_z/RMSE_h', '='*24)
print(q1_uniq_vals)
print(q2_uniq_vals)
print('Proceding ...')
curves_z, curves_h, metrics, q1_opt, q2_opt = q12_sensibility_with_optimum(A, G2, C, z0, h, T=1, N=200, 
                                                                           q1_list=q1_vals, q2_list=q2_vals, elbow_on='rmse', show=False, save=True, file_end="rmse_elbow", outdir="figs")
print('-'*105, '\nMetrics:')
print(metrics)


# %%
# Simulation for q1_opt and q2_opt (step 1 to 4)
print('='*33, f'Simulation for q1*={q1_opt} and q2*={q2_opt}', '='*33)
print('Proceding ...')
h_true, z_true, y_m, z0_est, z_est, h_est = simulate_n1(A, G2, C, z0, h, T, N, q1_opt, q2_opt)

# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
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
make_plots(h_true, z_true, z_est, h_est, t, show=False, save=True, file_end=f"q1_{q1_opt}_q2_{q2_opt}_from_elbow", outdir = "figs")


# %%
# Sensibility of q1 and q2: sweep q1 and q2, compute solutions & metrics, and search optimum q1 and q2 as L-curve compromise on YMisfit/z0HEnergy
print('='*24, f'Sensibility of q1 and q2 with elbow on YMisfit/z0HEnergy', '='*24)
print(q1_uniq_vals)
print(q2_uniq_vals)
print('Proceding ...')
curves_z, curves_h, metrics, q1_opt, q2_opt = q12_sensibility_with_optimum(A, G2, C, z0, h, T=1, N=200, 
                                                                           q1_list=q1_vals, q2_list=q2_vals, elbow_on='misfit', show=False, save=True, file_end="misfit_elbow", outdir="figs")
print('-'*105, '\nMetrics:')
print(metrics)


# %%
# Simulation for q1_opt and q2_opt (step 1 to 4)
print('='*33, f'Simulation for q1*={q1_opt} and q2*={q2_opt}', '='*33)
print('Proceding ...')
h_true, z_true, y_m, z0_est, z_est, h_est = simulate_n1(A, G2, C, z0, h, T, N, q1_opt, q2_opt)

# Step 5: metrics
z0_error, z0_energy, z_rmse, h_rmse, misfit, henergy = compute_metrics(C, z0, h_true, z_true, y_m, z0_est, z_est, h_est, t)
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
make_plots(h_true, z_true, z_est, h_est, t, show=False, save=True, file_end=f"q1_{q1_opt}_q2_{q2_opt}_from_elbow", outdir = "figs")
