## Resolution of a problem of irrigation water estimation using the shooting method, in dimensions $n=1$ et $n=2$

**But.** 

For given $A\in\mathcal{M}_{n}(\mathbb{R})$, $G_2\in\mathcal{M}_{n}(\mathbb{R})$, $C\in\mathcal{M}_{n}(\mathbb{R})$, and $q_1,q_2>0$ considered ponderations, we want to construct the optimal pair
```math
\begin{aligned}
h^\star(t) &= -\tfrac{1}{q_2} G_2^\top p(t),\\
z(0) &= -\tfrac{1}{q_1} p(0)
\end{aligned}
```
where $(z,p)$ is solution of the optimality system  given by the TPBVP:
```math
\begin{aligned}
z'(t) &= A z(t) + G h^\star(t) = A z(t) - \tfrac{1}{q_2} G G^\top p(t), \quad z(0) = -\tfrac{1}{q_1} s,\\
p'(t) &= -A^\top p(t) - C^\top\!\big(C z(t) - y^{\mathrm m}(t)\big), \quad p(T)=0,
\end{aligned}
```
and $s := p(0)\in\mathbb{R}^n$ is **the shoot unknown**. 

---

### Implementation scheme (pseudo-code)
1. Give input data 
    - System: A, G2, C
    - Objective: z0, h as known
    - Time discretization: T, N
    - Regulations: q1, q2
2. Forward z_true and y_m from known z0 and h
    - Integrate $z'(t) = A z(t) + G2 h(t)$ using RK4 over grid t
    - Then compute $Y(t) = C z(t)$
3. Backward p and z from y_m with shooting method on p(0)
    - Define a function that integrate $(z'(t), p'(t))$ with $z(0) = -s/q1$, $p(T) = 0$ under the shoot $s=p(0)$.
    - Define the shooting function $F(s) := p(T; s)$ and find the good $s^\star$ such that $F(s^\star)=0$ by multiple integrations with $s$.
    - Integrate final (z, p) with the good $s^\star$.
4. Reconstruction of the irrigation h
    - $h(t) = -(1/q2) * G2^T p(t)$
    - $z(0) = -(1/q1) * p(0)$
5. Make plots and metrics
    - Metrics: 
        - $z_0 - z_0estimated$
        - $RMSE_z = \sqrt{mean((z_{true} - z_{estimated})^2)}$
        - $RMSE_h = \sqrt{mean((z_{true} - z_{estimated})^2)}$
        - $Y Misfit = \int_0^T \|Cz(t)-y^m(t)\|^2 dt$
        - $H Energy = \int_0^T \|h(t)\|^2 dt$
    - Plots:
        - $z_{true}$ vs $z_{estimated}$
        - $h_{true}$ vs $h_{estimated}$
        - $z_{estimated}$ as function of q2
        - $h_{estimated}$ as function of q2
        - $RMSE_z$ as function of q2
        - $RMSE_h$ as function of q2
        - $Y Misfit$ as function of q2
        - $H Energy$ as function of q2

---

### Input example
*With $n=1$:*
- System: $A=2, G_2=3, C=4.9$
- Objective: $z_{0true}=4, h_{\mathrm{true}}(t)=t(1 - t)$
- Time discretization: $T=1, N=200$
- Regulations: $q_1=1, q_2=1$

*With $n=2$:*
- System: $A = \begin{bmatrix}2 & 0 \\[3pt] 0 & 4\end{bmatrix},\;G_2 = \begin{bmatrix}3 & 0 \\ 0 & 2.5\end{bmatrix}$ and $C = \begin{bmatrix}4.9 & 0 \\ 0 & 9\end{bmatrix}$
- Objective: $z_{0true} = \begin{bmatrix}4 \\ 5\end{bmatrix}$, $
h_{1\mathrm{true}}(t) = 0.5\,t(1-t), \;
h_{2\mathrm{true}}(t) = 1-(t-0.5)^2,
$
- Time discretization: $T=1, N=200$
- Regulations: $q_1=1, q_2=1$

---

### Ouput example (metrics)
*With $n=1$:* <br>
z0 true = 4, z0 estimated = 2.617, error = 1.383<br>
RMSE(z) = 10.881<br>
RMSE(h) = 1.430<br>
Y Misfit = 1.466<br>
h Energy = 2.108

*With $n=2$:* <br>
z0 true = [4 5]', z0 estimated = [ ]', error = 1.634<br>
RMSE(z) = 0.213<br>
RMSE(h) = 1.705<br>
Y Misfit = 3.249<br>
h Energy = 8.619

---

### Files use
The files [simulation_n1.ipynb](simulation_n1.ipynb) and [simulation_n2.ipynb](simulation_n2.ipynb) are Jupyter notebooks that allow step-by-step execution of the script with direct display of the results, while the files [simulation_n1.py](simulation_n1.py) and [simulation_n2.py](simulation_n2.py) allow one-shot execution with final display of the results in the console. The generated figures are saved in the subfolder \verb|figs| in both execution modes. We note that the two-dimensional case can easily be extended to $n>2$

---

### Remark (valable for all $n$)
- **Coût.** On intègre **$n+1$** fois un système de taille **$2n$** (une fois pour $\Beta$, $n$ fois pour les colonnes de $\Alpha$). C’est raisonnable et **trivialement parallélisable** (les $n$ tirs $e_i$ sont indépendants).
- **Conditionnement.** Si $\Alpha$ est mal conditionnée, résoudre $\Alpha s^\star=-\Beta$ en **moindres carrés** (\`np.linalg.lstsq\`) est robuste (cas $q_2$ très grands, observabilité faible, etc.).
- **Cas $m\neq n$.** $G$ peut avoir $m$ colonnes (plusieurs commandes) : la formule $h^\star = -\tfrac{1}{q_2}G^\top p$ et la dynamique $A - \tfrac{1}{q_2}GG^\top$ restent valides.
- **Observations partielles.** $C$ peut être **rectangulaire** ($p\times n$) ; pas besoin que $C=I$.
- **Paramètres variables en temps.** Si $A,G,C$ dépendent de $t$, on remplace $M$ et $b(t)$ par leurs versions **dépendantes du temps** dans l’intégrateur (même schéma).
- **Contraintes ou pénalisations supplémentaires.** Bornes sur $h$, pondérations temporelles, régularisations additionnelles : soit via une **projection** a posteriori, soit en adaptant le **fonctionnel** (tout en conservant un problème quadratique si possible).
