# Module 22 — Robotics 2

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** Robotics 1 (module 21) treated the robot as *geometry*: where is the hand, how do I plan a path, where am I on a known map? This module adds the two things that make a robot actually move and actually build its own map: **dynamics** — the forces and torques behind the motion (Robotics 1 never asked *what makes* the joints accelerate) — and **SLAM**, estimating the robot's pose *and* the map at the same time, when neither is given. In between sits **model-based control**: once you know the dynamics, you can cancel them and make a wildly non-linear arm behave like a simple decoupled system. Three pillars — **dynamics, control, SLAM** — each with one project.
>
> **Prior knowledge**: linear algebra, ordinary differential equations, probability, least squares. From this repo the following build directly into it: **module 21** (kinematics, the Jacobian, the Bayes/Kalman filter, particle-filter localisation on a *known* map — SLAM removes the "known"), **module 20** (ICP — the scan-matching front-end that produces SLAM's loop-closure constraints), **module 19** (homogeneous transformations — poses live in SE(2)/SE(3)), **module 14** (LQR/Riccati — the optimal-control view of the controllers here), **module 07** (Bayes filtering — EKF-SLAM is one big EKF). **Module 21 is a mandatory preceding module.**

> **Note on the scope.** As with modules 15–21 no official module description is available; I scoped the content myself along the standard advanced-robotics canon (Siciliano/Sciavicco for dynamics and control, Thrun/Grisetti for SLAM) and as the direct continuation of Robotics 1. **Deliberately without a real robot, without a physics engine and without a SLAM library** (`mujoco`, `g2o`, `gtsam` are all missing here): the teachable, transferable core is the **mathematics** — the manipulator equation and where the mass matrix, Coriolis and gravity terms come from; why inverse-dynamics control *exactly* linearises the system; and how a **pose graph** turns SLAM into one sparse non-linear least-squares problem that a loop closure snaps into consistency. Whoever calls `gtsam.optimize()` does not understand SLAM; whoever assembles the sparse information matrix $H=\sum J^\top\Omega J$ by hand and watches the drift collapse does. All projects are from scratch in `numpy`/`scipy` — CPU seconds.

---

## Contents

1. [Learning objectives](#learning-objectives)
2. [Basics](#basics)
3. [Intermediate](#intermediate)
4. [Advanced topics](#advanced-topics)
5. [Summary / cheat sheet](#summary--cheat-sheet)
6. [Self-test](#self-test)
7. [Literature & sources](#literature--sources)

---

## Learning objectives

After this module you should be able to …

- explain the difference between **kinematics and dynamics** and why control needs the dynamics.
- derive the **manipulator equation** $\mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q) = \boldsymbol\tau$ from the **Lagrangian**, and say what each term is: the **inertia (mass) matrix**, the **Coriolis/centrifugal** term, the **gravity** term.
- name the structural **properties** of the equation (symmetric positive-definite $\mathbf M$, the skew-symmetry of $\dot{\mathbf M}-2\mathbf C$ = passivity) and use them as correctness checks.
- distinguish **forward dynamics** (torques → motion, for simulation) from **inverse dynamics** (motion → torques, for control), and simulate a robot by numerically integrating the forward dynamics.
- design **model-based controllers**: **inverse-dynamics (computed-torque) control**, which exactly linearises and decouples the system, and **PD + gravity compensation**; and place them next to the **LQR** of module 14.
- state the **SLAM problem** (estimate the trajectory *and* the map at once) and the two classic solutions: **EKF-SLAM** (one growing Gaussian over pose + landmarks) and **graph-based SLAM** (a sparse non-linear least-squares problem over a **pose graph**).
- set up and solve **pose-graph optimisation** by **Gauss-Newton**: the SE(2) error function, its Jacobians, the sparse information matrix $\mathbf H=\sum\mathbf J^\top\boldsymbol\Omega\mathbf J$, the anchoring of the gauge freedom, and why a single **loop closure** corrects an entire drifted trajectory.

---

## Basics

### 1. From kinematics to dynamics: what actually moves the robot

Robotics 1 answered *geometric* questions. Forward kinematics maps joint angles to a hand pose; the Jacobian maps joint velocities to hand velocities. But nowhere did we ask the physical question: **given a torque at the motors, how does the robot accelerate?** That is **dynamics**, and it is what you need to (a) **simulate** a robot without a real one and (b) **control** it well.

The distinction in one line:

- **Kinematics**: relations between positions and velocities ($\mathbf x = f(\mathbf q)$, $\dot{\mathbf x}=\mathbf J\dot{\mathbf q}$). No masses, no forces.
- **Dynamics**: the relation between **forces/torques and accelerations** — Newton's second law for a mechanism with many coupled links.

A robot arm is not $n$ independent motors. Moving joint 2 flings joint 3 around (**coupling**); spinning fast throws the links outward (**centrifugal/Coriolis forces**); and gravity pulls on every link differently depending on the pose. All of this is captured by one matrix equation.

### 2. The manipulator equation

For a rigid robot with generalised coordinates $\mathbf q\in\mathbb R^n$ (the joint variables), the equations of motion take the universal form

$$\boxed{\;\mathbf M(\mathbf q)\,\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\,\dot{\mathbf q} + \mathbf g(\mathbf q) = \boldsymbol\tau\;}$$

- $\mathbf M(\mathbf q)\in\mathbb R^{n\times n}$ — the **inertia (mass) matrix**, symmetric and positive definite. It is the rotational analogue of "mass": $\ddot{\mathbf q}=\mathbf M^{-1}(\dots)$, so a heavier/further-out configuration accelerates less for the same torque. It depends on the configuration $\mathbf q$ because the arm's effective inertia changes as it folds and unfolds.
- $\mathbf C(\mathbf q,\dot{\mathbf q})\,\dot{\mathbf q}\in\mathbb R^n$ — the **Coriolis and centrifugal** term. It is **quadratic in the velocities** (terms $\dot q_i\dot q_j$ and $\dot q_i^2$) — these are the forces that only appear when the robot is *moving*, and they are the reason fast motions are hard to control.
- $\mathbf g(\mathbf q)\in\mathbb R^n$ — the **gravity** term: the joint torques needed just to hold the arm up against gravity in configuration $\mathbf q$.
- $\boldsymbol\tau\in\mathbb R^n$ — the applied **joint torques** (the control input).

(A real robot adds friction and external contact forces; we omit them, as is standard for the ideal model.)

### 3. Where it comes from: the Lagrangian

The cleanest derivation is **Lagrangian mechanics**. Define the **Lagrangian** $\mathcal L = T - P$ (kinetic minus potential energy). The **Euler–Lagrange equations**

$$\frac{\mathrm d}{\mathrm dt}\frac{\partial \mathcal L}{\partial \dot q_i} - \frac{\partial \mathcal L}{\partial q_i} = \tau_i,\qquad i=1,\dots,n$$

produce exactly the manipulator equation. The key object is the **kinetic energy**, which is always a quadratic form in the joint velocities:

$$T = \tfrac12\,\dot{\mathbf q}^\top \mathbf M(\mathbf q)\,\dot{\mathbf q}.$$

**That is the definition of the mass matrix** — it *is* the matrix of the kinetic-energy quadratic form. Carrying the Euler–Lagrange derivative through gives:

- $\tfrac{\mathrm d}{\mathrm dt}(\mathbf M\dot{\mathbf q}) = \mathbf M\ddot{\mathbf q} + \dot{\mathbf M}\dot{\mathbf q}$ → the $\mathbf M\ddot{\mathbf q}$ term plus velocity-dependent pieces,
- the $-\partial T/\partial q_i$ term contributes the rest of the velocity-quadratic pieces; together they collect into $\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}$,
- $\partial P/\partial q_i$ gives the gravity term $\mathbf g(\mathbf q)$.

The Coriolis matrix has an explicit formula through the **Christoffel symbols of the first kind**:

$$C_{ij} = \sum_{k=1}^{n} c_{ijk}\,\dot q_k,\qquad c_{ijk} = \tfrac12\Big(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\Big).$$

You never need to memorise this — but it shows that $\mathbf C$ is completely determined by $\mathbf M$ (and its derivatives). The **basic project** derives $\mathbf M$, $\mathbf C$, $\mathbf g$ for the 2-link arm by exactly this route and checks them numerically.

### 4. The worked example: the planar 2-link arm

Take a planar arm, two revolute joints, link lengths $l_1,l_2$, with the link masses lumped as **point masses** $m_1$ (at the end of link 1) and $m_2$ (at the end of link 2), gravity $g$ pointing down. Writing $s_1=\sin q_1$, $c_1=\cos q_1$, $s_{12}=\sin(q_1{+}q_2)$, etc., the derivation of section 3 yields (a standard textbook result):

$$\mathbf M(\mathbf q)=\begin{pmatrix} m_1 l_1^2 + m_2\big(l_1^2 + l_2^2 + 2 l_1 l_2\cos q_2\big) & m_2\big(l_2^2 + l_1 l_2\cos q_2\big)\\[4pt] m_2\big(l_2^2 + l_1 l_2\cos q_2\big) & m_2 l_2^2\end{pmatrix}$$

$$\mathbf C(\mathbf q,\dot{\mathbf q})=\begin{pmatrix} -h\,\dot q_2 & -h\,(\dot q_1+\dot q_2)\\[4pt] h\,\dot q_1 & 0\end{pmatrix},\qquad h = m_2 l_1 l_2\sin q_2$$

$$\mathbf g(\mathbf q)=\begin{pmatrix} (m_1+m_2)\,g\,l_1\cos q_1 + m_2\,g\,l_2\cos(q_1{+}q_2)\\[4pt] m_2\,g\,l_2\cos(q_1{+}q_2)\end{pmatrix}$$

Read the structure off directly: $\mathbf M$ depends only on $q_2$ (the *shape* of the arm, not its orientation in the world) and is symmetric; the coupling term $\propto\cos q_2$ vanishes when the arm is folded at $90°$; the Coriolis coefficient $h\propto\sin q_2$ vanishes when the arm is straight or folded; gravity depends on absolute angles because "up" is fixed in the world. Every one of these facts is a check the basic project runs.

### 5. Forward vs. inverse dynamics

The manipulator equation is read in **two directions**, and both matter:

- **Forward dynamics** — *given torques, find the motion*. Solve for the acceleration
  $$\ddot{\mathbf q} = \mathbf M(\mathbf q)^{-1}\big(\boldsymbol\tau - \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} - \mathbf g(\mathbf q)\big),$$
  then **numerically integrate** $(\mathbf q,\dot{\mathbf q})$ forward in time. This is how you **simulate** a robot (the basic project does exactly this — it is a physics engine in ten lines).
- **Inverse dynamics** — *given a desired motion, find the torques*. Just evaluate the left-hand side:
  $$\boldsymbol\tau = \mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q).$$
  No matrix inversion, no integration — a direct evaluation. This is the heart of model-based **control** (the medium project): if you know what acceleration you want, inverse dynamics tells you the torque that produces it.

**Integrators.** To integrate the forward dynamics you discretise time in steps $\Delta t$. Plain (explicit) Euler $\mathbf q_{t+1}=\mathbf q_t+\dot{\mathbf q}_t\Delta t$ is simple but injects energy and blows up; **semi-implicit (symplectic) Euler** (update the velocity first, then use the *new* velocity for the position) conserves energy far better and is the right default for mechanical systems; **Runge–Kutta 4 (RK4)** is more accurate still. The basic project uses this to demonstrate **energy conservation**: with no gravity and no torque, the total energy $E=T+P$ must stay constant — a stringent test that the dynamics *and* the integrator are correct.

---

## Intermediate

### 6. Model-based control I: PD + gravity compensation

The simplest useful controller for a manipulator is a **PD controller with gravity compensation**. With the position error $\mathbf e = \mathbf q - \mathbf q_d$ (actual minus desired) and a *constant* target $\mathbf q_d$:

$$\boldsymbol\tau = -\mathbf K_p\,\mathbf e - \mathbf K_d\,\dot{\mathbf e} + \mathbf g(\mathbf q).$$

The $\mathbf g(\mathbf q)$ term cancels gravity exactly, so the arm no longer sags; the PD part pulls the error to zero like a spring–damper. One can prove (Lyapunov, with $V=\tfrac12\dot{\mathbf q}^\top\mathbf M\dot{\mathbf q}+\tfrac12\mathbf e^\top\mathbf K_p\mathbf e$) that this is **globally asymptotically stable for regulation** — reaching a fixed setpoint. But it is **not** exact for **tracking** a fast-moving trajectory: the Coriolis and inertia-coupling terms are left uncompensated, so a moving target is followed only with a lag/error that grows with speed. That is what the next controller fixes.

### 7. Model-based control II: inverse-dynamics (computed-torque) control

The signature idea of manipulator control is to **use the model to cancel the non-linearity**. Choose the torque

$$\boxed{\;\boldsymbol\tau = \mathbf M(\mathbf q)\big(\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e\big) + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q)\;}$$

with $\mathbf e=\mathbf q-\mathbf q_d$. Substitute this into the manipulator equation. The $\mathbf C\dot{\mathbf q}$ and $\mathbf g$ terms **cancel exactly**, and because $\mathbf M$ is invertible you are left with

$$\ddot{\mathbf q} = \ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e \quad\Longrightarrow\quad \boxed{\;\ddot{\mathbf e} + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e = \mathbf 0\;}$$

The closed-loop error obeys a **linear, decoupled, second-order** ODE — one independent spring–damper per joint, with **exactly** the eigenvalues you pick through $\mathbf K_p,\mathbf K_d$ (choose them for critical damping, e.g. $\mathbf K_d = 2\sqrt{\mathbf K_p}$). This is **feedback linearisation**: a genuinely non-linear, coupled MIMO system is turned into $n$ trivial linear ones *by cancellation through the model*. It gives near-perfect **trajectory tracking**, whereas PD+gravity only regulates. The price is that you must **know the model** ($\mathbf M,\mathbf C,\mathbf g$) and evaluate it every control step; model error degrades the cancellation (which motivates robust/adaptive control, beyond this module). The **medium project** implements both controllers and measures the tracking-error gap.

**The bridge to module 14 (LQR).** Once feedback linearisation has produced the linear system $\ddot{\mathbf e}=\mathbf u$ (with $\mathbf u=-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e$), *how* you choose the gains is a **linear optimal-control** question — exactly the **LQR** problem of module 14: minimise $\int(\mathbf e^\top\mathbf Q\mathbf e+\mathbf u^\top\mathbf R\mathbf u)\,\mathrm dt$ and the Riccati equation hands you the optimal $\mathbf K_p,\mathbf K_d$. So computed-torque control and LQR compose: the first linearises, the second optimally stabilises the result. **Impedance/compliance control** is the same family with a different goal — instead of tracking a position, you make the arm *behave like* a chosen spring–damper against external forces (essential for contact tasks); it shapes the effective $\mathbf M,\mathbf K_d,\mathbf K_p$ that the environment feels.

### 8. The SLAM problem

Module 21 localised a robot **on a known map** (the landmarks were given). The far harder and more realistic problem is **SLAM — Simultaneous Localisation and Mapping**: the robot is dropped into an **unknown** environment and must estimate **its own trajectory *and* the map at the same time**, from nothing but its odometry and its sensor. This is a **chicken-and-egg** problem: a good map needs a known trajectory (to place the observed features), and a good trajectory needs a known map (to localise against). SLAM solves both **jointly**.

The reason it is possible at all is **loop closure**. As the robot drives, odometry drift makes its estimated trajectory wander away from the truth (module 21 P03 measured exactly this unbounded growth). But when the robot **returns to a place it has seen before** and *recognises* it, that recognition is a hard constraint: "pose 200 and pose 5 are actually at the same spot." That single constraint, propagated back through the whole trajectory, **corrects the entire accumulated drift**. Loop closure is what turns dead-reckoning-with-a-sensor into a globally consistent map.

Two constraint types feed a SLAM system, and this module's earlier work produces both:
- **Odometry constraints** between consecutive poses — from the motion model (module 21).
- **Loop-closure constraints** between non-consecutive poses that observe the same place — produced by matching the current sensor scan against an earlier one, i.e. by **ICP from module 20**. ICP returns the relative transform between two scans; that relative transform *is* a loop-closure constraint.

---

## Advanced topics

### 9. EKF-SLAM: SLAM as one big Kalman filter

The first classic solution keeps the module-21 machinery and simply **enlarges the state**. The EKF state vector stacks the robot pose **and every landmark position** seen so far:

$$\mathbf x = \big(\underbrace{x_r,y_r,\theta_r}_{\text{robot}},\; \underbrace{m_{1x},m_{1y}}_{\text{landmark 1}},\;\dots,\; \underbrace{m_{Nx},m_{Ny}}_{\text{landmark }N}\big)^\top,$$

and an EKF (module 21) is run over this joint state: the motion step moves the robot part and inflates its covariance; each landmark observation is a correction step that couples the robot and that landmark. The crucial and beautiful fact is that the **covariance matrix becomes fully correlated** — observing one landmark improves the estimate of *all* of them, through the robot pose. A loop closure (re-seeing an old landmark) then snaps the whole map into place via those correlations.

The catch is **cost**: the covariance is $(3+2N)\times(3+2N)$ and the update is $O(N^2)$ in the number of landmarks. EKF-SLAM is therefore limited to a few hundred landmarks, and it inherits the EKF's fragility to linearisation error and to **wrong data association** (a single mismatched loop closure corrupts the whole filter, irreversibly). These limits are exactly what pushed the field toward the graph formulation.

### 10. Graph-based SLAM: SLAM as sparse non-linear least squares

The modern, dominant view is **graph-based (or "smoothing") SLAM**. Build a **pose graph**:

- **nodes** = the robot poses $\mathbf x_1,\dots,\mathbf x_T$ over the whole trajectory (in 2D, each $\mathbf x_i\in SE(2)$, i.e. $(x,y,\theta)$);
- **edges** = relative-pose **constraints**, each a measurement $\mathbf z_{ij}$ of "where pose $j$ is, seen from pose $i$", with an **information matrix** $\boldsymbol\Omega_{ij}$ (inverse covariance — how much you trust that edge). Sequential edges are odometry; long-range edges are loop closures.

Every edge contributes an **error** — the mismatch between what the edge *says* the relative pose should be and what the current pose estimates *imply*:

$$\mathbf e_{ij}(\mathbf x_i,\mathbf x_j) = \mathbf z_{ij}\ominus\big(\mathbf x_i^{-1}\oplus\mathbf x_j\big),$$

where $\oplus,\ominus$ are composition and difference of poses in $SE(2)$ (module 19's transformations, with the angle wrapped to $(-\pi,\pi]$). SLAM is then a single **non-linear least-squares** problem: find the trajectory that best satisfies *all* constraints,

$$\boxed{\;\mathbf x^\ast = \arg\min_{\mathbf x}\; \sum_{(i,j)}\mathbf e_{ij}(\mathbf x)^\top\,\boldsymbol\Omega_{ij}\,\mathbf e_{ij}(\mathbf x)\;}$$

**Solving it by Gauss–Newton.** Linearise each error around the current estimate, $\mathbf e_{ij}(\mathbf x+\Delta\mathbf x)\approx\mathbf e_{ij}+\mathbf J_{ij}\Delta\mathbf x$, where the Jacobian $\mathbf J_{ij}$ is non-zero **only in the two blocks** belonging to poses $i$ and $j$. The normal equations of the linearised problem are

$$\mathbf H\,\Delta\mathbf x = -\mathbf b,\qquad \mathbf H=\sum_{(i,j)}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij},\qquad \mathbf b=\sum_{(i,j)}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij},$$

then update $\mathbf x\leftarrow\mathbf x\oplus\Delta\mathbf x$ and repeat until convergence. The matrix $\mathbf H$ is the **information matrix** of the whole trajectory. Because each edge touches only two poses, $\mathbf H$ is **sparse** (block-tridiagonal from odometry, plus a few off-diagonal blocks from loop closures) — which is exactly why graph-SLAM scales to millions of poses where EKF-SLAM chokes: you solve a large **sparse** linear system (`scipy.sparse.linalg.spsolve`) instead of inverting a dense covariance.

**Gauge freedom (the anchor).** The cost depends only on *relative* poses, so the whole graph can be translated and rotated freely without changing the error — $\mathbf H$ is singular (rank-deficient by 3 in 2D). You remove this freedom by **fixing one pose** (usually pose 0 at the origin): drop its rows/columns from the linear system, or add a strong prior on it. Without the anchor the solve fails.

**Why one loop closure fixes everything.** Before loop closure the graph is a chain: the Gauss–Newton solution is just the (drifted) odometry, because a chain has no conflicting constraints. Add **one** loop-closure edge saying "pose $T$ coincides with pose $0$", and now the constraints *disagree* — the accumulated drift shows up as a large error on that one edge. Minimising the total cost **distributes that error back over every pose in the loop**, in proportion to the information weights, and the trajectory snaps into a consistent shape. The **final project** builds this end to end and shows the drift collapse quantitatively.

### 11. The full SLAM system and its front-end

A working SLAM system has two halves. The **back-end** is the optimisation of section 10 (or the EKF of section 9) — the mathematics this module builds. The **front-end** turns raw sensor data into the graph's edges: it decides *which* poses are connected and *what* the relative-pose measurement is. Its two jobs are **scan matching** (compute the relative transform between two poses' sensor readings — **ICP, module 20**, or feature matching) and **data association / place recognition** (decide that two scans are the *same place* — the loop-closure detection). The front-end is where SLAM is fragile: a single **false loop closure** (claiming two different places are the same) injects a wrong hard constraint that the back-end will faithfully — and catastrophically — enforce. Modern systems guard against this with **robust cost functions** (Huber, switchable constraints, max-mixtures) that let the optimiser *reject* an edge that disagrees too strongly with the rest, instead of trusting every edge equally.

### 12. Learning-based robotics (brief outlook)

Everything above is **model-based**: you write down the dynamics or the measurement model. The complementary modern thread is to **learn** the controller or the model from data. **Imitation learning** (behavioural cloning) fits a policy to expert demonstrations — simple, but it drifts off the demonstrated distribution (addressed by DAgger). **Reinforcement learning** (modules 13/14) learns control by trial and error and, with domain randomisation, can transfer from simulation to reality (sim-to-real). **Learned dynamics models** replace or correct the analytic $\mathbf M,\mathbf C,\mathbf g$ where they are hard to model (friction, contact, deformable objects), and feed model-based RL / MPC. These methods do not *replace* the mathematics of this module — they build on it: a learned residual is added to an analytic model, an RL policy is trained in a simulator that is exactly the forward dynamics of section 5, and a learned SLAM front-end still feeds the graph optimiser of section 10.

---

## Summary / cheat sheet

**Kinematics vs. dynamics**: kinematics = positions/velocities (no mass); dynamics = torques ↔ accelerations. Control needs the dynamics.

**Manipulator equation**: $\mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q) = \boldsymbol\tau$.
$\mathbf M$ = inertia (SPD, config-dependent); $\mathbf C\dot{\mathbf q}$ = Coriolis/centrifugal (quadratic in $\dot{\mathbf q}$); $\mathbf g$ = gravity; $\boldsymbol\tau$ = joint torques.

**Lagrangian**: $\mathcal L=T-P$, $T=\tfrac12\dot{\mathbf q}^\top\mathbf M\dot{\mathbf q}$ (defines $\mathbf M$); Euler–Lagrange $\frac{\mathrm d}{\mathrm dt}\partial_{\dot q}\mathcal L-\partial_q\mathcal L=\tau$. $\mathbf C$ from Christoffel symbols of $\mathbf M$.

**Properties (checks)**: $\mathbf M=\mathbf M^\top\succ0$; $\dot{\mathbf M}-2\mathbf C$ skew-symmetric (passivity); with $\boldsymbol\tau=0$, no gravity → $E=T+P$ conserved.

**Forward dynamics** (simulate): $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$, then integrate (semi-implicit Euler / RK4).
**Inverse dynamics** (control): $\boldsymbol\tau=\mathbf M\ddot{\mathbf q}+\mathbf C\dot{\mathbf q}+\mathbf g$ (direct evaluation).

**PD + gravity**: $\boldsymbol\tau=-\mathbf K_p\mathbf e-\mathbf K_d\dot{\mathbf e}+\mathbf g(\mathbf q)$ — stable for **regulation**, lags on tracking.
**Computed torque**: $\boldsymbol\tau=\mathbf M(\ddot{\mathbf q}_d-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e)+\mathbf C\dot{\mathbf q}+\mathbf g$ → $\ddot{\mathbf e}+\mathbf K_d\dot{\mathbf e}+\mathbf K_p\mathbf e=0$ (feedback linearisation; exact **tracking**). Gains via LQR (module 14).

**SLAM** = estimate trajectory **and** map jointly (unknown map). Enabled by **loop closure**; constraints from odometry (module 21) + scan matching (ICP, module 20).

**EKF-SLAM**: one EKF over (pose + all landmarks); covariance fully correlated; $O(N^2)$/update, fragile to bad data association.

**Graph SLAM**: nodes = poses $\in SE(2)$, edges = relative constraints $(\mathbf z_{ij},\boldsymbol\Omega_{ij})$. Error $\mathbf e_{ij}=\mathbf z_{ij}\ominus(\mathbf x_i^{-1}\oplus\mathbf x_j)$. Minimise $\sum\mathbf e_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$ by Gauss–Newton: $\mathbf H\Delta\mathbf x=-\mathbf b$, $\mathbf H=\sum\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij}$ (**sparse**), $\mathbf b=\sum\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$. **Anchor pose 0** (gauge freedom). One loop closure corrects the whole trajectory.

**Front-end vs. back-end**: back-end = the optimisation; front-end = scan matching + data association. **False loop closures** are catastrophic → robust cost functions.

---

## Self-test

<details>
<summary><b>1.</b> What is the difference between kinematics and dynamics, and why does control need the dynamics?</summary>

**Kinematics** relates positions and velocities without any notion of mass or force ($\mathbf x=f(\mathbf q)$, $\dot{\mathbf x}=\mathbf J\dot{\mathbf q}$). **Dynamics** relates **torques/forces to accelerations** — Newton's second law for the coupled mechanism. Control needs the dynamics because the actuators apply **torques**, and to make the robot accelerate the way you want (track a trajectory, reject a disturbance) you must know how torque turns into motion, i.e. $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$.
</details>

<details>
<summary><b>2.</b> Name the three terms of the manipulator equation and what each represents.</summary>

$\mathbf M(\mathbf q)\ddot{\mathbf q}+\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}+\mathbf g(\mathbf q)=\boldsymbol\tau$. **$\mathbf M(\mathbf q)$** is the inertia (mass) matrix — symmetric positive definite, configuration-dependent; it maps acceleration to the torque needed to produce it. **$\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}$** is the Coriolis and centrifugal term — quadratic in the joint velocities, the forces that appear only when the robot is moving. **$\mathbf g(\mathbf q)$** is the gravity term — the torques needed to hold the arm up in configuration $\mathbf q$. $\boldsymbol\tau$ is the applied joint torque (the input).
</details>

<details>
<summary><b>3.</b> How is the mass matrix defined via the kinetic energy, and why does it depend on the configuration?</summary>

The kinetic energy of a rigid robot is always a quadratic form in the joint velocities, $T=\tfrac12\dot{\mathbf q}^\top\mathbf M(\mathbf q)\dot{\mathbf q}$, and $\mathbf M(\mathbf q)$ is *by definition* the matrix of that form (hence symmetric and positive definite). It depends on $\mathbf q$ because the effective inertia of the mechanism changes with its shape: a stretched-out arm has a very different rotational inertia about the base than a folded one, so the same torque produces a different acceleration depending on the configuration.
</details>

<details>
<summary><b>4.</b> Distinguish forward and inverse dynamics and give the use of each.</summary>

**Forward dynamics**: given the torques, compute the acceleration $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$ and integrate it forward — this is **simulation** (predicting how the robot moves). **Inverse dynamics**: given a desired motion $(\mathbf q,\dot{\mathbf q},\ddot{\mathbf q})$, evaluate $\boldsymbol\tau=\mathbf M\ddot{\mathbf q}+\mathbf C\dot{\mathbf q}+\mathbf g$ directly — this is the core of **control** (computing the torque that produces a wanted acceleration). Forward dynamics needs a matrix solve and an integrator; inverse dynamics is a direct evaluation.
</details>

<details>
<summary><b>5.</b> Why does inverse-dynamics (computed-torque) control linearise the system, and what is the resulting error dynamics?</summary>

Choosing $\boldsymbol\tau=\mathbf M(\ddot{\mathbf q}_d-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e)+\mathbf C\dot{\mathbf q}+\mathbf g$ and substituting it into the manipulator equation makes the $\mathbf C\dot{\mathbf q}$ and $\mathbf g$ terms cancel exactly; dividing out the invertible $\mathbf M$ leaves $\ddot{\mathbf q}=\ddot{\mathbf q}_d-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e$, i.e. $\ddot{\mathbf e}+\mathbf K_d\dot{\mathbf e}+\mathbf K_p\mathbf e=\mathbf 0$. The closed loop is a **linear, decoupled** second-order system (one spring–damper per joint) with eigenvalues chosen through $\mathbf K_p,\mathbf K_d$ — this is feedback linearisation, and it gives exact trajectory tracking (unlike PD+gravity, which only regulates). The cost is needing an accurate model.
</details>

<details>
<summary><b>6.</b> What makes SLAM harder than the localisation of module 21, and what makes it solvable?</summary>

Module 21 localised against a **known** map (given landmarks). SLAM has **no map**: it must estimate the trajectory **and** the map simultaneously — a chicken-and-egg problem, because mapping needs a known trajectory and localisation needs a known map. It is solvable because of **loop closure**: when the robot recognises a previously visited place, that recognition is a constraint linking a late pose to an early one, and propagating that constraint back through the trajectory corrects the accumulated odometry drift, yielding a globally consistent estimate.
</details>

<details>
<summary><b>7.</b> In graph-based SLAM, what are the nodes and edges, and what optimisation problem is solved?</summary>

**Nodes** are the robot poses along the trajectory (each $\in SE(2)$ in 2D). **Edges** are relative-pose constraints between two poses, each with a measurement $\mathbf z_{ij}$ and an information matrix $\boldsymbol\Omega_{ij}$ (odometry edges between consecutive poses, loop-closure edges between non-consecutive ones). Each edge has an error $\mathbf e_{ij}=\mathbf z_{ij}\ominus(\mathbf x_i^{-1}\oplus\mathbf x_j)$, and SLAM solves the non-linear least-squares problem $\min_{\mathbf x}\sum_{ij}\mathbf e_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$ — the trajectory that best satisfies all constraints at once.
</details>

<details>
<summary><b>8.</b> Write the Gauss–Newton normal equations for pose-graph SLAM and explain why $\mathbf H$ is sparse.</summary>

Linearising each error ($\mathbf e_{ij}(\mathbf x+\Delta\mathbf x)\approx\mathbf e_{ij}+\mathbf J_{ij}\Delta\mathbf x$) gives $\mathbf H\Delta\mathbf x=-\mathbf b$ with $\mathbf H=\sum_{ij}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij}$ and $\mathbf b=\sum_{ij}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$; you solve for $\Delta\mathbf x$, update $\mathbf x\leftarrow\mathbf x\oplus\Delta\mathbf x$ and iterate. $\mathbf H$ is **sparse** because each edge's Jacobian $\mathbf J_{ij}$ is non-zero only in the two blocks for poses $i$ and $j$, so $\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij}$ fills only those $2\times2$ block positions. Odometry gives a block-tridiagonal structure; loop closures add a few off-diagonal blocks. This sparsity is why graph-SLAM scales far beyond EKF-SLAM.
</details>

<details>
<summary><b>9.</b> What is the gauge freedom in pose-graph SLAM and how do you handle it?</summary>

The cost depends only on **relative** poses, so translating or rotating the entire graph leaves every error unchanged — the solution is determined only up to a global rigid transform. Consequently $\mathbf H$ is singular (rank-deficient by 3 in 2D, the dimension of $SE(2)$). You remove this freedom by **anchoring one pose** — fixing pose 0 at the origin (dropping its variables from the linear system, or adding a strong prior on it). Without the anchor the linear solve is under-determined and fails.
</details>

<details>
<summary><b>10.</b> Why is a single false loop closure so dangerous, and what defends against it?</summary>

A loop closure is a **hard relative-pose constraint** with a high information weight. A **false** one (declaring two different places to be the same) tells the optimiser that two poses that are genuinely far apart coincide; the back-end faithfully enforces this by dragging the whole trajectory to satisfy it, corrupting the map — and because least squares trusts every edge, it cannot recover on its own. The defence is **robust cost functions** (Huber loss, switchable constraints, max-mixtures) that let the optimiser **down-weight or reject** an edge whose error is grossly inconsistent with the rest of the graph, rather than treating all edges as equally trustworthy.
</details>

---

## Literature & sources

**Textbooks — dynamics & control**
- **Siciliano, Sciavicco, Villani & Oriolo, *Robotics: Modelling, Planning and Control*** (Springer). Chapters on Lagrangian dynamics and on motion/force control cover sections 2–7 rigorously. *In-depth, the standard reference.*
- **Spong, Hutchinson & Vidyasagar, *Robot Modeling and Control***. Especially clear on the Euler–Lagrange derivation of the manipulator equation and on computed-torque control. *Beginner- to intermediate-friendly.*
- **Lynch & Park, *Modern Robotics*** — free PDF + free lecture videos (Northwestern/Coursera). Modern treatment of dynamics (chapter 8) and control (chapter 11). *Free, excellent.*
- **Murray, Li & Sastry, *A Mathematical Introduction to Robotic Manipulation*** — free PDF. Deeper, Lie-group flavour. *Free, advanced.*

**Textbooks & tutorials — SLAM**
- **Thrun, Burgard & Fox, *Probabilistic Robotics*** (MIT Press). The reference for EKF-SLAM, particle-filter SLAM (FastSLAM) and GraphSLAM (chapters 10–11). *Mandatory for the estimation side.*
- **Grisetti, Kümmerle, Stachniss & Burgard, "A Tutorial on Graph-Based SLAM"**, *IEEE ITS Magazine 2010* — the single best short read for section 10; the final project follows its formulation directly. *Free, in-depth, highly recommended.*
- **Cadena et al., "Past, Present, and Future of SLAM: Towards the Robust-Perception Age"**, *IEEE T-RO 2016*. The modern survey (front-end/back-end, robustness). *Free, survey.*

**Key papers**
- **Lu & Milios (1997)** — the origin of pose-graph optimisation for SLAM. *Historical.*
- **Kümmerle et al., "g2o: A General Framework for Graph Optimization"**, *ICRA 2011* — the library that popularised back-end optimisation (what we build a tiny version of). *In-depth.*
- **Dellaert & Kaess, "Factor Graphs for Robot Perception"**, *Foundations and Trends 2017* — the factor-graph / smoothing view (GTSAM). *Advanced.*

**Freely available courses**
- **Cyrill Stachniss (University of Bonn)**, YouTube — outstanding lecture series on Kalman/particle filters, EKF-SLAM and graph-based SLAM. *Free, highly recommended.*
- **Modern Robotics** (Kevin Lynch, Northwestern) — dynamics and control videos. *Free.*

**For hands-on practice**
- The **three projects** build the manipulator dynamics + a forward-dynamics simulator (basic), computed-torque vs. PD+gravity control (medium) and a from-scratch **pose-graph SLAM** optimiser with loop closure (final) — all from scratch, the best way to make the mathematics concrete.

---

> **Next module:** the sequence of dedicated robotics modules ends here; the broader "applications" block of the curriculum (3D point cloud processing done, plus advanced automation, self-aware computing, etc.) continues to draw on the kinematics, dynamics, estimation and optimisation built across modules 19–22.

---
---

# Modul 22 — Robotics 2 (deutsche Fassung)

> **Worum geht es?** Robotics 1 (Modul 21) behandelte den Roboter als *Geometrie*: Wo ist die Hand, wie plane ich einen Weg, wo bin ich auf einer bekannten Karte? Dieses Modul ergänzt die zwei Dinge, die einen Roboter wirklich bewegen und wirklich seine eigene Karte bauen lassen: die **Dynamik** — die Kräfte und Momente hinter der Bewegung (Robotics 1 fragte nie, *was* die Gelenke beschleunigt) — und **SLAM**, die gleichzeitige Schätzung von Pose *und* Karte, wenn beides unbekannt ist. Dazwischen steht die **modellbasierte Regelung**: Kennt man die Dynamik, kann man sie herausrechnen und einen hochgradig nichtlinearen Arm sich wie ein einfaches, entkoppeltes System verhalten lassen. Drei Säulen — **Dynamik, Regelung, SLAM** — je ein Projekt.
>
> **Vorkenntnisse**: lineare Algebra, gewöhnliche Differentialgleichungen, Wahrscheinlichkeitsrechnung, Least Squares. Aus diesem Repo bauen direkt auf: **Modul 21** (Kinematik, Jacobi, Bayes-/Kalman-Filter, Partikelfilter-Lokalisierung auf *bekannter* Karte — SLAM entfernt das „bekannt"), **Modul 20** (ICP — das Scan-Matching-Frontend, das SLAMs Loop-Closure-Constraints liefert), **Modul 19** (homogene Transformationen — Posen leben in SE(2)/SE(3)), **Modul 14** (LQR/Riccati — die optimalregelungstheoretische Sicht auf die Regler hier), **Modul 07** (Bayes-Filterung — EKF-SLAM ist ein großer EKF). **Modul 21 ist Pflicht-Vormodul.**

> **Hinweis zum Zuschnitt.** Wie bei den Modulen 15–21 liegt keine offizielle Modulbeschreibung vor; ich habe den Inhalt selbst zugeschnitten, entlang des Standard-Kanons der fortgeschrittenen Robotik (Siciliano/Sciavicco für Dynamik und Regelung, Thrun/Grisetti für SLAM) und als direkte Fortsetzung von Robotics 1. **Bewusst ohne echten Roboter, ohne Physik-Engine und ohne SLAM-Bibliothek** (`mujoco`, `g2o`, `gtsam` fehlen hier alle): Der lehrbare, übertragbare Kern ist die **Mathematik** — die Manipulatorgleichung und woher Massenmatrix, Coriolis- und Gravitationsterm kommen; warum inverse-Dynamik-Regelung das System *exakt* linearisiert; und wie ein **Posengraph** SLAM in ein einziges dünnbesetztes nichtlineares Least-Squares-Problem verwandelt, das ein Loop Closure in Konsistenz einrasten lässt. Wer `gtsam.optimize()` aufruft, versteht SLAM nicht; wer die dünnbesetzte Informationsmatrix $H=\sum J^\top\Omega J$ von Hand zusammenbaut und den Drift kollabieren sieht, schon. Alle Projekte from scratch in `numpy`/`scipy` — CPU-Sekunden.

---

## Inhalt

1. [Lernziele](#lernziele)
2. [Grundlagen (Basics)](#grundlagen-basics)
3. [Aufbau (Intermediate)](#aufbau-intermediate)
4. [Advanced-Themen](#advanced-themen)
5. [Zusammenfassung / Cheat-Sheet](#zusammenfassung--cheat-sheet)
6. [Selbsttest](#selbsttest)
7. [Literatur & Quellen](#literatur--quellen)

---

## Lernziele

Nach diesem Modul solltest du …

- den Unterschied zwischen **Kinematik und Dynamik** erklären können und warum Regelung die Dynamik braucht.
- die **Manipulatorgleichung** $\mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q) = \boldsymbol\tau$ aus dem **Lagrange-Formalismus** herleiten und jeden Term benennen: die **Massenmatrix (Trägheitsmatrix)**, den **Coriolis-/Zentrifugal**-Term, den **Gravitations**-Term.
- die strukturellen **Eigenschaften** der Gleichung nennen (symmetrische positiv-definite $\mathbf M$, Schiefsymmetrie von $\dot{\mathbf M}-2\mathbf C$ = Passivität) und als Korrektheits-Checks nutzen.
- **Vorwärtsdynamik** (Momente → Bewegung, für die Simulation) von **inverser Dynamik** (Bewegung → Momente, für die Regelung) unterscheiden und einen Roboter durch numerische Integration der Vorwärtsdynamik simulieren.
- **modellbasierte Regler** entwerfen: die **inverse-Dynamik-Regelung (Computed Torque)**, die das System exakt linearisiert und entkoppelt, und **PD + Gravitationskompensation**; und sie neben den **LQR** aus Modul 14 stellen.
- das **SLAM-Problem** formulieren (Trajektorie *und* Karte gleichzeitig schätzen) und die zwei klassischen Lösungen: **EKF-SLAM** (eine wachsende Gaußverteilung über Pose + Landmarken) und **graphbasiertes SLAM** (ein dünnbesetztes nichtlineares Least-Squares-Problem über einem **Posengraphen**).
- **Pose-Graph-Optimierung** per **Gauß-Newton** aufstellen und lösen: die SE(2)-Fehlerfunktion, ihre Jacobi-Matrizen, die dünnbesetzte Informationsmatrix $\mathbf H=\sum\mathbf J^\top\boldsymbol\Omega\mathbf J$, das Verankern der Eichfreiheit, und warum ein einziges **Loop Closure** eine ganze verdriftete Trajektorie korrigiert.

---

## Grundlagen (Basics)

### 1. Von der Kinematik zur Dynamik: Was den Roboter tatsächlich bewegt

Robotics 1 beantwortete *geometrische* Fragen. Die Vorwärtskinematik bildet Gelenkwinkel auf eine Handpose ab; die Jacobi-Matrix bildet Gelenkgeschwindigkeiten auf Handgeschwindigkeiten ab. Nirgends aber stellten wir die physikalische Frage: **Gegeben ein Moment an den Motoren, wie beschleunigt der Roboter?** Das ist **Dynamik**, und sie ist es, was man braucht, um (a) einen Roboter **ohne echten** zu **simulieren** und (b) ihn gut zu **regeln**.

Die Unterscheidung in einer Zeile:

- **Kinematik**: Beziehungen zwischen Positionen und Geschwindigkeiten ($\mathbf x = f(\mathbf q)$, $\dot{\mathbf x}=\mathbf J\dot{\mathbf q}$). Keine Massen, keine Kräfte.
- **Dynamik**: die Beziehung zwischen **Kräften/Momenten und Beschleunigungen** — Newtons zweites Gesetz für einen Mechanismus aus vielen gekoppelten Gliedern.

Ein Roboterarm ist nicht $n$ unabhängige Motoren. Gelenk 2 zu bewegen schleudert Gelenk 3 herum (**Kopplung**); schnelles Drehen wirft die Glieder nach außen (**Zentrifugal-/Coriolis-Kräfte**); und die Schwerkraft zieht an jedem Glied je nach Pose anders. All das erfasst eine einzige Matrixgleichung.

### 2. Die Manipulatorgleichung

Für einen starren Roboter mit verallgemeinerten Koordinaten $\mathbf q\in\mathbb R^n$ (den Gelenkvariablen) nehmen die Bewegungsgleichungen die universelle Form an

$$\boxed{\;\mathbf M(\mathbf q)\,\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\,\dot{\mathbf q} + \mathbf g(\mathbf q) = \boldsymbol\tau\;}$$

- $\mathbf M(\mathbf q)\in\mathbb R^{n\times n}$ — die **Massenmatrix (Trägheitsmatrix)**, symmetrisch und positiv definit. Sie ist das rotatorische Analogon der „Masse": $\ddot{\mathbf q}=\mathbf M^{-1}(\dots)$, also beschleunigt eine schwerere/weiter ausgestreckte Konfiguration bei gleichem Moment weniger. Sie hängt von der Konfiguration $\mathbf q$ ab, weil sich die effektive Trägheit des Arms beim Ein- und Ausklappen ändert.
- $\mathbf C(\mathbf q,\dot{\mathbf q})\,\dot{\mathbf q}\in\mathbb R^n$ — der **Coriolis- und Zentrifugal**-Term. Er ist **quadratisch in den Geschwindigkeiten** (Terme $\dot q_i\dot q_j$ und $\dot q_i^2$) — das sind die Kräfte, die nur auftreten, wenn der Roboter sich *bewegt*, und der Grund, warum schnelle Bewegungen schwer zu regeln sind.
- $\mathbf g(\mathbf q)\in\mathbb R^n$ — der **Gravitations**-Term: die Gelenkmomente, die man allein braucht, um den Arm in Konfiguration $\mathbf q$ gegen die Schwerkraft zu halten.
- $\boldsymbol\tau\in\mathbb R^n$ — die anliegenden **Gelenkmomente** (die Stellgröße).

(Ein echter Roboter hat zusätzlich Reibung und externe Kontaktkräfte; wir lassen sie weg, wie beim idealen Modell üblich.)

### 3. Woher sie kommt: der Lagrange-Formalismus

Die sauberste Herleitung ist die **Lagrange-Mechanik**. Definiere die **Lagrange-Funktion** $\mathcal L = T - P$ (kinetische minus potentielle Energie). Die **Euler-Lagrange-Gleichungen**

$$\frac{\mathrm d}{\mathrm dt}\frac{\partial \mathcal L}{\partial \dot q_i} - \frac{\partial \mathcal L}{\partial q_i} = \tau_i,\qquad i=1,\dots,n$$

erzeugen exakt die Manipulatorgleichung. Das Schlüsselobjekt ist die **kinetische Energie**, die stets eine quadratische Form in den Gelenkgeschwindigkeiten ist:

$$T = \tfrac12\,\dot{\mathbf q}^\top \mathbf M(\mathbf q)\,\dot{\mathbf q}.$$

**Das ist die Definition der Massenmatrix** — sie *ist* die Matrix der quadratischen Form der kinetischen Energie. Führt man die Euler-Lagrange-Ableitung durch, ergibt sich:

- $\tfrac{\mathrm d}{\mathrm dt}(\mathbf M\dot{\mathbf q}) = \mathbf M\ddot{\mathbf q} + \dot{\mathbf M}\dot{\mathbf q}$ → der Term $\mathbf M\ddot{\mathbf q}$ plus geschwindigkeitsabhängige Anteile,
- der Term $-\partial T/\partial q_i$ liefert den Rest der geschwindigkeits-quadratischen Anteile; zusammen bündeln sie sich zu $\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}$,
- $\partial P/\partial q_i$ ergibt den Gravitationsterm $\mathbf g(\mathbf q)$.

Die Coriolis-Matrix hat eine explizite Formel über die **Christoffel-Symbole erster Art**:

$$C_{ij} = \sum_{k=1}^{n} c_{ijk}\,\dot q_k,\qquad c_{ijk} = \tfrac12\Big(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\Big).$$

Die muss man nie auswendig lernen — aber sie zeigt, dass $\mathbf C$ vollständig durch $\mathbf M$ (und deren Ableitungen) bestimmt ist. Das **Basic-Projekt** leitet $\mathbf M$, $\mathbf C$, $\mathbf g$ für den 2-Gelenk-Arm genau auf diesem Weg her und prüft sie numerisch.

### 4. Das durchgerechnete Beispiel: der planare 2-Gelenk-Arm

Nimm einen planaren Arm, zwei Drehgelenke, Gliedlängen $l_1,l_2$, die Gliedmassen als **Punktmassen** $m_1$ (am Ende von Glied 1) und $m_2$ (am Ende von Glied 2) konzentriert, Schwerkraft $g$ nach unten. Mit $s_1=\sin q_1$, $c_1=\cos q_1$, $s_{12}=\sin(q_1{+}q_2)$ usw. liefert die Herleitung aus Abschnitt 3 (ein Standard-Lehrbuchergebnis):

$$\mathbf M(\mathbf q)=\begin{pmatrix} m_1 l_1^2 + m_2\big(l_1^2 + l_2^2 + 2 l_1 l_2\cos q_2\big) & m_2\big(l_2^2 + l_1 l_2\cos q_2\big)\\[4pt] m_2\big(l_2^2 + l_1 l_2\cos q_2\big) & m_2 l_2^2\end{pmatrix}$$

$$\mathbf C(\mathbf q,\dot{\mathbf q})=\begin{pmatrix} -h\,\dot q_2 & -h\,(\dot q_1+\dot q_2)\\[4pt] h\,\dot q_1 & 0\end{pmatrix},\qquad h = m_2 l_1 l_2\sin q_2$$

$$\mathbf g(\mathbf q)=\begin{pmatrix} (m_1+m_2)\,g\,l_1\cos q_1 + m_2\,g\,l_2\cos(q_1{+}q_2)\\[4pt] m_2\,g\,l_2\cos(q_1{+}q_2)\end{pmatrix}$$

Man liest die Struktur direkt ab: $\mathbf M$ hängt nur von $q_2$ ab (der *Form* des Arms, nicht seiner Orientierung in der Welt) und ist symmetrisch; der Kopplungsterm $\propto\cos q_2$ verschwindet, wenn der Arm bei $90°$ eingeklappt ist; der Coriolis-Koeffizient $h\propto\sin q_2$ verschwindet, wenn der Arm gestreckt oder eingeklappt ist; die Gravitation hängt von absoluten Winkeln ab, weil „oben" in der Welt fixiert ist. Jede dieser Tatsachen ist ein Check, den das Basic-Projekt durchführt.

### 5. Vorwärts- vs. inverse Dynamik

Die Manipulatorgleichung liest man in **zwei Richtungen**, und beide zählen:

- **Vorwärtsdynamik** — *gegeben Momente, finde die Bewegung*. Nach der Beschleunigung auflösen
  $$\ddot{\mathbf q} = \mathbf M(\mathbf q)^{-1}\big(\boldsymbol\tau - \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} - \mathbf g(\mathbf q)\big),$$
  dann $(\mathbf q,\dot{\mathbf q})$ **numerisch in der Zeit integrieren**. So **simuliert** man einen Roboter (das Basic-Projekt macht genau das — eine Physik-Engine in zehn Zeilen).
- **Inverse Dynamik** — *gegeben eine gewünschte Bewegung, finde die Momente*. Einfach die linke Seite auswerten:
  $$\boldsymbol\tau = \mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q).$$
  Keine Matrixinversion, keine Integration — eine direkte Auswertung. Das ist das Herz der modellbasierten **Regelung** (Medium-Projekt): Weiß man, welche Beschleunigung man will, sagt einem die inverse Dynamik das Moment, das sie erzeugt.

**Integratoren.** Zur Integration der Vorwärtsdynamik diskretisiert man die Zeit in Schritte $\Delta t$. Das schlichte (explizite) Euler-Verfahren $\mathbf q_{t+1}=\mathbf q_t+\dot{\mathbf q}_t\Delta t$ ist einfach, spritzt aber Energie ein und explodiert; das **semi-implizite (symplektische) Euler-Verfahren** (erst die Geschwindigkeit aktualisieren, dann die *neue* Geschwindigkeit für die Position nutzen) erhält die Energie weit besser und ist der richtige Standard für mechanische Systeme; **Runge-Kutta 4 (RK4)** ist noch genauer. Das Basic-Projekt nutzt das, um **Energieerhaltung** zu demonstrieren: ohne Schwerkraft und ohne Moment muss die Gesamtenergie $E=T+P$ konstant bleiben — ein strenger Test dafür, dass Dynamik *und* Integrator korrekt sind.

---

## Aufbau (Intermediate)

### 6. Modellbasierte Regelung I: PD + Gravitationskompensation

Der einfachste brauchbare Regler für einen Manipulator ist ein **PD-Regler mit Gravitationskompensation**. Mit dem Positionsfehler $\mathbf e = \mathbf q - \mathbf q_d$ (Ist minus Soll) und einem *konstanten* Ziel $\mathbf q_d$:

$$\boldsymbol\tau = -\mathbf K_p\,\mathbf e - \mathbf K_d\,\dot{\mathbf e} + \mathbf g(\mathbf q).$$

Der Term $\mathbf g(\mathbf q)$ kompensiert die Schwerkraft exakt, sodass der Arm nicht mehr durchhängt; der PD-Teil zieht den Fehler wie eine Feder-Dämpfer-Kombination auf null. Man kann (per Lyapunov, mit $V=\tfrac12\dot{\mathbf q}^\top\mathbf M\dot{\mathbf q}+\tfrac12\mathbf e^\top\mathbf K_p\mathbf e$) beweisen, dass dies **global asymptotisch stabil für die Regelung** eines festen Sollwerts ist. Aber es ist **nicht** exakt für die **Verfolgung** einer schnell bewegten Trajektorie: Die Coriolis- und Trägheitskopplungsterme bleiben unkompensiert, sodass ein bewegtes Ziel nur mit einem Nachlauf/Fehler verfolgt wird, der mit der Geschwindigkeit wächst. Genau das behebt der nächste Regler.

### 7. Modellbasierte Regelung II: inverse-Dynamik- (Computed-Torque-) Regelung

Die Signatur-Idee der Manipulatorregelung ist, das **Modell zu nutzen, um die Nichtlinearität zu eliminieren**. Wähle das Moment

$$\boxed{\;\boldsymbol\tau = \mathbf M(\mathbf q)\big(\ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e\big) + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q)\;}$$

mit $\mathbf e=\mathbf q-\mathbf q_d$. Setze dies in die Manipulatorgleichung ein. Die Terme $\mathbf C\dot{\mathbf q}$ und $\mathbf g$ **heben sich exakt weg**, und weil $\mathbf M$ invertierbar ist, bleibt

$$\ddot{\mathbf q} = \ddot{\mathbf q}_d - \mathbf K_d\dot{\mathbf e} - \mathbf K_p\mathbf e \quad\Longrightarrow\quad \boxed{\;\ddot{\mathbf e} + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e = \mathbf 0\;}$$

Der geschlossene Regelkreis-Fehler gehorcht einer **linearen, entkoppelten, zweiter-Ordnung**-DGL — ein unabhängiger Feder-Dämpfer pro Gelenk, mit **exakt** den Eigenwerten, die du über $\mathbf K_p,\mathbf K_d$ wählst (wähle sie z. B. für kritische Dämpfung, $\mathbf K_d = 2\sqrt{\mathbf K_p}$). Das ist **Feedback-Linearisierung**: Ein echt nichtlineares, gekoppeltes MIMO-System wird *durch Wegheben mittels des Modells* in $n$ triviale lineare verwandelt. Es liefert nahezu perfekte **Trajektorienverfolgung**, während PD+Gravitation nur regelt. Der Preis: Man muss das **Modell kennen** ($\mathbf M,\mathbf C,\mathbf g$) und jeden Regeltakt auswerten; Modellfehler verschlechtern die Kompensation (was robuste/adaptive Regelung motiviert, jenseits dieses Moduls). Das **Medium-Projekt** implementiert beide Regler und misst die Lücke im Verfolgungsfehler.

**Die Brücke zu Modul 14 (LQR).** Hat die Feedback-Linearisierung erst das lineare System $\ddot{\mathbf e}=\mathbf u$ (mit $\mathbf u=-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e$) erzeugt, ist die Frage, *wie* man die Verstärkungen wählt, eine **lineare optimalregelungs**-Frage — genau das **LQR**-Problem aus Modul 14: minimiere $\int(\mathbf e^\top\mathbf Q\mathbf e+\mathbf u^\top\mathbf R\mathbf u)\,\mathrm dt$, und die Riccati-Gleichung liefert die optimalen $\mathbf K_p,\mathbf K_d$. Computed-Torque-Regelung und LQR verketten sich also: die erste linearisiert, der zweite stabilisiert das Ergebnis optimal. **Impedanz-/Nachgiebigkeitsregelung** ist dieselbe Familie mit anderem Ziel — statt eine Position zu verfolgen, lässt man den Arm sich *verhalten wie* eine gewählte Feder-Dämpfer-Kombination gegenüber externen Kräften (essenziell für Kontaktaufgaben); sie formt die effektiven $\mathbf M,\mathbf K_d,\mathbf K_p$, die die Umgebung spürt.

### 8. Das SLAM-Problem

Modul 21 lokalisierte einen Roboter **auf einer bekannten Karte** (die Landmarken waren gegeben). Das weit schwerere und realistischere Problem ist **SLAM — Simultaneous Localisation and Mapping**: Der Roboter wird in eine **unbekannte** Umgebung gesetzt und muss **seine eigene Trajektorie *und* die Karte gleichzeitig** schätzen, aus nichts als seiner Odometrie und seinem Sensor. Das ist ein **Henne-Ei**-Problem: Eine gute Karte braucht eine bekannte Trajektorie (um die beobachteten Merkmale zu platzieren), und eine gute Trajektorie braucht eine bekannte Karte (um sich dagegen zu lokalisieren). SLAM löst beides **gemeinsam**.

Der Grund, warum es überhaupt möglich ist, ist das **Loop Closure**. Während der Roboter fährt, lässt der Odometriedrift seine geschätzte Trajektorie von der Wahrheit abwandern (Modul 21 P03 hat genau dieses unbeschränkte Wachstum gemessen). Kehrt der Roboter aber an einen **schon gesehenen Ort** zurück und *erkennt* ihn, ist diese Wiedererkennung ein harter Constraint: „Pose 200 und Pose 5 sind eigentlich am selben Ort." Dieser eine Constraint, durch die ganze Trajektorie zurückpropagiert, **korrigiert den gesamten aufgelaufenen Drift**. Das Loop Closure ist es, was Dead Reckoning mit Sensor in eine global konsistente Karte verwandelt.

Zwei Constraint-Typen speisen ein SLAM-System, und die frühere Arbeit dieses Moduls erzeugt beide:
- **Odometrie-Constraints** zwischen aufeinanderfolgenden Posen — aus dem Bewegungsmodell (Modul 21).
- **Loop-Closure-Constraints** zwischen nicht-aufeinanderfolgenden Posen, die denselben Ort beobachten — erzeugt durch das Matchen des aktuellen Sensorscans gegen einen früheren, also durch **ICP aus Modul 20**. ICP liefert die relative Transformation zwischen zwei Scans; diese relative Transformation *ist* ein Loop-Closure-Constraint.

---

## Advanced-Themen

### 9. EKF-SLAM: SLAM als ein großer Kalman-Filter

Die erste klassische Lösung behält die Maschinerie aus Modul 21 und **vergrößert einfach den Zustand**. Der EKF-Zustandsvektor stapelt die Roboterpose **und jede bisher gesehene Landmarkenposition**:

$$\mathbf x = \big(\underbrace{x_r,y_r,\theta_r}_{\text{Roboter}},\; \underbrace{m_{1x},m_{1y}}_{\text{Landmarke 1}},\;\dots,\; \underbrace{m_{Nx},m_{Ny}}_{\text{Landmarke }N}\big)^\top,$$

und ein EKF (Modul 21) läuft über diesen gemeinsamen Zustand: Der Bewegungsschritt bewegt den Roboterteil und bläht seine Kovarianz auf; jede Landmarkenbeobachtung ist ein Korrekturschritt, der Roboter und diese Landmarke koppelt. Die entscheidende und schöne Tatsache ist, dass die **Kovarianzmatrix vollständig korreliert wird** — eine Landmarke zu beobachten verbessert die Schätzung *aller* über die Roboterpose. Ein Loop Closure (eine alte Landmarke wiederzusehen) lässt dann die ganze Karte über diese Korrelationen einrasten.

Der Haken sind die **Kosten**: Die Kovarianz ist $(3+2N)\times(3+2N)$, und der Update ist $O(N^2)$ in der Landmarkenzahl. EKF-SLAM ist daher auf einige Hundert Landmarken beschränkt und erbt die Anfälligkeit des EKF gegenüber Linearisierungsfehlern und **falscher Datenassoziation** (ein einziges falsch zugeordnetes Loop Closure korrumpiert den ganzen Filter, irreversibel). Genau diese Grenzen trieben das Feld zur Graph-Formulierung.

### 10. Graphbasiertes SLAM: SLAM als dünnbesetztes nichtlineares Least Squares

Die moderne, dominierende Sicht ist **graphbasiertes (oder „Smoothing"-) SLAM**. Baue einen **Posengraphen**:

- **Knoten** = die Roboterposen $\mathbf x_1,\dots,\mathbf x_T$ über die ganze Trajektorie (in 2D jeweils $\mathbf x_i\in SE(2)$, also $(x,y,\theta)$);
- **Kanten** = relative-Pose-**Constraints**, jeweils eine Messung $\mathbf z_{ij}$ von „wo Pose $j$ ist, von Pose $i$ aus gesehen", mit einer **Informationsmatrix** $\boldsymbol\Omega_{ij}$ (inverse Kovarianz — wie sehr man dieser Kante traut). Sequentielle Kanten sind Odometrie; weitreichende Kanten sind Loop Closures.

Jede Kante trägt einen **Fehler** bei — die Diskrepanz zwischen dem, was die Kante über die relative Pose *sagt*, und dem, was die aktuellen Posenschätzungen *implizieren*:

$$\mathbf e_{ij}(\mathbf x_i,\mathbf x_j) = \mathbf z_{ij}\ominus\big(\mathbf x_i^{-1}\oplus\mathbf x_j\big),$$

wobei $\oplus,\ominus$ Verkettung und Differenz von Posen in $SE(2)$ sind (die Transformationen aus Modul 19, mit dem Winkel auf $(-\pi,\pi]$ normiert). SLAM ist dann ein einziges **nichtlineares Least-Squares**-Problem: finde die Trajektorie, die *alle* Constraints am besten erfüllt,

$$\boxed{\;\mathbf x^\ast = \arg\min_{\mathbf x}\; \sum_{(i,j)}\mathbf e_{ij}(\mathbf x)^\top\,\boldsymbol\Omega_{ij}\,\mathbf e_{ij}(\mathbf x)\;}$$

**Lösung per Gauß-Newton.** Linearisiere jeden Fehler um die aktuelle Schätzung, $\mathbf e_{ij}(\mathbf x+\Delta\mathbf x)\approx\mathbf e_{ij}+\mathbf J_{ij}\Delta\mathbf x$, wobei die Jacobi $\mathbf J_{ij}$ **nur in den zwei Blöcken** der Posen $i$ und $j$ von null verschieden ist. Die Normalengleichungen des linearisierten Problems sind

$$\mathbf H\,\Delta\mathbf x = -\mathbf b,\qquad \mathbf H=\sum_{(i,j)}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij},\qquad \mathbf b=\sum_{(i,j)}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij},$$

dann Update $\mathbf x\leftarrow\mathbf x\oplus\Delta\mathbf x$ und wiederholen bis zur Konvergenz. Die Matrix $\mathbf H$ ist die **Informationsmatrix** der ganzen Trajektorie. Weil jede Kante nur zwei Posen berührt, ist $\mathbf H$ **dünnbesetzt** (blocktridiagonal aus der Odometrie, plus ein paar Off-Diagonal-Blöcke aus den Loop Closures) — genau darum skaliert Graph-SLAM auf Millionen Posen, wo EKF-SLAM erstickt: Man löst ein großes **dünnbesetztes** lineares System (`scipy.sparse.linalg.spsolve`) statt eine dichte Kovarianz zu invertieren.

**Eichfreiheit (der Anker).** Die Kosten hängen nur von *relativen* Posen ab, also kann man den ganzen Graphen frei verschieben und drehen, ohne den Fehler zu ändern — $\mathbf H$ ist singulär (in 2D um 3 rangdefizient). Man entfernt diese Freiheit, indem man **eine Pose fixiert** (üblicherweise Pose 0 im Ursprung): ihre Zeilen/Spalten aus dem linearen System streichen oder einen starken Prior auf sie legen. Ohne Anker scheitert die Lösung.

**Warum ein Loop Closure alles korrigiert.** Vor dem Loop Closure ist der Graph eine Kette: Die Gauß-Newton-Lösung ist einfach die (verdriftete) Odometrie, weil eine Kette keine widersprüchlichen Constraints hat. Füge **eine** Loop-Closure-Kante hinzu, die sagt „Pose $T$ fällt mit Pose $0$ zusammen", und nun *widersprechen* sich die Constraints — der aufgelaufene Drift zeigt sich als großer Fehler auf dieser einen Kante. Die Minimierung der Gesamtkosten **verteilt diesen Fehler zurück auf jede Pose in der Schleife**, proportional zu den Informationsgewichten, und die Trajektorie rastet in eine konsistente Form ein. Das **Final-Projekt** baut das von Anfang bis Ende und zeigt den Drift-Kollaps quantitativ.

### 11. Das vollständige SLAM-System und sein Frontend

Ein funktionierendes SLAM-System hat zwei Hälften. Das **Backend** ist die Optimierung aus Abschnitt 10 (oder der EKF aus Abschnitt 9) — die Mathematik, die dieses Modul baut. Das **Frontend** verwandelt Rohsensordaten in die Kanten des Graphen: Es entscheidet, *welche* Posen verbunden sind und *was* die relative-Pose-Messung ist. Seine zwei Aufgaben sind **Scan-Matching** (die relative Transformation zwischen den Sensordaten zweier Posen berechnen — **ICP, Modul 20**, oder Feature-Matching) und **Datenassoziation / Ortswiedererkennung** (entscheiden, dass zwei Scans *derselbe Ort* sind — die Loop-Closure-Erkennung). Das Frontend ist, wo SLAM fragil ist: Ein einziges **falsches Loop Closure** (die Behauptung, zwei verschiedene Orte seien derselbe) injiziert einen falschen harten Constraint, den das Backend treu — und katastrophal — durchsetzt. Moderne Systeme schützen sich dagegen mit **robusten Kostenfunktionen** (Huber, Switchable Constraints, Max-Mixtures), die den Optimierer eine Kante *ablehnen* lassen, die zu stark mit dem Rest widerspricht, statt jeder Kante gleich zu trauen.

### 12. Lernbasierte Robotik (kurzer Ausblick)

Alles bisher ist **modellbasiert**: Man schreibt die Dynamik oder das Messmodell hin. Der komplementäre moderne Strang ist, den Regler oder das Modell aus Daten zu **lernen**. **Imitation Learning** (Behavioural Cloning) passt eine Policy an Experten-Demonstrationen an — einfach, aber sie driftet von der demonstrierten Verteilung ab (behoben durch DAgger). **Reinforcement Learning** (Module 13/14) lernt Regelung durch Versuch und Irrtum und kann mit Domain Randomisation von der Simulation in die Realität übertragen (Sim-to-Real). **Gelernte Dynamikmodelle** ersetzen oder korrigieren die analytischen $\mathbf M,\mathbf C,\mathbf g$, wo sie schwer zu modellieren sind (Reibung, Kontakt, verformbare Objekte), und speisen modellbasiertes RL / MPC. Diese Methoden *ersetzen* die Mathematik dieses Moduls nicht — sie bauen darauf auf: Ein gelerntes Residuum wird zu einem analytischen Modell addiert, eine RL-Policy wird in einem Simulator trainiert, der exakt die Vorwärtsdynamik aus Abschnitt 5 ist, und ein gelerntes SLAM-Frontend speist immer noch den Graph-Optimierer aus Abschnitt 10.

---

## Zusammenfassung / Cheat-Sheet

**Kinematik vs. Dynamik**: Kinematik = Positionen/Geschwindigkeiten (keine Masse); Dynamik = Momente ↔ Beschleunigungen. Regelung braucht die Dynamik.

**Manipulatorgleichung**: $\mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf g(\mathbf q) = \boldsymbol\tau$.
$\mathbf M$ = Trägheit (SPD, konfigurationsabhängig); $\mathbf C\dot{\mathbf q}$ = Coriolis/Zentrifugal (quadratisch in $\dot{\mathbf q}$); $\mathbf g$ = Gravitation; $\boldsymbol\tau$ = Gelenkmomente.

**Lagrange**: $\mathcal L=T-P$, $T=\tfrac12\dot{\mathbf q}^\top\mathbf M\dot{\mathbf q}$ (definiert $\mathbf M$); Euler-Lagrange $\frac{\mathrm d}{\mathrm dt}\partial_{\dot q}\mathcal L-\partial_q\mathcal L=\tau$. $\mathbf C$ aus den Christoffel-Symbolen von $\mathbf M$.

**Eigenschaften (Checks)**: $\mathbf M=\mathbf M^\top\succ0$; $\dot{\mathbf M}-2\mathbf C$ schiefsymmetrisch (Passivität); mit $\boldsymbol\tau=0$, ohne Gravitation → $E=T+P$ erhalten.

**Vorwärtsdynamik** (simulieren): $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$, dann integrieren (semi-implizites Euler / RK4).
**Inverse Dynamik** (regeln): $\boldsymbol\tau=\mathbf M\ddot{\mathbf q}+\mathbf C\dot{\mathbf q}+\mathbf g$ (direkte Auswertung).

**PD + Gravitation**: $\boldsymbol\tau=-\mathbf K_p\mathbf e-\mathbf K_d\dot{\mathbf e}+\mathbf g(\mathbf q)$ — stabil für **Regelung**, nachlaufend beim Verfolgen.
**Computed Torque**: $\boldsymbol\tau=\mathbf M(\ddot{\mathbf q}_d-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e)+\mathbf C\dot{\mathbf q}+\mathbf g$ → $\ddot{\mathbf e}+\mathbf K_d\dot{\mathbf e}+\mathbf K_p\mathbf e=0$ (Feedback-Linearisierung; exakte **Verfolgung**). Verstärkungen per LQR (Modul 14).

**SLAM** = Trajektorie **und** Karte gemeinsam schätzen (unbekannte Karte). Ermöglicht durch **Loop Closure**; Constraints aus Odometrie (Modul 21) + Scan-Matching (ICP, Modul 20).

**EKF-SLAM**: ein EKF über (Pose + alle Landmarken); Kovarianz vollständig korreliert; $O(N^2)$/Update, fragil bei schlechter Datenassoziation.

**Graph-SLAM**: Knoten = Posen $\in SE(2)$, Kanten = relative Constraints $(\mathbf z_{ij},\boldsymbol\Omega_{ij})$. Fehler $\mathbf e_{ij}=\mathbf z_{ij}\ominus(\mathbf x_i^{-1}\oplus\mathbf x_j)$. Minimiere $\sum\mathbf e_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$ per Gauß-Newton: $\mathbf H\Delta\mathbf x=-\mathbf b$, $\mathbf H=\sum\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij}$ (**dünnbesetzt**), $\mathbf b=\sum\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$. **Pose 0 verankern** (Eichfreiheit). Ein Loop Closure korrigiert die ganze Trajektorie.

**Frontend vs. Backend**: Backend = die Optimierung; Frontend = Scan-Matching + Datenassoziation. **Falsche Loop Closures** sind katastrophal → robuste Kostenfunktionen.

---

## Selbsttest

<details>
<summary><b>1.</b> Was ist der Unterschied zwischen Kinematik und Dynamik, und warum braucht Regelung die Dynamik?</summary>

**Kinematik** verknüpft Positionen und Geschwindigkeiten ohne jeden Begriff von Masse oder Kraft ($\mathbf x=f(\mathbf q)$, $\dot{\mathbf x}=\mathbf J\dot{\mathbf q}$). **Dynamik** verknüpft **Momente/Kräfte mit Beschleunigungen** — Newtons zweites Gesetz für den gekoppelten Mechanismus. Regelung braucht die Dynamik, weil die Aktoren **Momente** aufbringen, und um den Roboter so beschleunigen zu lassen, wie man will (eine Trajektorie verfolgen, eine Störung ausregeln), muss man wissen, wie Moment in Bewegung übergeht, also $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$.
</details>

<details>
<summary><b>2.</b> Nenne die drei Terme der Manipulatorgleichung und was jeder darstellt.</summary>

$\mathbf M(\mathbf q)\ddot{\mathbf q}+\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}+\mathbf g(\mathbf q)=\boldsymbol\tau$. **$\mathbf M(\mathbf q)$** ist die Trägheits- (Massen-)Matrix — symmetrisch positiv definit, konfigurationsabhängig; sie bildet Beschleunigung auf das dafür nötige Moment ab. **$\mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q}$** ist der Coriolis- und Zentrifugalterm — quadratisch in den Gelenkgeschwindigkeiten, die Kräfte, die nur bei Bewegung auftreten. **$\mathbf g(\mathbf q)$** ist der Gravitationsterm — die Momente, um den Arm in Konfiguration $\mathbf q$ zu halten. $\boldsymbol\tau$ ist das anliegende Gelenkmoment (die Stellgröße).
</details>

<details>
<summary><b>3.</b> Wie ist die Massenmatrix über die kinetische Energie definiert, und warum hängt sie von der Konfiguration ab?</summary>

Die kinetische Energie eines starren Roboters ist stets eine quadratische Form in den Gelenkgeschwindigkeiten, $T=\tfrac12\dot{\mathbf q}^\top\mathbf M(\mathbf q)\dot{\mathbf q}$, und $\mathbf M(\mathbf q)$ ist *per Definition* die Matrix dieser Form (daher symmetrisch und positiv definit). Sie hängt von $\mathbf q$ ab, weil sich die effektive Trägheit des Mechanismus mit seiner Form ändert: Ein ausgestreckter Arm hat eine ganz andere Rotationsträgheit um die Basis als ein eingeklappter, sodass dasselbe Moment je nach Konfiguration eine andere Beschleunigung erzeugt.
</details>

<details>
<summary><b>4.</b> Unterscheide Vorwärts- und inverse Dynamik und gib den Nutzen jeder an.</summary>

**Vorwärtsdynamik**: Gegeben die Momente, berechne die Beschleunigung $\ddot{\mathbf q}=\mathbf M^{-1}(\boldsymbol\tau-\mathbf C\dot{\mathbf q}-\mathbf g)$ und integriere sie vorwärts — das ist **Simulation** (vorhersagen, wie der Roboter sich bewegt). **Inverse Dynamik**: Gegeben eine gewünschte Bewegung $(\mathbf q,\dot{\mathbf q},\ddot{\mathbf q})$, werte $\boldsymbol\tau=\mathbf M\ddot{\mathbf q}+\mathbf C\dot{\mathbf q}+\mathbf g$ direkt aus — das ist der Kern der **Regelung** (das Moment berechnen, das eine gewünschte Beschleunigung erzeugt). Vorwärtsdynamik braucht einen Matrix-Solve und einen Integrator; inverse Dynamik ist eine direkte Auswertung.
</details>

<details>
<summary><b>5.</b> Warum linearisiert die inverse-Dynamik- (Computed-Torque-) Regelung das System, und wie lautet die resultierende Fehlerdynamik?</summary>

Wählt man $\boldsymbol\tau=\mathbf M(\ddot{\mathbf q}_d-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e)+\mathbf C\dot{\mathbf q}+\mathbf g$ und setzt es in die Manipulatorgleichung ein, heben sich die Terme $\mathbf C\dot{\mathbf q}$ und $\mathbf g$ exakt weg; Herausdividieren des invertierbaren $\mathbf M$ lässt $\ddot{\mathbf q}=\ddot{\mathbf q}_d-\mathbf K_d\dot{\mathbf e}-\mathbf K_p\mathbf e$, also $\ddot{\mathbf e}+\mathbf K_d\dot{\mathbf e}+\mathbf K_p\mathbf e=\mathbf 0$. Der geschlossene Regelkreis ist ein **lineares, entkoppeltes** System zweiter Ordnung (ein Feder-Dämpfer pro Gelenk) mit über $\mathbf K_p,\mathbf K_d$ gewählten Eigenwerten — das ist Feedback-Linearisierung, und sie liefert exakte Trajektorienverfolgung (anders als PD+Gravitation, das nur regelt). Der Preis ist, ein genaues Modell zu brauchen.
</details>

<details>
<summary><b>6.</b> Was macht SLAM schwerer als die Lokalisierung aus Modul 21, und was macht es lösbar?</summary>

Modul 21 lokalisierte gegen eine **bekannte** Karte (gegebene Landmarken). SLAM hat **keine Karte**: Es muss Trajektorie **und** Karte gleichzeitig schätzen — ein Henne-Ei-Problem, denn Kartierung braucht eine bekannte Trajektorie und Lokalisierung eine bekannte Karte. Lösbar ist es dank des **Loop Closure**: Erkennt der Roboter einen zuvor besuchten Ort wieder, ist diese Wiedererkennung ein Constraint, der eine späte Pose mit einer frühen verbindet, und das Zurückpropagieren dieses Constraints durch die Trajektorie korrigiert den aufgelaufenen Odometriedrift und liefert eine global konsistente Schätzung.
</details>

<details>
<summary><b>7.</b> Was sind im graphbasierten SLAM die Knoten und Kanten, und welches Optimierungsproblem wird gelöst?</summary>

**Knoten** sind die Roboterposen entlang der Trajektorie (in 2D jeweils $\in SE(2)$). **Kanten** sind relative-Pose-Constraints zwischen zwei Posen, jeweils mit einer Messung $\mathbf z_{ij}$ und einer Informationsmatrix $\boldsymbol\Omega_{ij}$ (Odometrie-Kanten zwischen aufeinanderfolgenden Posen, Loop-Closure-Kanten zwischen nicht-aufeinanderfolgenden). Jede Kante hat einen Fehler $\mathbf e_{ij}=\mathbf z_{ij}\ominus(\mathbf x_i^{-1}\oplus\mathbf x_j)$, und SLAM löst das nichtlineare Least-Squares-Problem $\min_{\mathbf x}\sum_{ij}\mathbf e_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$ — die Trajektorie, die alle Constraints zugleich am besten erfüllt.
</details>

<details>
<summary><b>8.</b> Schreibe die Gauß-Newton-Normalengleichungen für Pose-Graph-SLAM auf und erkläre, warum $\mathbf H$ dünnbesetzt ist.</summary>

Das Linearisieren jedes Fehlers ($\mathbf e_{ij}(\mathbf x+\Delta\mathbf x)\approx\mathbf e_{ij}+\mathbf J_{ij}\Delta\mathbf x$) ergibt $\mathbf H\Delta\mathbf x=-\mathbf b$ mit $\mathbf H=\sum_{ij}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij}$ und $\mathbf b=\sum_{ij}\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf e_{ij}$; man löst nach $\Delta\mathbf x$, aktualisiert $\mathbf x\leftarrow\mathbf x\oplus\Delta\mathbf x$ und iteriert. $\mathbf H$ ist **dünnbesetzt**, weil die Jacobi $\mathbf J_{ij}$ jeder Kante nur in den zwei Blöcken der Posen $i$ und $j$ von null verschieden ist, sodass $\mathbf J_{ij}^\top\boldsymbol\Omega_{ij}\mathbf J_{ij}$ nur jene $2\times2$-Blockpositionen füllt. Die Odometrie gibt eine blocktridiagonale Struktur; Loop Closures fügen ein paar Off-Diagonal-Blöcke hinzu. Diese Dünnbesetztheit ist der Grund, warum Graph-SLAM weit über EKF-SLAM hinaus skaliert.
</details>

<details>
<summary><b>9.</b> Was ist die Eichfreiheit im Pose-Graph-SLAM und wie behandelt man sie?</summary>

Die Kosten hängen nur von **relativen** Posen ab, also lässt das Verschieben oder Drehen des ganzen Graphen jeden Fehler unverändert — die Lösung ist nur bis auf eine globale starre Transformation bestimmt. Folglich ist $\mathbf H$ singulär (in 2D um 3 rangdefizient, die Dimension von $SE(2)$). Man entfernt diese Freiheit durch **Verankern einer Pose** — Fixieren von Pose 0 im Ursprung (ihre Variablen aus dem linearen System streichen oder einen starken Prior auf sie legen). Ohne Anker ist der lineare Solve unterbestimmt und scheitert.
</details>

<details>
<summary><b>10.</b> Warum ist ein einziges falsches Loop Closure so gefährlich, und was schützt dagegen?</summary>

Ein Loop Closure ist ein **harter relative-Pose-Constraint** mit hohem Informationsgewicht. Ein **falsches** (das zwei verschiedene Orte für denselben erklärt) sagt dem Optimierer, dass zwei tatsächlich weit entfernte Posen zusammenfallen; das Backend setzt das treu durch, indem es die ganze Trajektorie zieht, um es zu erfüllen, und korrumpiert die Karte — und weil Least Squares jeder Kante traut, kann es sich nicht von selbst erholen. Der Schutz sind **robuste Kostenfunktionen** (Huber-Loss, Switchable Constraints, Max-Mixtures), die den Optimierer eine Kante **herabgewichten oder ablehnen** lassen, deren Fehler grob inkonsistent mit dem Rest des Graphen ist, statt alle Kanten als gleich vertrauenswürdig zu behandeln.
</details>

---

## Literatur & Quellen

**Lehrbücher — Dynamik & Regelung**
- **Siciliano, Sciavicco, Villani & Oriolo, *Robotics: Modelling, Planning and Control*** (Springer). Die Kapitel zur Lagrange-Dynamik und zur Bewegungs-/Kraftregelung decken Abschnitt 2–7 rigoros ab. *Vertiefend, die Standardreferenz.*
- **Spong, Hutchinson & Vidyasagar, *Robot Modeling and Control***. Besonders klar bei der Euler-Lagrange-Herleitung der Manipulatorgleichung und der Computed-Torque-Regelung. *Einsteiger- bis mittelfreundlich.*
- **Lynch & Park, *Modern Robotics*** — freies PDF + kostenlose Vorlesungsvideos (Northwestern/Coursera). Moderne Behandlung von Dynamik (Kap. 8) und Regelung (Kap. 11). *Kostenlos, exzellent.*
- **Murray, Li & Sastry, *A Mathematical Introduction to Robotic Manipulation*** — freies PDF. Tiefer, Lie-Gruppen-orientiert. *Kostenlos, fortgeschritten.*

**Lehrbücher & Tutorials — SLAM**
- **Thrun, Burgard & Fox, *Probabilistic Robotics*** (MIT Press). Die Referenz für EKF-SLAM, Partikelfilter-SLAM (FastSLAM) und GraphSLAM (Kap. 10–11). *Pflicht für die Schätz-Seite.*
- **Grisetti, Kümmerle, Stachniss & Burgard, „A Tutorial on Graph-Based SLAM"**, *IEEE ITS Magazine 2010* — der beste kurze Text zu Abschnitt 10; das Final-Projekt folgt seiner Formulierung direkt. *Kostenlos, vertiefend, sehr empfohlen.*
- **Cadena et al., „Past, Present, and Future of SLAM: Towards the Robust-Perception Age"**, *IEEE T-RO 2016*. Der moderne Überblick (Frontend/Backend, Robustheit). *Kostenlos, Survey.*

**Schlüssel-Papers**
- **Lu & Milios (1997)** — der Ursprung der Pose-Graph-Optimierung für SLAM. *Historisch.*
- **Kümmerle et al., „g2o: A General Framework for Graph Optimization"**, *ICRA 2011* — die Bibliothek, die die Backend-Optimierung populär machte (wovon wir eine winzige Version bauen). *Vertiefend.*
- **Dellaert & Kaess, „Factor Graphs for Robot Perception"**, *Foundations and Trends 2017* — die Faktorgraph-/Smoothing-Sicht (GTSAM). *Fortgeschritten.*

**Frei verfügbare Kurse**
- **Cyrill Stachniss (Uni Bonn)**, YouTube — herausragende Vorlesungsreihe zu Kalman-/Partikelfiltern, EKF-SLAM und graphbasiertem SLAM. *Kostenlos, sehr empfohlen.*
- **Modern Robotics** (Kevin Lynch, Northwestern) — Videos zu Dynamik und Regelung. *Kostenlos.*

**Zum Ausprobieren**
- Die **drei Projekte** bauen die Manipulator-Dynamik + einen Vorwärtsdynamik-Simulator (basic), Computed-Torque- vs. PD+Gravitations-Regelung (medium) und einen from-scratch **Pose-Graph-SLAM**-Optimierer mit Loop Closure (final) — alles from scratch, der beste Weg, die Mathematik konkret zu machen.

---

> **Nächstes Modul:** Die Folge der eigenen Robotik-Module endet hier; der breitere „Anwendungs"-Block des Curriculums (3D Point Cloud Processing fertig, dazu Advanced Automation, Self-aware Computing usw.) greift weiter auf die Kinematik, Dynamik, Schätzung und Optimierung zurück, die über die Module 19–22 aufgebaut wurden.
