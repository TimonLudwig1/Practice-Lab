# Module 19 — 3D User Interfaces

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** As soon as content lies in **three-dimensional space** — in VR/AR, in a CAD program, in a game — the interaction familiar from the 2D desktop (mouse, windows, menus) is no longer enough. A **3D user interface (3DUI)** is an interface through which a human **selects and manipulates objects in 3D space, moves within it and orients themselves**. This module covers the *principles* and the *mathematics* of that interaction: how are the coordinate systems related? How does a virtual pointing ray hit an object? How do you reach a distant object with a short arm? And why is pointing in 3D *fundamentally harder* than in 2D?
>
> **Prior knowledge**: linear algebra (matrices, vectors, the dot product), some 3D geometry. From this repo the following build directly into it: **module 17** (Core XR — rotation mathematics/quaternions, Fitts' law, tracking, cybersickness) and **module 18** (multimodal — reference resolution, pointing ambiguity, user study methodology). Module 17 is a **mandatory preceding module**; much of what follows is its direct continuation.

> **Note on the scope.** As with modules 15–18 no official module description is available; I scoped the content myself, closely along the authoritative reference (Bowman, LaViola, Kruijff, Poupyrev: *3D User Interfaces — Theory and Practice*) and consistently with the XR block of this repo. **Again without concrete hardware** (no VR headset, no tracked controller): the teachable, transferable core is the **transformation mathematics, the interaction techniques as algorithms, the pointing precision models (Fitts in 3D) and the evaluation methodology**. You learn to hold a controller in minutes; understanding *why* ray-casting breaks down with distance and *how* Go-Go extends the arm non-linearly is the master-level competence. The projects simulate pointing and selection realistically with pure geometry/statistics on the CPU.

---

## Contents

1. [Learning objectives](#learning-objectives)
2. [Basics](#basics)
3. [Building up (intermediate)](#building-up-intermediate)
4. [Advanced topics](#advanced-topics)
5. [Summary / cheat sheet](#summary--cheat-sheet)
6. [Self-test](#self-test)
7. [Literature & sources](#literature--sources)

---

## Learning objectives

After this module you should be able to …

- name and distinguish the **four universal 3D interaction tasks** after Bowman: **selection, manipulation, travel, wayfinding** — plus system control.
- fully master **coordinate systems and homogeneous transformations**: the $4\times4$ matrix, why translation can only be written as a matrix in homogeneous form, the **transformation chain** object→world→camera→image, and how to invert it.
- derive and implement the **ray-object intersection mathematics**: ray-sphere, ray-AABB (the slab method), ray-triangle (Möller-Trumbore).
- understand the most important **selection and manipulation techniques** as algorithms: ray-casting, cone/bubble, virtual hand, **Go-Go** (the non-linear arm extension — with its formula), HOMER, as well as **isomorphic vs. "magic" (non-isomorphic)** techniques and **DOF separation**.
- explain why **pointing in 3D is harder** than in 2D (**angular Fitts' law**, the Heisenberg effect, missing physical constraints, depth perception) and how control-display gain helps.
- place the **travel metaphors** (walking, steering, target-based/teleport, manipulation-based) and their coupling to **cybersickness** (cf. module 17).
- **design and evaluate** a 3D selection experiment soundly (ISO 9241-9 throughput, Fitts in 3D, docking, within-subject statistics as in module 17).

---

## Basics

### 1. Why is 3D interaction hard?

The desktop spoiled humans for 40 years with **2D, WIMP** (windows, icons, menus, pointer) and hard physical constraints (the mouse lies on the table). In 3D space all of that falls away. Four fundamental difficulties:

1. **Six degrees of freedom (6 DoF).** An object in space has three positional and three orientational degrees of freedom. A menu has two. The human has to control six simultaneously — more than perception can cleanly separate.
2. **No physical constraints / no passive haptics.** On the table the mouse stops at the edge; in the air there is no stop, no support, no feedback when "touching" a virtual object. The hand tires ("gorilla arm").
3. **Precision.** Without support the hand trembles (tremor); when clicking, the pointer slips (the **Heisenberg effect** of 3D interaction, section 13).
4. **Depth perception.** How far away is the object? Humans **systematically underestimate distances in VR** (often 70–80 % of the real distance), which distorts grasping and moving.

That is why 3D interaction needs *its own* techniques — you cannot simply "lift the mouse into the air".

### 2. The four universal interaction tasks (Bowman)

Almost every 3D interaction can be decomposed into four **canonical tasks**. This taxonomy structures the whole module:

| Task | Question | Example techniques |
|---|---|---|
| **Selection** | *Which* object do I mean? | ray-casting, virtual hand, Go-Go, bubble/cone |
| **Manipulation** | *move/rotate/scale* an object | virtual hand, HOMER, scaled-world grab, widgets |
| **Travel** (motor locomotion) | How do I *move* my viewpoint? | walking, steering, teleport, grab-the-world |
| **Wayfinding** (cognitive orientation) | *Where* am I, how do I *get there*? | maps, landmarks, a compass, trail markers |

In addition **system control** (choosing commands/modes — 3D menus, gestures, speech; this is where module 18 applies) and **symbolic input** (text in 3D — notoriously hard).

> **Remember the travel/wayfinding distinction:** *travel* is the **motor** component (the moving itself), *wayfinding* the **cognitive** one (knowing where to go). Both together = *navigation*. A good 3DUI supports both separately.

### 3. Coordinate systems and homogeneous transformations

The mathematical foundation. A 3D object exists in several **frames of reference** simultaneously, and interaction means constantly converting between them:

```
 object        world          camera/eye      image/screen
 coordinates → coordinates  → coordinates   → coordinates
   (model)      (world)        (view)         (projection)
      M_model       M_view         M_proj
```

A point is transformed through the chain: $\mathbf{p}_{\text{screen}} = M_{\text{proj}}\,M_{\text{view}}\,M_{\text{model}}\,\mathbf{p}_{\text{object}}$.

**Homogeneous coordinates.** The trick that holds it all together: you extend a 3D point $\mathbf{p}=(x,y,z)$ by a fourth coordinate $w=1$: $\tilde{\mathbf{p}} = (x,y,z,1)^\top$. The reason is purely pragmatic but profound: **translation is not a linear map** (it does not fix the origin) and *cannot* be written as a $3\times3$ matrix. In homogeneous coordinates it can after all — as a $4\times4$ matrix:

$$T(\mathbf{t}) = \begin{pmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix}, \quad
R = \begin{pmatrix} & & & 0 \\ & \mathbf{R}_{3\times3} & & 0 \\ & & & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}, \quad
S(\mathbf{s}) = \begin{pmatrix} s_x & & & 0 \\ & s_y & & 0 \\ & & s_z & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Now **rotation, translation and scaling are all $4\times4$ matrices** and can be **chained by matrix multiplication** — that is the entire reason for homogeneous coordinates. A rigid transformation (rigid body, rotation + translation) is

$$M = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}, \qquad
  M^{-1} = \begin{pmatrix} \mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}.$$

The inverse uses $\mathbf{R}^{-1}=\mathbf{R}^\top$ (rotation matrices are orthogonal, cf. module 17) — you never have to invert numerically. **The order is decisive** (matrix multiplication is not commutative, as already with the rotations in module 17): $T\,R$ (first rotate, then translate) $\neq R\,T$.

> **Careful, a convention:** it is applied from right to left onto the column vector: $M\mathbf{p} = T(R(S\mathbf{p}))$ means "first scale, then rotate, then translate". The $\mathbf{R}$ here can be built directly from a quaternion (module 17) — 3DUIs almost always store orientations as quaternions and convert to a matrix only for rendering.

### 4. Depth perception (depth cues)

For a human to point and grasp in 3D space, they have to **estimate distance**. The brain uses many **depth cues** for that, which you have to know because VR systems deliver them to differing degrees:

- **Occlusion**: the strongest cue — what occludes is nearer. Always correct in VR.
- **Perspective / relative size**: distant things appear smaller; parallel lines converge.
- **Stereopsis** (binocular disparity): the two eye images differ — the main advantage of a VR headset, but it only works up close (< ~10 m).
- **Motion parallax**: during head movement, near things move faster than distant ones — needs good head tracking (module 17).
- **Accommodation/convergence**: focus and eye position — the source of the **vergence-accommodation conflict** (VAC) from module 17.

**The consequence:** VR delivers stereopsis and parallax well, but the VAC and other missing cues lead to a systematic **underestimation of distance**. For a 3DUI that means: objects appear nearer than they are → grasping techniques have to compensate for that (see Go-Go, section 6).

---

## Building up (intermediate)

### 5. Selection I: ray-casting and the ray-object intersection mathematics

By far the most common selection technique: a **virtual ray** goes out from the controller (or the hand); the first object hit is selected. A ray is parametric

$$\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}, \quad t \ge 0,$$

with origin $\mathbf{o}$ and a **normalized** direction $\mathbf{d}$ ($\|\mathbf{d}\|=1$, then $t$ is the Euclidean distance). Selection = the **nearest intersection** over all objects. The mathematics per object type:

**Ray-sphere** (the object as a bounding sphere, centre $\mathbf{c}$, radius $R$). Substitute $\mathbf{r}(t)$ into $\|\mathbf{p}-\mathbf{c}\|^2 = R^2$. With $\mathbf{m}=\mathbf{o}-\mathbf{c}$:

$$\|\mathbf{m}+t\mathbf{d}\|^2 = R^2 \;\Longrightarrow\; t^2\underbrace{(\mathbf{d}\cdot\mathbf{d})}_{=1} + 2t(\mathbf{m}\cdot\mathbf{d}) + (\mathbf{m}\cdot\mathbf{m}-R^2)=0.$$

A quadratic equation $t^2 + 2bt + c = 0$ with $b=\mathbf{m}\cdot\mathbf{d}$, $c=\mathbf{m}\cdot\mathbf{m}-R^2$. The discriminant is $\Delta = b^2 - c$. If $\Delta<0$: no hit. Otherwise $t = -b - \sqrt{\Delta}$ (the nearer, front intersection); if that one is $<0$ and the other $>0$, the origin is *inside* the sphere.

**Ray-AABB** (an axis-aligned bounding box, the slab method). For every axis $i\in\{x,y,z\}$ the box is a "slab" $[\min_i, \max_i]$. The ray enters at $t_{i,1}=(\min_i - o_i)/d_i$ and exits at $t_{i,2}=(\max_i-o_i)/d_i$ (mind the sign of $d_i$ → swap). The ray hits the box iff the **maximum of the entries $\le$ the minimum of the exits**:

$$t_{\text{enter}} = \max_i \min(t_{i,1}, t_{i,2}), \quad t_{\text{exit}} = \min_i \max(t_{i,1}, t_{i,2}), \quad \text{hit} \iff t_{\text{enter}} \le t_{\text{exit}} \wedge t_{\text{exit}}\ge 0.$$

**Ray-triangle (Möller-Trumbore)** — for real mesh geometry. A triangle with corners $\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2$; a point in it is $\mathbf{v}_0 + u\,\mathbf{e}_1 + v\,\mathbf{e}_2$ with $\mathbf{e}_1=\mathbf{v}_1-\mathbf{v}_0$, $\mathbf{e}_2=\mathbf{v}_2-\mathbf{v}_0$ and **barycentric** coordinates $u,v\ge0$, $u+v\le1$. Equating this with $\mathbf{r}(t)$ gives a $3\times3$ system that is solved without matrix inversion via scalar triple products:

$$\mathbf{p} = \mathbf{d}\times\mathbf{e}_2,\quad \det = \mathbf{e}_1\cdot\mathbf{p}, \quad \mathbf{s}=\mathbf{o}-\mathbf{v}_0,\quad
u = \frac{\mathbf{s}\cdot\mathbf{p}}{\det},\quad \mathbf{q}=\mathbf{s}\times\mathbf{e}_1,\quad v=\frac{\mathbf{d}\cdot\mathbf{q}}{\det},\quad t=\frac{\mathbf{e}_2\cdot\mathbf{q}}{\det}.$$

A hit iff $u\ge0,\ v\ge0,\ u+v\le1,\ t>0$ (and $|\det|$ not ~0, otherwise the ray is parallel to the triangle). This is the standard algorithm of every ray tracer and selection system.

**Why ray-casting is so popular** — and where it fails: it allows **distant** objects to be selected effortlessly (the ray is arbitrarily long). But: at **distance** an object subtends a tiny angle, and even minimal hand tremor misses it (section 12, angular Fitts). Under **occlusion/density** it is unclear which of several objects lying behind or next to each other is meant — that is a **disambiguation problem** as in module 18. This is exactly where cone/bubble techniques come in.

### 6. Selection II: virtual hand and Go-Go (non-linear reach extension)

**Virtual hand.** The most direct technique: a virtual hand follows the real one **isomorphically** (1:1); you select by **touching** (a collision hand ↔ object). Intuitive and precise — but the reach is **limited to the arm's length**. You can only grasp what is within arm's reach.

**Go-Go** (Poupyrev et al. 1996) — the most elegant solution to the reach problem. The idea: the virtual hand follows the real one **non-linearly**. Within a threshold $D$ (typically ~2/3 of the arm's length) the mapping is 1:1 (precise near interaction); *beyond* $D$ the virtual reach grows **quadratically**, so that small real arm extensions produce large virtual reaches. Let $r_r$ be the real hand distance from the body and $r_v$ the virtual one:

$$r_v = \begin{cases} r_r & r_r < D \\[4pt] r_r + k\,(r_r - D)^2 & r_r \ge D \end{cases}$$

with a gain coefficient $k$ (controlling how fast the reach grows). The function is **continuous and continuously differentiable** at $r_r=D$ (value $D$, derivative $1$) — no jump, no kink in the velocity, which makes it feel "natural". Go-Go thus keeps the **precision** of the virtual hand up close and gains the **reach** of ray-casting at a distance. The **basic** and **medium** projects implement this function.

> Go-Go is the prototype of a **"magic" (non-isomorphic)** technique: it deliberately breaks with 1:1 reality in order to achieve better usability. That is a core principle of 3D interaction — *isomorphism* (fidelity to reality) is *not* always optimal.

### 7. Selection III: volume techniques (cone/bubble) against ambiguity

When ray-casting fails at small/dense/distant targets, a **selection volume** instead of a line helps:

- **Cone/flashlight**: a **cone** instead of a ray; all objects in the cone are candidates. The object chosen is the one with the **smallest angular distance to the cone axis**. This makes hitting small/distant targets easier — but in **dense** scenes many objects are in the cone → a new ambiguity.
- **Bubble cursor** (Grossman & Balakrishnan, originally 2D, 3D variants exist): a cursor with a **dynamic radius** that adapts so that **exactly one** target is enclosed — it always fully encompasses the *nearest* object. This effectively makes every target "as large as its Voronoi region" → drastically better Fitts performance in sparsely populated scenes.

The choice between a point and a volume technique is a **precision vs. ambiguity trade-off** that the **final project** measures empirically: ray-casting is precise but unusable for small distant targets; cone/bubble grab those easily but over-select in a crowd. The disambiguation "which of the candidate objects?" is conceptually the same problem as reference resolution in module 18 (there with time + semantics, here with angle + distance).

### 8. Manipulation: DOF, isomorphic vs. magic, HOMER, DOF separation

Selection is often followed by **manipulation** (moving, rotating, scaling). The core concepts:

- **Isomorphic 6-DoF manipulation**: the virtual hand transfers translation + rotation 1:1 onto the object. Natural, but again limited in reach and precision.
- **HOMER** (*hand-centered object manipulation extending ray-casting*): selects by ray-casting (a large reach), then **the virtual hand jumps to the object** and manipulates it hand-centred. It combines the reach of ray-casting with virtual-hand manipulation.
- **Scaled-world grab**: scales the world on grabbing so that the distant object comes into reach — mathematically elegant, but it can disorient.
- **DOF separation**: often you do *not* want to control all 6 DoF simultaneously (e.g. move a picture only *along the wall*). Techniques **constrain** DoF, **snap** to grids/edges or separate translation from rotation via widgets/handles. This compensates for the human weakness of guiding many DoF precisely at the same time.

> **Perceptual structure (Jacob et al.):** degrees of freedom should be grouped the way a human *perceives* them. Position (x,y,z) is **integral** (you move the hand as a whole), position-vs-colour would be **separable**. A manipulation technique that separates integral DoF or couples separable ones feels wrong. The design rule: **the DOF structure of the task = the DOF structure of the technique.**

### 9. Travel & wayfinding

**Travel metaphors** (the motor locomotion):

- **Physical walking / real walking**: the most immersive, the least prone to sickness — but limited by the real room (redirected walking as a trick).
- **Steering**: continuous direction input — *gaze-directed* (the direction of view) or *pointing-directed* (the direction of the hand). Simple, but continuous visual movement without bodily movement → **vection → cybersickness** (cf. module 17, the sensory conflict theory).
- **Target-based / teleport**: you choose a target and are **immediately transported there**. The de facto standard in VR, **because it minimizes cybersickness** (no continuous vection) — at the cost of spatial understanding.
- **Manipulation-based** ("grab the world"): you grab the world and pull yourself through it.

**Wayfinding support** (the cognitive orientation): **landmarks**, **maps** (you-are-here, with the delicate question of orientation — track-up vs. north-up), a **compass/trails/breadcrumbs**, good **sight lines**. The goal is building a mental **cognitive map**.

---

## Advanced topics

### 10. Fitts' law in 3D: why pointing breaks down with distance

In module 17 we introduced Fitts' law for 2D pointing: the movement time

$$MT = a + b\,\underbrace{\log_2\!\Big(\tfrac{D}{W}+1\Big)}_{\text{index of difficulty }ID}$$

with target distance $D$ and target width $W$. In 3D — especially with **ray-casting** — the relevant quantity is **not the linear but the *angular* extent**. A target of (transverse) width $W$ at distance $L$ from the eye/controller subtends the angle

$$\theta_W \approx 2\arctan\!\Big(\frac{W}{2L}\Big) \approx \frac{W}{L} \quad (\text{for } W \ll L),$$

and the pointer has to sweep over an **angle** $\theta_D$. The **angular Fitts' law** replaces distance/width by angles:

$$MT = a + b\,\log_2\!\Big(\frac{\theta_D}{\theta_W}+1\Big).$$

> **The decisive consequence:** $\theta_W \approx W/L$ **shrinks with the distance $L$**. An object twice as far away is *angularly* half as large → $ID$ rises → selection takes longer and becomes more error-prone. **Ray-casting precision degrades linearly with the target distance** — that is the hard, quantitative reason why distant small targets are so hard to hit in VR, and why Go-Go (bringing the target "closer") or bubble (enlarging the effective $\theta_W$) help. The **medium project** measures exactly this angular Fitts relation.

### 11. Control-display gain and adaptive gain (PRISM)

**Control-display (C/D) gain** is the ratio of display movement to control movement. A gain $>1$ (a small hand movement → a large pointer movement) gives **reach/speed** and costs **precision**; a gain $<1$ the other way round. A fixed gain is a compromise. **Adaptive methods** such as **PRISM** (Frees et al.) lower the gain during *slow* hand movement (precise aiming) and raise it during fast movement (large-scale moving) — they use the hand velocity as an indicator of intent. That is the continuous generalization of the Go-Go idea.

### 12. The Heisenberg effect of 3D interaction

A subtle, practically relevant effect: when **actuating the selection button** (pressing the trigger) the hand wobbles — the pointer jumps away from the target at the moment of the click. The smaller/more distant the target (a small $\theta_W$), the more fatal. Countermeasures: **freeze** the pointer state shortly **before** the click, filter click events temporally, or confirm via a **different modality** (speech instead of the trigger — module 18 again). This is the 3D analogue of the wobble during a mouse click, only much stronger without the stabilizing table support.

### 13. Ray-casting precision as a stochastic model

For the evaluation one models the pointing precision as **angular noise**: the real pointing direction scatters in a Gaussian way around the intended one, with standard deviation $\sigma_\theta$ (hand tremor + tracking noise + Heisenberg). A target with angular radius $\theta_W/2$ at distance $L$ is hit if the angular deviation is smaller:

$$P(\text{hit}) = P\!\big(|\epsilon_\theta| < \tfrac{\theta_W}{2}\big) = P\!\big(|\epsilon_\theta| < \tfrac{W}{2L}\big), \quad \epsilon_\theta \sim \mathcal{N}(0,\sigma_\theta^2).$$

This couples directly to the inverse-variance thinking from module 18: precision is $1/\sigma_\theta^2$, and everything that lowers $\sigma_\theta$ (support, stabilization, C/D gain, prediction as in module 17) increases the hit probability. The **projects** build exactly this model.

### 14. Evaluation: ISO 9241-9 and throughput in 3D

The standard for assessing pointing devices/techniques is **ISO 9241-9**, whose core is the **throughput** (in bits/s):

$$TP = \frac{ID_e}{MT}, \qquad ID_e = \log_2\!\Big(\frac{D_e}{W_e}+1\Big),$$

with the **effective** width $W_e = 4.133\,\sigma_x$ (from the spread of the actual click positions — the "effective" width corrects for the accuracy the user really used, so that ~96 % of the clicks lie in the target) and the effective distance $D_e$. Throughput combines **speed and accuracy into one number** and makes techniques comparable. Typical tasks: **reciprocal tapping** (back and forth between two targets), **docking** (bringing an object into a target pose — this tests manipulation including rotation). The statistical analysis (within-subject, counterbalancing, effect sizes, Wilcoxon/ANOVA) follows exactly the methodology from **module 17** — the **final project** applies it to a comparison of selection techniques.

---

## Summary / cheat sheet

**The four tasks (Bowman)**: selection · manipulation · travel (motor) · wayfinding (cognitive) · [+ system control].

**Homogeneous transformations**
- A homogeneous point: $\tilde{\mathbf p}=(x,y,z,1)$. The reason: translation becomes writable as a $4\times4$ matrix → everything is chainable.
- The chain: $\mathbf p_{\text{screen}} = M_{\text{proj}} M_{\text{view}} M_{\text{model}}\,\mathbf p_{\text{obj}}$. The order counts (not commutative).
- The rigid-body inverse: $M^{-1}=\begin{psmallmatrix}\mathbf R^\top & -\mathbf R^\top\mathbf t\\ 0 & 1\end{psmallmatrix}$ (it uses $\mathbf R^{-1}=\mathbf R^\top$).

**The ray $\mathbf r(t)=\mathbf o+t\mathbf d$ ($\|\mathbf d\|=1$), intersection**
| Object | The core |
|---|---|
| Sphere | $t^2+2bt+c=0$, $b=\mathbf m\cdot\mathbf d$, $c=\|\mathbf m\|^2-R^2$, $\mathbf m=\mathbf o-\mathbf c$; $t=-b-\sqrt{b^2-c}$ |
| AABB | Slabs: a hit $\iff \max_i\min(t_{i1},t_{i2}) \le \min_i\max(t_{i1},t_{i2})$ |
| Triangle | Möller-Trumbore: $u,v$ barycentric $\ge0$, $u+v\le1$, $t>0$ |

**Selection techniques**: ray-casting (reach, but weak in angular precision) · virtual hand (precise, short reach) · **Go-Go** $r_v=r_r+k(r_r-D)^2$ for $r_r\ge D$ (both) · cone/bubble (against small/sparse targets, but ambiguity in a crowd).

**Fitts in 3D**: angular, $ID=\log_2(\theta_D/\theta_W+1)$ with $\theta_W\approx W/L$ → **precision falls with the distance $L$**.

**Throughput (ISO 9241-9)**: $TP=ID_e/MT$, $W_e=4.133\,\sigma_x$ (the effective width from the click spread).

**Travel & sickness**: teleport minimizes vection/sickness (module 17); steering is simple but prone to sickness; real walking is the best but limited by the room.

**Design rules**: a "magic" (non-isomorphic) technique often beats the faithful one; the DOF structure of the technique = the DOF structure of the task; handle the Heisenberg effect on clicking.

---

## Self-test

<details>
<summary><b>1.</b> Why do you need homogeneous ($4\times4$) coordinates — do $3\times3$ rotation matrices not suffice?</summary>

Because **translation is not a linear map** (it does not map the origin onto itself) and therefore *cannot* be written as a $3\times3$ matrix. In homogeneous coordinates (a fourth component $w=1$) translation becomes a $4\times4$ matrix with the displacement vector in the last column. With that, **rotation, translation and scaling are all matrices** and can be chained by multiplication into **one** transformation chain — that is the entire purpose.
</details>

<details>
<summary><b>2.</b> Name the four universal 3D interaction tasks and the difference between travel and wayfinding.</summary>

**Selection, manipulation, travel, wayfinding** (+ system control). **Travel** is the *motor* locomotion (the moving of the viewpoint itself), **wayfinding** the *cognitive* orientation (knowing where you are and how to get to the goal). Together they form *navigation*.
</details>

<details>
<summary><b>3.</b> Derive the ray-sphere intersection equation. When is there no hit?</summary>

Substitute the ray $\mathbf r(t)=\mathbf o+t\mathbf d$ ($\|\mathbf d\|=1$) into the sphere equation $\|\mathbf p-\mathbf c\|^2=R^2$. With $\mathbf m=\mathbf o-\mathbf c$: $\|\mathbf m+t\mathbf d\|^2=R^2 \Rightarrow t^2 + 2t(\mathbf m\cdot\mathbf d) + (\|\mathbf m\|^2-R^2)=0$. That is $t^2+2bt+c=0$ with $b=\mathbf m\cdot\mathbf d$, $c=\|\mathbf m\|^2-R^2$. The discriminant is $\Delta=b^2-c$. **No hit if $\Delta<0$** (the ray misses the sphere). Otherwise the nearer intersection is $t=-b-\sqrt\Delta$.
</details>

<details>
<summary><b>4.</b> Write down the Go-Go function and explain which problem it solves and why its form (quadratic, with a threshold) makes sense.</summary>

$$r_v = \begin{cases} r_r & r_r<D\\ r_r + k(r_r-D)^2 & r_r\ge D\end{cases}$$

It solves the **reach problem** of the virtual hand: up close ($r_r<D$) the mapping stays **1:1** → full precision; beyond the threshold $D$ the virtual reach grows **quadratically** → small real arm extensions reach distant objects. The quadratic form is **continuous and differentiable with derivative 1** at $r_r=D$ (no jump, no kink in the velocity) → it feels natural. Go-Go thus unites near precision and far reach and is an example of a "magic" (non-isomorphic) technique.
</details>

<details>
<summary><b>5.</b> Why does ray-casting get worse with increasing target distance? Use the angular Fitts' law.</summary>

Because the quantity relevant for pointing is the **angular** radius: a target of width $W$ at distance $L$ subtends only $\theta_W\approx W/L$. That **shrinks with $L$** — double the distance = half the angular size. In the angular Fitts' law $ID=\log_2(\theta_D/\theta_W+1)$ the index of difficulty therefore rises, the movement time grows and (at a fixed angular noise $\sigma_\theta$) the hit probability $P(|\epsilon_\theta|<\theta_W/2)$ falls. The conclusion: ray-casting precision falls with distance.
</details>

<details>
<summary><b>6.</b> What is the Heisenberg effect of 3D interaction, and how do you counter it?</summary>

When **pressing the selection button** the hand wobbles, so that the pointer jumps away from the target at the moment of the click — particularly bad for small/distant (angularly tiny) targets. Countermeasures: **freeze the pointer state shortly before the click**, filter clicks temporally, or move the confirmation into a **different modality** (e.g. a speech command instead of the trigger — module 18).
</details>

<details>
<summary><b>7.</b> Why is teleport travel so widespread in VR, and what is the price?</summary>

Because teleport (target-based) produces **no continuous visual self-motion** → no **vection** → minimal **cybersickness** (the sensory conflict theory, module 17). The price is **worse spatial understanding / wayfinding**: through the "jumping" the user builds a less coherent cognitive map and can judge the distance/direction of the path less well.
</details>

<details>
<summary><b>8.</b> What is the "effective width" $W_e$ in the ISO 9241-9 throughput, and why is it used instead of the target width $W$?</summary>

$W_e = 4.133\,\sigma_x$ is computed from the **spread of the actual click positions**. It captures the **accuracy the user really used**: if somebody aims sloppily (a wide spread), $W_e$ rises; if they aim more precisely than necessary, it falls. The factor 4.133 normalizes so that ~96 % of the clicks fall into the effective target. This **factors out the speed-accuracy trade-off** — the throughput $TP=ID_e/MT$ assesses techniques fairly, no matter whether a user acts fast-and-imprecise or slow-and-precise.
</details>

<details>
<summary><b>9.</b> When is a bubble cursor / a cone technique superior to ray-casting, and when not?</summary>

**Superior** for **small, distant or isolated** targets: the bubble cursor enlarges the effective target to its Voronoi region, the cone technique lowers the required angular precision → both make hitting dramatically easier (better Fitts performance). **Not superior** in **dense** scenes: then many objects lie in the cone/radius and a **disambiguation problem** arises — which object is meant? Ray-casting is more precise there. It is a precision vs. ambiguity trade-off.
</details>

<details>
<summary><b>10.</b> What does the principle "the DOF structure of the technique = the DOF structure of the task" say?</summary>

Degrees of freedom should be **grouped** in the technique the way a human **perceives and controls** them in the task (integral vs. separable, after Jacob et al.). Position (x,y,z) is *integral* — you move the hand as a whole; separating such DoF (e.g. having to set each axis individually) feels wrong. Conversely you should not artificially couple *separable* parts of a task. In practice that means: choose constraints/widgets/snapping so that they match the natural DoF perception of the task.
</details>

---

## Literature & sources

**Textbooks**
- **LaViola, Kruijff, McMahan, Bowman & Poupyrev, *3D User Interfaces: Theory and Practice* (2nd ed., 2017).** *The* standard work — the taxonomy of the four tasks, all techniques, evaluation. Mandatory. *Friendly from beginner to advanced.*
- **Foley/van Dam or a similar computer graphics textbook** for homogeneous transformations and ray-object intersection; alternatively **Marschner & Shirley, *Fundamentals of Computer Graphics*** (the chapters on transformations, ray tracing). *Beginner-friendly.*
- **Ericson, *Real-Time Collision Detection*** — the authoritative reference for ray-sphere/AABB/triangle tests. *In depth.*

**Key papers (freely findable)**
- **Poupyrev, Billinghurst, Weghorst & Ichikawa, "The Go-Go Interaction Technique"**, *UIST 1996*. The reach extension. *Short, worth reading.*
- **Bowman & Hodges, "An Evaluation of Techniques for Grabbing and Manipulating Remote Objects" (HOMER)**, *I3D 1997*. *In depth.*
- **Grossman & Balakrishnan, "The Bubble Cursor"**, *CHI 2005*. A dynamic cursor radius, Fitts-optimal. *Beginner-friendly.*
- **Möller & Trumbore, "Fast, Minimum Storage Ray-Triangle Intersection"**, *Journal of Graphics Tools 1997*. The standard algorithm. *In depth, but compact.*
- **Frees, Kessler & Kay, "PRISM Interaction for Enhancing Control in Immersive Virtual Environments"**, *ACM TOCHI 2007*. Adaptive C/D gain. *In depth.*
- **MacKenzie, "Fitts' Law as a Research and Design Tool in HCI"**, *HCI 1992* — the reference for Fitts/throughput (also for ISO 9241-9). *Beginner-friendly.*

**Freely available courses / materials**
- **Scratchapixel** (scratchapixel.com) — excellent, free tutorials on transformations and ray-object intersection with complete derivations. *Free, beginner-friendly.*
- Various **VR/3DUI lectures** (e.g. by Doug Bowman / Virginia Tech) with freely available slides. *Free, in depth.*

**To try out**
- The **three projects of this module** build transformation chains + ray-casting (basic), the angular Fitts model + Go-Go (medium) and a complete comparison of selection techniques under clutter with ISO throughput (final) — the best deepening is to implement them.

---

> **The next module:** module 20 "3D Point Cloud Processing" — the processing of 3D point clouds (registration/ICP, segmentation, feature descriptors, PointNet). The 3D geometry and transformation mathematics from this module (section 3) is the direct foundation.

---

# Modul 19 — 3D User Interfaces (deutsche Fassung)

> **Worum geht es?** Sobald Inhalte im **dreidimensionalen Raum** liegen — in VR/AR, in einem CAD-Programm, in einem Spiel —, reicht die aus dem 2D-Desktop gewohnte Interaktion (Maus, Fenster, Menü) nicht mehr. Ein **3D User Interface (3DUI)** ist eine Schnittstelle, über die der Mensch Objekte im 3D-Raum **auswählt, manipuliert, sich darin bewegt und orientiert**. Dieses Modul behandelt die *Prinzipien* und die *Mathematik* dieser Interaktion: Wie hängen die Koordinatensysteme zusammen? Wie trifft ein virtueller Zeigestrahl ein Objekt? Wie erreicht man mit einem kurzen Arm ein weit entferntes Objekt? Und warum ist Zeigen in 3D *fundamental schwerer* als in 2D?
>
> **Vorkenntnisse**: lineare Algebra (Matrizen, Vektoren, Skalarprodukt), etwas 3D-Geometrie. Aus diesem Repo bauen direkt auf: **Modul 17** (Core XR — Rotationsmathematik/Quaternionen, Fitts' Law, Tracking, Cybersickness) und **Modul 18** (Multimodal — Referenzauflösung, Zeige-Ambiguität, Nutzerstudien-Methodik). Modul 17 ist **Pflicht-Vormodul**; vieles hier ist dessen direkte Fortsetzung.

> **Hinweis zum Zuschnitt.** Wie bei den Modulen 15–18 liegt keine offizielle Modulbeschreibung vor; ich habe den Inhalt selbst zugeschnitten, eng an der maßgeblichen Referenz (Bowman, LaViola, Kruijff, Poupyrev: *3D User Interfaces — Theory and Practice*) und konsistent mit dem XR-Block dieses Repos. **Wieder ohne konkrete Hardware** (kein VR-Headset, kein Tracked Controller): Der lehrbare, übertragbare Kern sind die **Transformationsmathematik, die Interaktionstechniken als Algorithmen, die Zeige-Präzisionsmodelle (Fitts in 3D) und die Evaluationsmethodik**. Ein Controller zu halten lernt man in Minuten; zu verstehen, *warum* Ray-Casting mit der Distanz zusammenbricht und *wie* Go-Go den Arm nichtlinear verlängert, ist die Master-Kompetenz. Die Projekte simulieren Zeigen und Auswahl realistisch mit reiner Geometrie/Statistik auf der CPU.

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

- die **vier universellen 3D-Interaktionsaufgaben** nach Bowman benennen und abgrenzen können: **Selektion, Manipulation, Travel (Fortbewegung), Wayfinding (Orientierung)** — plus System Control.
- **Koordinatensysteme und homogene Transformationen** vollständig beherrschen: die $4\times4$-Matrix, warum man Translation nur homogen als Matrix schreiben kann, die **Transformationskette** Objekt→Welt→Kamera→Bild, und wie man sie invertiert.
- die **Ray-Objekt-Schnitt-Mathematik** herleiten und implementieren können: Ray-Kugel, Ray-AABB (Slab-Methode), Ray-Dreieck (Möller-Trumbore).
- die wichtigsten **Selektions- und Manipulationstechniken** als Algorithmen verstehen: Ray-Casting, Cone/Bubble, Virtual Hand, **Go-Go** (die nichtlineare Armverlängerung — mit Formel), HOMER, sowie **isomorphe vs. „magische" (non-isomorphic)** Techniken und **DOF-Separation**.
- erklären können, warum **Zeigen in 3D schwerer** ist als in 2D (**angulares Fitts' Law**, Heisenberg-Effekt, fehlende physische Constraints, Tiefenwahrnehmung) und wie Control-Display-Gain hilft.
- die **Travel-Metaphern** (Walking, Steering, Target-based/Teleport, Manipulation-based) und ihre Kopplung an **Cybersickness** (Rückverweis Modul 17) einordnen.
- ein 3D-Selektionsexperiment **sauber gestalten und auswerten** (ISO-9241-9-Throughput, Fitts in 3D, Docking, within-subject-Statistik wie in Modul 17).

---

## Grundlagen (Basics)

### 1. Warum ist 3D-Interaktion schwer?

Der Desktop hat den Menschen 40 Jahre lang mit **2D, WIMP** (Windows, Icons, Menus, Pointer) und harten physischen Constraints (die Maus liegt auf dem Tisch) verwöhnt. Im 3D-Raum fällt das alles weg. Vier fundamentale Schwierigkeiten:

1. **Sechs Freiheitsgrade (6 DoF).** Ein Objekt im Raum hat drei Positions- und drei Orientierungsfreiheitsgrade. Ein Menü hat zwei. Der Mensch muss sechs gleichzeitig kontrollieren — mehr, als die Wahrnehmung sauber trennen kann.
2. **Keine physischen Constraints / keine passive Haptik.** Auf dem Tisch stoppt die Maus an der Tischkante; in der Luft gibt es keinen Anschlag, keine Auflage, kein Feedback beim „Berühren" eines virtuellen Objekts. Die Hand ermüdet („gorilla arm").
3. **Präzision.** Ohne Auflage zittert die Hand (Tremor); beim Klicken verrutscht der Zeiger (der **Heisenberg-Effekt** der 3D-Interaktion, Abschnitt 13).
4. **Tiefenwahrnehmung.** Wie weit ist das Objekt? Menschen **unterschätzen Distanzen in VR** systematisch (oft 70–80 % der echten Distanz), was Greifen und Bewegen verzerrt.

Deshalb braucht 3D-Interaktion *eigene* Techniken — man kann die Maus nicht einfach „in die Luft heben".

### 2. Die vier universellen Interaktionsaufgaben (Bowman)

Fast jede 3D-Interaktion lässt sich in vier **kanonische Aufgaben** zerlegen. Diese Taxonomie strukturiert das ganze Modul:

| Aufgabe | Frage | Beispieltechniken |
|---|---|---|
| **Selektion** | *Welches* Objekt meine ich? | Ray-Casting, Virtual Hand, Go-Go, Bubble/Cone |
| **Manipulation** | Objekt *bewegen/drehen/skalieren* | Virtual Hand, HOMER, Scaled-World Grab, Widgets |
| **Travel** (motorische Fortbewegung) | Wie *bewege* ich meinen Blickpunkt? | Walking, Steering, Teleport, Grab-the-world |
| **Wayfinding** (kognitive Orientierung) | *Wo* bin ich, wie komme ich *hin*? | Karten, Landmarken, Kompass, Wegmarkierungen |

Dazu **System Control** (Befehle/Modi wählen — 3D-Menüs, Gesten, Sprache; hier greift Modul 18) und **symbolische Eingabe** (Text in 3D — notorisch schwer).

> **Merke die Travel/Wayfinding-Unterscheidung:** *Travel* ist die **motorische** Komponente (das Bewegen selbst), *Wayfinding* die **kognitive** (das Wissen, wohin). Beide zusammen = *Navigation*. Ein gutes 3DUI unterstützt beide getrennt.

### 3. Koordinatensysteme und homogene Transformationen

Das mathematische Fundament. Ein 3D-Objekt existiert in mehreren **Bezugssystemen (frames)** gleichzeitig, und Interaktion heißt ständig, zwischen ihnen umzurechnen:

```
 Objekt-       Welt-         Kamera-/Eye-    Bild-/Screen-
 koordinaten → koordinaten → koordinaten  → koordinaten
   (model)      (world)        (view)         (projection)
      M_model       M_view         M_proj
```

Ein Punkt wird durch die Kette transformiert: $\mathbf{p}_{\text{screen}} = M_{\text{proj}}\,M_{\text{view}}\,M_{\text{model}}\,\mathbf{p}_{\text{object}}$.

**Homogene Koordinaten.** Der Trick, der alles zusammenhält: Man erweitert einen 3D-Punkt $\mathbf{p}=(x,y,z)$ um eine vierte Koordinate $w=1$: $\tilde{\mathbf{p}} = (x,y,z,1)^\top$. Der Grund ist rein pragmatisch, aber tiefgreifend: **Translation ist keine lineare Abbildung** (sie fixiert den Ursprung nicht) und lässt sich *nicht* als $3\times3$-Matrix schreiben. In homogenen Koordinaten geht es doch — als $4\times4$-Matrix:

$$T(\mathbf{t}) = \begin{pmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix}, \quad
R = \begin{pmatrix} & & & 0 \\ & \mathbf{R}_{3\times3} & & 0 \\ & & & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}, \quad
S(\mathbf{s}) = \begin{pmatrix} s_x & & & 0 \\ & s_y & & 0 \\ & & s_z & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Jetzt sind **Rotation, Translation und Skalierung alle $4\times4$-Matrizen** und lassen sich durch **Matrixmultiplikation verketten** — das ist der ganze Grund für homogene Koordinaten. Eine starre Transformation (rigid body, Rotation + Translation) ist

$$M = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}, \qquad
  M^{-1} = \begin{pmatrix} \mathbf{R}^\top & -\mathbf{R}^\top\mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}.$$

Die Inverse nutzt $\mathbf{R}^{-1}=\mathbf{R}^\top$ (Rotationsmatrizen sind orthogonal, Rückverweis Modul 17) — man muss nie numerisch invertieren. **Reihenfolge ist entscheidend** (Matrixmultiplikation ist nicht kommutativ, wie schon bei den Rotationen in Modul 17): $T\,R$ (erst rotieren, dann verschieben) $\neq R\,T$.

> **Achtung, Konvention:** Angewandt wird von rechts nach links auf den Spaltenvektor: $M\mathbf{p} = T(R(S\mathbf{p}))$ heißt „erst skalieren, dann rotieren, dann translatieren". Die $\mathbf{R}$ hier kann direkt aus einem Quaternion (Modul 17) gebaut werden — 3DUIs speichern Orientierungen fast immer als Quaternion und konvertieren nur zum Rendern in die Matrix.

### 4. Tiefenwahrnehmung (depth cues)

Damit der Mensch im 3D-Raum zeigen und greifen kann, muss er **Distanz schätzen**. Das Gehirn nutzt dafür viele **Tiefenhinweise (depth cues)**, die man kennen muss, weil VR-Systeme sie unterschiedlich gut liefern:

- **Okklusion** (Verdeckung): der stärkste Hinweis — was verdeckt, ist näher. Immer korrekt in VR.
- **Perspektive / relative Größe**: Fernes erscheint kleiner; parallele Linien konvergieren.
- **Stereopsis** (binokulare Disparität): die zwei Augenbilder differieren — Hauptvorteil eines VR-Headsets, wirkt aber nur im Nahbereich (< ~10 m).
- **Bewegungsparallaxe**: bei Kopfbewegung wandert Nahes schneller als Fernes — braucht gutes Head-Tracking (Modul 17).
- **Akkommodation/Konvergenz**: Fokus- und Augenstellung — Quelle des **Vergenz-Akkommodations-Konflikts** (VAC) aus Modul 17.

**Konsequenz:** VR liefert Stereopsis und Parallaxe gut, aber der VAC und fehlende weitere Cues führen zur systematischen **Distanzunterschätzung**. Für ein 3DUI heißt das: Objekte wirken näher, als sie sind → Greiftechniken müssen das kompensieren (siehe Go-Go, Abschnitt 6).

---

## Aufbau (Intermediate)

### 5. Selektion I: Ray-Casting und die Ray-Objekt-Schnitt-Mathematik

Die mit Abstand häufigste Selektionstechnik: Vom Controller (oder der Hand) geht ein **virtueller Strahl (ray)** aus; das erste getroffene Objekt wird selektiert. Ein Strahl ist parametrisch

$$\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}, \quad t \ge 0,$$

mit Ursprung $\mathbf{o}$ und **normierter** Richtung $\mathbf{d}$ ($\|\mathbf{d}\|=1$, dann ist $t$ die euklidische Distanz). Selektion = **nächster Schnittpunkt** über alle Objekte. Die Mathematik pro Objekttyp:

**Ray-Kugel** (Objekt als Bounding Sphere, Zentrum $\mathbf{c}$, Radius $R$). Setze $\mathbf{r}(t)$ in $\|\mathbf{p}-\mathbf{c}\|^2 = R^2$ ein. Mit $\mathbf{m}=\mathbf{o}-\mathbf{c}$:

$$\|\mathbf{m}+t\mathbf{d}\|^2 = R^2 \;\Longrightarrow\; t^2\underbrace{(\mathbf{d}\cdot\mathbf{d})}_{=1} + 2t(\mathbf{m}\cdot\mathbf{d}) + (\mathbf{m}\cdot\mathbf{m}-R^2)=0.$$

Eine quadratische Gleichung $t^2 + 2bt + c = 0$ mit $b=\mathbf{m}\cdot\mathbf{d}$, $c=\mathbf{m}\cdot\mathbf{m}-R^2$. Diskriminante $\Delta = b^2 - c$. Ist $\Delta<0$: kein Treffer. Sonst $t = -b - \sqrt{\Delta}$ (der nähere, vordere Schnittpunkt); ist dieser $<0$ und der andere $>0$, ist der Ursprung *innerhalb* der Kugel.

**Ray-AABB** (achsenparallele Bounding Box, Slab-Methode). Für jede Achse $i\in\{x,y,z\}$ ist die Box ein „Slab" $[\min_i, \max_i]$. Der Strahl tritt bei $t_{i,1}=(\min_i - o_i)/d_i$ ein und bei $t_{i,2}=(\max_i-o_i)/d_i$ aus (Vorzeichen von $d_i$ beachten → tauschen). Der Strahl trifft die Box gdw. das **Intervall-Maximum der Eintritte $\le$ das Intervall-Minimum der Austritte** ist:

$$t_{\text{enter}} = \max_i \min(t_{i,1}, t_{i,2}), \quad t_{\text{exit}} = \min_i \max(t_{i,1}, t_{i,2}), \quad \text{Treffer} \iff t_{\text{enter}} \le t_{\text{exit}} \wedge t_{\text{exit}}\ge 0.$$

**Ray-Dreieck (Möller-Trumbore)** — für echte Mesh-Geometrie. Ein Dreieck mit Ecken $\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2$; ein Punkt darin ist $\mathbf{v}_0 + u\,\mathbf{e}_1 + v\,\mathbf{e}_2$ mit $\mathbf{e}_1=\mathbf{v}_1-\mathbf{v}_0$, $\mathbf{e}_2=\mathbf{v}_2-\mathbf{v}_0$ und **baryzentrischen** Koordinaten $u,v\ge0$, $u+v\le1$. Gleichsetzen mit $\mathbf{r}(t)$ ergibt ein $3\times3$-System, das man ohne Matrixinversion über Spatprodukte löst:

$$\mathbf{p} = \mathbf{d}\times\mathbf{e}_2,\quad \det = \mathbf{e}_1\cdot\mathbf{p}, \quad \mathbf{s}=\mathbf{o}-\mathbf{v}_0,\quad
u = \frac{\mathbf{s}\cdot\mathbf{p}}{\det},\quad \mathbf{q}=\mathbf{s}\times\mathbf{e}_1,\quad v=\frac{\mathbf{d}\cdot\mathbf{q}}{\det},\quad t=\frac{\mathbf{e}_2\cdot\mathbf{q}}{\det}.$$

Treffer gdw. $u\ge0,\ v\ge0,\ u+v\le1,\ t>0$ (und $|\det|$ nicht ~0, sonst Strahl parallel zum Dreieck). Das ist der Standard-Algorithmus jedes Raytracers und Selektions-Systems.

**Warum Ray-Casting so beliebt ist** — und wo es scheitert: Es erlaubt, **weit entfernte** Objekte mühelos zu selektieren (der Strahl ist beliebig lang). Aber: Bei **Distanz** subtendiert ein Objekt einen winzigen Winkel, und schon minimales Handzittern verfehlt es (Abschnitt 12, angulares Fitts). Bei **Verdeckung/Dichte** ist unklar, welches von mehreren hintereinander/nah beieinanderliegenden Objekten gemeint ist — das ist ein **Disambiguierungsproblem** wie in Modul 18. Genau hier setzen Cone/Bubble-Techniken an.

### 6. Selektion II: Virtual Hand und Go-Go (nichtlineare Reichweitenverlängerung)

**Virtual Hand.** Die direkteste Technik: Eine virtuelle Hand folgt **isomorph** (1:1) der echten Hand; man selektiert durch **Berühren** (Kollision Hand ↔ Objekt). Intuitiv und präzise — aber die Reichweite ist **auf die Armlänge begrenzt**. Man kann nur greifen, was in Armnähe ist.

**Go-Go** (Poupyrev et al. 1996) — die eleganteste Lösung des Reichweitenproblems. Idee: Die virtuelle Hand folgt der echten **nichtlinear**. Innerhalb einer Schwelle $D$ (typisch ~2/3 der Armlänge) ist die Abbildung 1:1 (präzise Nahinteraktion); *jenseits* von $D$ wächst die virtuelle Reichweite **quadratisch**, sodass kleine reale Armstreckungen große virtuelle Reichweiten erzeugen. Sei $r_r$ die reale Handdistanz vom Körper und $r_v$ die virtuelle:

$$r_v = \begin{cases} r_r & r_r < D \\[4pt] r_r + k\,(r_r - D)^2 & r_r \ge D \end{cases}$$

mit einem Verstärkungskoeffizienten $k$ (steuert, wie schnell die Reichweite wächst). Die Funktion ist **stetig und stetig differenzierbar** an $r_r=D$ (Wert $D$, Ableitung $1$) — kein Sprung, kein Knick in der Geschwindigkeit, was sie „natürlich" anfühlen lässt. Go-Go behält so die **Präzision** der Virtual Hand im Nahbereich und gewinnt die **Reichweite** des Ray-Castings in der Ferne. Das **Basic-** und **Medium-Projekt** implementieren diese Funktion.

> Go-Go ist der Prototyp einer **„magischen" (non-isomorphic)** Technik: Sie bricht bewusst mit der 1:1-Realität, um eine bessere Usability zu erzielen. Das ist ein Kernprinzip der 3D-Interaktion — der *Isomorphismus* (Realtreue) ist *nicht* immer optimal.

### 7. Selektion III: Volumen-Techniken (Cone/Bubble) gegen Ambiguität

Wenn Ray-Casting an kleinen/dichten/fernen Zielen scheitert, hilft ein **Selektionsvolumen** statt einer Linie:

- **Cone/Flashlight**: Statt eines Strahls ein **Kegel**; alle Objekte im Kegel sind Kandidaten. Ein Objekt wird gewählt, das den **kleinsten Winkelabstand zur Kegelachse** hat. Erleichtert das Treffen kleiner/ferner Ziele — aber bei **dichten** Szenen sind viele Objekte im Kegel → neue Ambiguität.
- **Bubble Cursor** (Grossman & Balakrishnan, ursprünglich 2D, 3D-Varianten existieren): Ein Cursor mit **dynamischem Radius**, der sich so anpasst, dass **genau ein** Ziel umschlossen wird — er umfasst immer das *nächstgelegene* Objekt vollständig. Macht effektiv jedes Ziel „so groß wie sein Voronoi-Gebiet" → drastisch bessere Fitts-Performance in dünn besetzten Szenen.

Die Wahl zwischen Punkt- und Volumentechnik ist ein **Präzisions-vs-Ambiguitäts-Trade-off**, den das **Final-Projekt** empirisch vermisst: Ray-Casting ist präzise, aber unbrauchbar bei kleinen fernen Zielen; Cone/Bubble greifen die leicht, über-selektieren aber im Gedränge. Die Disambiguierung „welches der Kandidaten-Objekte?" ist konzeptuell dasselbe Problem wie die Referenzauflösung in Modul 18 (dort mit Zeit + Semantik, hier mit Winkel + Distanz).

### 8. Manipulation: DOF, isomorph vs. magisch, HOMER, DOF-Separation

Nach der Selektion folgt oft die **Manipulation** (bewegen, drehen, skalieren). Kernkonzepte:

- **6-DoF-Manipulation isomorph**: Die virtuelle Hand überträgt Translation + Rotation 1:1 aufs Objekt. Natürlich, aber wieder reichweiten- und präzisionsbegrenzt.
- **HOMER** (*Hand-centered Object Manipulation Extending Ray-casting*): selektiert per Ray-Casting (große Reichweite), dann **springt die virtuelle Hand zum Objekt** und manipuliert es hand-zentriert. Kombiniert Ray-Casting-Reichweite mit Virtual-Hand-Manipulation.
- **Scaled-World Grab**: skaliert die Welt beim Greifen so, dass das ferne Objekt in Reichweite kommt — mathematisch elegant, kann aber desorientieren.
- **DOF-Separation**: Oft will man *nicht* alle 6 DoF gleichzeitig kontrollieren (z. B. ein Bild nur *entlang der Wand* verschieben). Techniken **beschränken (constrain)** DoF, **rasten (snap)** an Gitter/Kanten oder trennen Translation von Rotation über Widgets/Handles. Das kompensiert die menschliche Schwäche, viele DoF simultan präzise zu führen.

> **Perzeptuelle Struktur (Jacob et al.):** Freiheitsgrade sollten so gruppiert werden, wie der Mensch sie *wahrnimmt*. Position (x,y,z) ist **integral** (man bewegt die Hand als Ganzes), Position-vs-Farbe wäre **separabel**. Eine Manipulationstechnik, die integrale DoF trennt oder separable koppelt, fühlt sich falsch an. Design-Regel: **DOF-Struktur der Aufgabe = DOF-Struktur der Technik.**

### 9. Travel & Wayfinding

**Travel-Metaphern** (die motorische Fortbewegung):

- **Physical Walking / Real Walking**: am immersivsten, am wenigsten sickness-anfällig — aber durch den realen Raum begrenzt (Redirected Walking als Trick).
- **Steering**: kontinuierliche Richtungsangabe — *gaze-directed* (Blickrichtung) oder *pointing-directed* (Handrichtung). Einfach, aber kontinuierliche visuelle Bewegung ohne körperliche → **Vection → Cybersickness** (Rückverweis Modul 17, Sensory-Conflict-Theorie).
- **Target-based / Teleport**: man wählt ein Ziel, wird **sofort dorthin versetzt**. Der De-facto-Standard in VR, **weil er Cybersickness minimiert** (keine kontinuierliche Vection) — auf Kosten des räumlichen Verständnisses.
- **Manipulation-based** („grab the world"): man greift die Welt und zieht sich hindurch.

**Wayfinding-Unterstützung** (die kognitive Orientierung): **Landmarken**, **Karten** (You-are-here, mit der heiklen Frage der Ausrichtung — track-up vs. north-up), **Kompass/Trails/Breadcrumbs**, gute **Sichtlinien**. Ziel ist der Aufbau einer mentalen **cognitive map**.

---

## Advanced-Themen

### 10. Fitts' Law in 3D: warum Zeigen mit der Distanz zusammenbricht

In Modul 17 haben wir Fitts' Law für 2D-Zeigen eingeführt: die Bewegungszeit

$$MT = a + b\,\underbrace{\log_2\!\Big(\tfrac{D}{W}+1\Big)}_{\text{Index of Difficulty }ID}$$

mit Zieldistanz $D$ und Zielbreite $W$. In 3D — speziell beim **Ray-Casting** — ist die relevante Größe **nicht die lineare, sondern die *anguläre* Ausdehnung**. Ein Ziel der (Quer-)Breite $W$ in Distanz $L$ vom Auge/Controller subtendiert den Winkel

$$\theta_W \approx 2\arctan\!\Big(\frac{W}{2L}\Big) \approx \frac{W}{L} \quad (\text{für } W \ll L),$$

und der Zeiger muss über einen **Winkel** $\theta_D$ schwenken. Das **angulare Fitts' Law** ersetzt Distanz/Breite durch Winkel:

$$MT = a + b\,\log_2\!\Big(\frac{\theta_D}{\theta_W}+1\Big).$$

> **Die entscheidende Konsequenz:** $\theta_W \approx W/L$ **schrumpft mit der Distanz $L$**. Ein Objekt doppelt so weit weg ist *anguär* halb so groß → $ID$ steigt → Selektion dauert länger und wird fehleranfälliger. **Ray-Casting-Präzision verschlechtert sich linear mit der Zieldistanz** — das ist der harte, quantitative Grund, warum ferne kleine Ziele in VR so schwer zu treffen sind, und warum Go-Go (bringt das Ziel „näher") oder Bubble (vergrößert das effektive $\theta_W$) helfen. Das **Medium-Projekt** misst genau diese angulare Fitts-Beziehung.

### 11. Control-Display-Gain und adaptive Verstärkung (PRISM)

**Control-Display (C/D) Gain** ist das Verhältnis von Anzeige-Bewegung zu Kontroll-Bewegung. Gain $>1$ (kleine Handbewegung → große Zeigerbewegung) gibt **Reichweite/Geschwindigkeit**, kostet **Präzision**; Gain $<1$ umgekehrt. Ein fester Gain ist ein Kompromiss. **Adaptive Verfahren** wie **PRISM** (Frees et al.) senken den Gain bei *langsamer* Handbewegung (präzises Zielen) und heben ihn bei schneller (großräumiges Bewegen) — sie nutzen die Handgeschwindigkeit als Absichtsindikator. Das ist die kontinuierliche Verallgemeinerung des Go-Go-Gedankens.

### 12. Der Heisenberg-Effekt der 3D-Interaktion

Ein subtiler, praxisrelevanter Effekt: Beim **Betätigen des Auswahl-Buttons** (Trigger drücken) verwackelt die Hand — der Zeiger springt im Moment des Klicks weg vom Ziel. Je kleiner/ferner das Ziel (kleines $\theta_W$), desto fataler. Gegenmaßnahmen: den Zeiger-Zustand **kurz vor** dem Klick einfrieren, Klick-Ereignisse zeitlich filtern, oder Bestätigung über eine **andere Modalität** (Sprache statt Trigger — wieder Modul 18). Das ist das 3D-Analogon zum Wackeln beim Maus-Klick, nur ohne die stabilisierende Tischauflage viel stärker.

### 13. Ray-Casting-Präzision als stochastisches Modell

Für die Evaluation modelliert man die Zeigepräzision als **angulares Rauschen**: Die reale Zeigerichtung streut gaußförmig um die intendierte, mit Standardabweichung $\sigma_\theta$ (Hand-Tremor + Tracking-Rauschen + Heisenberg). Ein Ziel mit angularem Radius $\theta_W/2$ in Distanz $L$ wird getroffen, wenn die Winkelabweichung kleiner ist:

$$P(\text{Treffer}) = P\!\big(|\epsilon_\theta| < \tfrac{\theta_W}{2}\big) = P\!\big(|\epsilon_\theta| < \tfrac{W}{2L}\big), \quad \epsilon_\theta \sim \mathcal{N}(0,\sigma_\theta^2).$$

Das koppelt direkt an das inverse-Varianz-Denken aus Modul 18: Präzision ist $1/\sigma_\theta^2$, und alles, was $\sigma_\theta$ senkt (Auflage, Stabilisierung, C/D-Gain, Prediction wie in Modul 17), erhöht die Trefferwahrscheinlichkeit. Die **Projekte** bauen genau dieses Modell.

### 14. Evaluation: ISO 9241-9 und Throughput in 3D

Der Standard zur Bewertung von Zeigegeräten/-techniken ist **ISO 9241-9**, dessen Kern der **Throughput** (Durchsatz, in bits/s) ist:

$$TP = \frac{ID_e}{MT}, \qquad ID_e = \log_2\!\Big(\frac{D_e}{W_e}+1\Big),$$

mit **effektiver** Breite $W_e = 4.133\,\sigma_x$ (aus der Streuung der tatsächlichen Klickpositionen — die „effektive" Breite korrigiert für die vom Nutzer real genutzte Genauigkeit, sodass ~96 % der Klicks im Ziel liegen) und effektiver Distanz $D_e$. Throughput fasst **Geschwindigkeit und Genauigkeit in einer Zahl** zusammen und macht Techniken vergleichbar. Typische Aufgaben: **reciprocal tapping** (zwischen zwei Zielen hin und her), **Docking** (ein Objekt in Zielpose bringen — testet Manipulation inkl. Rotation). Die statistische Auswertung (within-subject, Counterbalancing, Effektstärken, Wilcoxon/ANOVA) folgt exakt der Methodik aus **Modul 17** — das **Final-Projekt** wendet sie auf einen Selektionstechnik-Vergleich an.

---

## Zusammenfassung / Cheat-Sheet

**Die vier Aufgaben (Bowman)**: Selektion · Manipulation · Travel (motorisch) · Wayfinding (kognitiv) · [+ System Control].

**Homogene Transformationen**
- Punkt homogen: $\tilde{\mathbf p}=(x,y,z,1)$. Grund: Translation wird als $4\times4$-Matrix schreibbar → alles verkettbar.
- Kette: $\mathbf p_{\text{screen}} = M_{\text{proj}} M_{\text{view}} M_{\text{model}}\,\mathbf p_{\text{obj}}$. Reihenfolge zählt (nicht kommutativ).
- Rigid-Body-Inverse: $M^{-1}=\begin{psmallmatrix}\mathbf R^\top & -\mathbf R^\top\mathbf t\\ 0 & 1\end{psmallmatrix}$ (nutzt $\mathbf R^{-1}=\mathbf R^\top$).

**Ray $\mathbf r(t)=\mathbf o+t\mathbf d$ ($\|\mathbf d\|=1$), Schnitt**
| Objekt | Kern |
|---|---|
| Kugel | $t^2+2bt+c=0$, $b=\mathbf m\cdot\mathbf d$, $c=\|\mathbf m\|^2-R^2$, $\mathbf m=\mathbf o-\mathbf c$; $t=-b-\sqrt{b^2-c}$ |
| AABB | Slabs: Treffer $\iff \max_i\min(t_{i1},t_{i2}) \le \min_i\max(t_{i1},t_{i2})$ |
| Dreieck | Möller-Trumbore: $u,v$ baryzentrisch $\ge0$, $u+v\le1$, $t>0$ |

**Selektionstechniken**: Ray-Casting (Reichweite, aber angular-präzisionsschwach) · Virtual Hand (präzise, kurze Reichweite) · **Go-Go** $r_v=r_r+k(r_r-D)^2$ für $r_r\ge D$ (beides) · Cone/Bubble (gegen kleine/dünne Ziele, aber Ambiguität im Gedränge).

**Fitts in 3D**: angular, $ID=\log_2(\theta_D/\theta_W+1)$ mit $\theta_W\approx W/L$ → **Präzision fällt mit Distanz $L$**.

**Throughput (ISO 9241-9)**: $TP=ID_e/MT$, $W_e=4.133\,\sigma_x$ (effektive Breite aus Klickstreuung).

**Travel & Sickness**: Teleport minimiert Vection/Sickness (Modul 17); Steering ist einfach aber sickness-anfällig; Real Walking am besten, aber raumbegrenzt.

**Design-Regeln**: „magische" (non-isomorphic) Technik schlägt oft die realtreue; DOF-Struktur der Technik = DOF-Struktur der Aufgabe; Heisenberg-Effekt beim Klick behandeln.

---

## Selbsttest

<details>
<summary><b>1.</b> Warum braucht man homogene ($4\times4$) Koordinaten — reichen nicht $3\times3$-Rotationsmatrizen?</summary>

Weil **Translation keine lineare Abbildung** ist (sie bildet den Ursprung nicht auf sich ab) und sich daher *nicht* als $3\times3$-Matrix schreiben lässt. In homogenen Koordinaten (vierte Komponente $w=1$) wird Translation zur $4\times4$-Matrix mit dem Verschiebungsvektor in der letzten Spalte. Damit sind **Rotation, Translation und Skalierung alle Matrizen** und lassen sich durch Multiplikation zu **einer** Transformationskette verketten — das ist der ganze Zweck.
</details>

<details>
<summary><b>2.</b> Nenne die vier universellen 3D-Interaktionsaufgaben und den Unterschied zwischen Travel und Wayfinding.</summary>

**Selektion, Manipulation, Travel, Wayfinding** (+ System Control). **Travel** ist die *motorische* Fortbewegung (das Bewegen des Blickpunkts selbst), **Wayfinding** die *kognitive* Orientierung (wissen, wo man ist und wie man zum Ziel kommt). Zusammen bilden sie *Navigation*.
</details>

<details>
<summary><b>3.</b> Leite die Ray-Kugel-Schnittgleichung her. Wann gibt es keinen Treffer?</summary>

Strahl $\mathbf r(t)=\mathbf o+t\mathbf d$ ($\|\mathbf d\|=1$) in die Kugelgleichung $\|\mathbf p-\mathbf c\|^2=R^2$ einsetzen. Mit $\mathbf m=\mathbf o-\mathbf c$: $\|\mathbf m+t\mathbf d\|^2=R^2 \Rightarrow t^2 + 2t(\mathbf m\cdot\mathbf d) + (\|\mathbf m\|^2-R^2)=0$. Das ist $t^2+2bt+c=0$ mit $b=\mathbf m\cdot\mathbf d$, $c=\|\mathbf m\|^2-R^2$. Diskriminante $\Delta=b^2-c$. **Kein Treffer, wenn $\Delta<0$** (Strahl verfehlt die Kugel). Sonst ist der nähere Schnittpunkt $t=-b-\sqrt\Delta$.
</details>

<details>
<summary><b>4.</b> Schreibe die Go-Go-Funktion auf und erkläre, welches Problem sie löst und warum ihre Form (quadratisch, mit Schwelle) sinnvoll ist.</summary>

$$r_v = \begin{cases} r_r & r_r<D\\ r_r + k(r_r-D)^2 & r_r\ge D\end{cases}$$

Sie löst das **Reichweitenproblem** der Virtual Hand: Im Nahbereich ($r_r<D$) bleibt die Abbildung **1:1** → volle Präzision; jenseits der Schwelle $D$ wächst die virtuelle Reichweite **quadratisch** → kleine reale Armstreckungen erreichen ferne Objekte. Die quadratische Form ist an $r_r=D$ **stetig und mit Ableitung 1 differenzierbar** (kein Sprung, kein Geschwindigkeitsknick) → fühlt sich natürlich an. Go-Go vereint so Nahpräzision und Fernreichweite und ist ein Beispiel für eine „magische" (non-isomorphic) Technik.
</details>

<details>
<summary><b>5.</b> Warum wird Ray-Casting mit zunehmender Zieldistanz schlechter? Nutze das angulare Fitts' Law.</summary>

Weil die für das Zeigen relevante Größe der **anguläre** Radius ist: Ein Ziel der Breite $W$ in Distanz $L$ subtendiert nur $\theta_W\approx W/L$. Das **schrumpft mit $L$** — doppelte Distanz = halbe anguläre Größe. Im angularen Fitts' Law $ID=\log_2(\theta_D/\theta_W+1)$ steigt damit der Schwierigkeitsindex, die Bewegungszeit wächst und (bei festem Winkelrauschen $\sigma_\theta$) die Trefferwahrscheinlichkeit $P(|\epsilon_\theta|<\theta_W/2)$ sinkt. Fazit: Ray-Casting-Präzision fällt mit der Distanz.
</details>

<details>
<summary><b>6.</b> Was ist der Heisenberg-Effekt der 3D-Interaktion, und wie begegnet man ihm?</summary>

Beim **Drücken des Auswahl-Buttons** verwackelt die Hand, sodass der Zeiger im Klick-Moment vom Ziel wegspringt — besonders schlimm bei kleinen/fernen (angular winzigen) Zielen. Gegenmittel: den Zeiger-Zustand **kurz vor dem Klick einfrieren**, Klicks zeitlich filtern, oder die Bestätigung in eine **andere Modalität** verlagern (z. B. Sprachkommando statt Trigger — Modul 18).
</details>

<details>
<summary><b>7.</b> Warum ist Teleport-Travel in VR so verbreitet, und was ist der Preis?</summary>

Weil Teleport (target-based) **keine kontinuierliche visuelle Eigenbewegung** erzeugt → keine **Vection** → minimale **Cybersickness** (Sensory-Conflict-Theorie, Modul 17). Der Preis ist **schlechteres räumliches Verständnis / Wayfinding**: Durch das „Springen" baut der Nutzer eine weniger zusammenhängende cognitive map auf und kann die Distanz/Richtung des Wegs schlechter einschätzen.
</details>

<details>
<summary><b>8.</b> Was ist die „effektive Breite" $W_e$ im ISO-9241-9-Throughput, und warum benutzt man sie statt der Zielbreite $W$?</summary>

$W_e = 4.133\,\sigma_x$ wird aus der **Streuung der tatsächlichen Klickpositionen** berechnet. Sie erfasst die **real vom Nutzer genutzte Genauigkeit**: Zielt jemand schlampig (breite Streuung), steigt $W_e$; zielt er präziser als nötig, sinkt sie. Der Faktor 4.133 normiert so, dass ~96 % der Klicks ins effektive Ziel fallen. Dadurch wird der **Speed-Accuracy-Tradeoff herausgerechnet** — Throughput $TP=ID_e/MT$ bewertet Techniken fair, egal ob ein Nutzer eher schnell-ungenau oder langsam-genau agiert.
</details>

<details>
<summary><b>9.</b> Wann ist ein Bubble Cursor / eine Cone-Technik dem Ray-Casting überlegen, wann nicht?</summary>

**Überlegen** bei **kleinen, fernen oder isolierten** Zielen: Der Bubble Cursor vergrößert das effektive Ziel auf sein Voronoi-Gebiet, die Cone-Technik senkt die nötige Winkelpräzision → beide erleichtern das Treffen dramatisch (bessere Fitts-Performance). **Nicht überlegen** in **dichten** Szenen: Dann liegen viele Objekte im Kegel/Radius, und es entsteht ein **Disambiguierungsproblem** — welches Objekt ist gemeint? Ray-Casting ist dort präziser. Es ist ein Präzisions-vs-Ambiguitäts-Trade-off.
</details>

<details>
<summary><b>10.</b> Was besagt das Prinzip „DOF-Struktur der Technik = DOF-Struktur der Aufgabe"?</summary>

Freiheitsgrade sollten in der Technik **so gruppiert** werden, wie der Mensch sie in der Aufgabe **wahrnimmt und kontrolliert** (integral vs. separabel, nach Jacob et al.). Position (x,y,z) ist *integral* — man bewegt die Hand als Ganzes; solche DoF zu trennen (z. B. jede Achse einzeln stellen zu müssen) fühlt sich falsch an. Umgekehrt sollte man *separable* Aufgabenanteile nicht künstlich koppeln. Praktisch heißt das: Constraints/Widgets/Snapping so wählen, dass sie der natürlichen DoF-Wahrnehmung der Aufgabe entsprechen.
</details>

---

## Literatur & Quellen

**Lehrbücher**
- **LaViola, Kruijff, McMahan, Bowman & Poupyrev, *3D User Interfaces: Theory and Practice* (2. Aufl., 2017).** *Das* Standardwerk — Taxonomie der vier Aufgaben, alle Techniken, Evaluation. Pflicht. *Einsteiger- bis fortgeschrittenenfreundlich.*
- **Foley/van Dam o. ä. Computergrafik-Lehrbuch** für homogene Transformationen und Ray-Objekt-Schnitt; alternativ **Marschner & Shirley, *Fundamentals of Computer Graphics*** (Kap. Transformationen, Ray Tracing). *Einsteigerfreundlich.*
- **Ericson, *Real-Time Collision Detection*** — die maßgebliche Referenz für Ray-Kugel/AABB/Dreieck-Tests. *Vertiefend.*

**Schlüssel-Papers (frei auffindbar)**
- **Poupyrev, Billinghurst, Weghorst & Ichikawa, „The Go-Go Interaction Technique"**, *UIST 1996*. Die Reichweiten-Verlängerung. *Kurz, lesenswert.*
- **Bowman & Hodges, „An Evaluation of Techniques for Grabbing and Manipulating Remote Objects" (HOMER)**, *I3D 1997*. *Vertiefend.*
- **Grossman & Balakrishnan, „The Bubble Cursor"**, *CHI 2005*. Dynamischer Cursor-Radius, Fitts-optimal. *Einsteigerfreundlich.*
- **Möller & Trumbore, „Fast, Minimum Storage Ray-Triangle Intersection"**, *Journal of Graphics Tools 1997*. Der Standard-Algorithmus. *Vertiefend, aber kompakt.*
- **Frees, Kessler & Kay, „PRISM Interaction for Enhancing Control in Immersive Virtual Environments"**, *ACM TOCHI 2007*. Adaptiver C/D-Gain. *Vertiefend.*
- **MacKenzie, „Fitts' Law as a Research and Design Tool in HCI"**, *HCI 1992* — die Referenz für Fitts/Throughput (auch für ISO 9241-9). *Einsteigerfreundlich.*

**Frei verfügbare Kurse / Materialien**
- **Scratchapixel** (scratchapixel.com) — hervorragende, kostenlose Tutorials zu Transformationen und Ray-Object-Intersection mit vollständiger Herleitung. *Kostenlos, einsteigerfreundlich.*
- Diverse **VR/3DUI-Vorlesungen** (z. B. von Doug Bowman / Virginia Tech) mit frei verfügbaren Folien. *Kostenlos, vertiefend.*

**Zum Ausprobieren**
- Die **drei Projekte dieses Moduls** bauen Transformationsketten + Ray-Casting (basic), das angulare Fitts-Modell + Go-Go (medium) und einen kompletten Selektionstechnik-Vergleich unter Clutter mit ISO-Throughput (final) — die beste Vertiefung ist, sie zu implementieren.

---

> **Nächstes Modul:** Modul 20 „3D Point Cloud Processing" — das Verarbeiten von 3D-Punktwolken (Registrierung/ICP, Segmentierung, Feature-Deskriptoren, PointNet). Die 3D-Geometrie und Transformationsmathematik aus diesem Modul (Abschnitt 3) ist die direkte Grundlage.
