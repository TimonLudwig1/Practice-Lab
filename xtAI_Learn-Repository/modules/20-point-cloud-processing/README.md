# Module 20 — 3D Point Cloud Processing

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** A **LiDAR**, a **depth camera** (Kinect/RealSense) or **photogrammetry** does not hand you the world as a clean grid but as a **point cloud** — an unordered set of 3D points $\{\mathbf p_1, \dots, \mathbf p_n\}$, each with optional attributes (colour, intensity, normal). This module covers how to extract meaning from that raw geometry: finding **neighbourhoods**, estimating **normals**, **registering** two scans (the famous **ICP**), **segmenting** planes and objects, and how **deep learning** (PointNet) copes with the unordered structure.
>
> **Prior knowledge**: linear algebra (eigenvalues, SVD, dot/cross product), some statistics. From this repo the following build directly into it: **module 05** (PCA — a normal is a local PCA; eigendecomposition of the covariance), **module 10** (Procrustes alignment — ICP solves one Procrustes problem per iteration), **module 19** (homogeneous/rigid transformations — the output of ICP is a $4\times4$ rigid matrix). **Module 19 is a mandatory preceding module.**

> **Note on the scope.** As with modules 15–19 no official module description is available; I scoped the content myself, closely along standard practice (the Open3D/PCL ecosystem, the registration and segmentation pipelines of robotics/autonomous vehicles) and consistently with the 3D block of this repo. **Deliberately without real sensor hardware and without the Open3D library** (which is missing here): the teachable core is the **algorithms and their mathematics** — kd-tree neighbourhoods, PCA normals, the **SVD solution of ICP**, the **RANSAC statistics**, PointNet's **permutation-invariance principle**. Whoever calls `open3d.registration.icp()` does not understand ICP; whoever derives the Kabsch rotation from the SVD and measures the convergence basin does. All projects work **from scratch** with pure `numpy`/`scipy` on synthetic, reproducible point clouds — CPU seconds.

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

- explain what makes a point cloud **fundamentally hard**: **unorderedness (permutation invariance)**, irregular density, missing topology/connectivity, noise, partial overlap.
- understand the **core neighbourhood operations** (kNN, radius search) and their efficient implementation via a **kd-tree**.
- estimate **normals and curvature** through a **local PCA** (eigendecomposition of the covariance matrix) — including the sign/orientation ambiguity.
- motivate and apply **voxel downsampling** and other sampling strategies (FPS).
- fully derive and implement the **ICP procedure**: the **closed-form SVD solution** of the Procrustes problem (Kabsch, with determinant correction against reflections), the iteration correspondence→solution→application, **point-to-point vs. point-to-plane**, and **why ICP converges only locally**.
- derive **RANSAC** for robust plane estimation — including the **iteration-count formula** $N=\log(1-p)/\log(1-w^s)$ — and combine it with clustering into a **segmentation pipeline**.
- place **feature descriptors** (PFH/FPFH, spin images) and the **global registration pipeline** (feature matching → RANSAC → ICP refinement).
- explain the **PointNet principle**: why a **symmetric function (max pooling)** over per-point MLPs produces permutation invariance, and where its limits are (→ PointNet++).

---

## Basics

### 1. What is a point cloud — and why is it hard?

A point cloud is a **set** $P = \{\mathbf p_1, \dots, \mathbf p_n\}$ with $\mathbf p_i \in \mathbb R^3$ (often plus attributes). The word **set** is the root of every difficulty:

1. **Unordered / permutation invariant.** There is no "first" or "tenth" row as in an image. Two clouds with the same points in a different order are **identical**. Every algorithm (and every neural network!) must be **invariant to the point order** — that is the central design condition (section 13).
2. **Irregular sampling / variable density.** LiDAR points are dense near the sensor and sparse far away. No uniform grid → you cannot apply a standard convolution (module 11).
3. **No topology.** A mesh has edges/faces; a point cloud has only points. "Which points belong to the same surface?" first has to be *derived* (neighbourhoods, normals, segmentation).
4. **Noise, outliers, partial overlap, occlusion.** Real scans are noisy, contain mismeasurements (outliers), and a single scan always sees only *one side* of an object.

These four points motivate practically the entire module.

### 2. Where do point clouds come from?

- **LiDAR** (Light Detection and Ranging): measures the time of flight/phase of laser pulses → distance per beam direction. The core of autonomous vehicles and of surveying.
- **Depth cameras** (structured light: Kinect v1; time-of-flight: Kinect v2, RealSense): deliver a depth image that is converted into 3D points by the camera model (module 19: back-projection with the inverse projection matrix).
- **Stereo / photogrammetry / SfM**: from several 2D images (modules 11/12) via triangulation.
- **Simulation** (as in this module): sampling from known surfaces — which gives *ground truth* for the evaluation.

### 3. Neighbourhoods and the kd-tree

Almost every point cloud operation needs **local neighbourhoods** — "give me the $k$ nearest points to $\mathbf p$" (**kNN**) or "all points within radius $r$" (**radius search**). Naively this costs $O(n)$ per query, hence $O(n^2)$ for all points — untenable with millions of points.

The **kd-tree (k-dimensional tree)** is the standard data structure: a binary tree that recursively splits space along alternating axes at medians. Construction in $O(n\log n)$, a neighbourhood query on average in $O(\log n)$. During a search you prune subtrees whose region is further away than the best candidate so far (branch-and-bound). We use `scipy.spatial.cKDTree` — implementing the data structure itself is not the learning objective, *using* it for kNN/radius is.

> **Note:** The kd-tree is the silent engine underneath almost everything that follows — normal estimation (local neighbourhood), ICP (nearest correspondences), clustering (radius neighbours). Efficient neighbourhood queries are the precondition for everything.

### 4. Downsampling: voxel grid and farthest point sampling

Raw scans are huge and unevenly dense. **Downsampling** reduces the number of points *and* evens out the density:

- **Voxel grid**: lay a 3D grid of edge length $v$ over the cloud; **replace all points inside a voxel by their centroid**. Result: at most one point per occupied voxel, approximately uniform density. The de-facto standard preprocessing step.
- **Farthest point sampling (FPS)**: iteratively pick the point that is **furthest** from the set chosen so far. This yields a well-spread subset of fixed size — used in PointNet++ (section 14).
- **Uniform/random**: simply every $k$-th point or at random — fast, but it does not respect the density.

### 5. Normal estimation: a local PCA

A **surface normal** $\mathbf n_i$ per point is the most important derived quantity (for shading, point-to-plane ICP, features, segmentation). Without a mesh you estimate it from the **local neighbourhood** — and that is **exactly a PCA** (back-reference module 05):

1. Take the $k$ nearest neighbours $\mathcal N(\mathbf p_i)$.
2. Form their centroid $\bar{\mathbf p}$ and the **covariance matrix**
   $$\mathbf C = \frac{1}{|\mathcal N|}\sum_{\mathbf q \in \mathcal N} (\mathbf q - \bar{\mathbf p})(\mathbf q - \bar{\mathbf p})^\top \in \mathbb R^{3\times3}.$$
3. Eigendecomposition $\mathbf C = \sum_j \lambda_j \mathbf v_j \mathbf v_j^\top$ with $\lambda_0 \le \lambda_1 \le \lambda_2$. The local surface spans the directions of the **two large** eigenvectors; the direction of **smallest variance** is **perpendicular** to it — that is the **normal**:
   $$\boxed{\;\mathbf n_i = \mathbf v_0 \quad (\text{eigenvector of the smallest eigenvalue } \lambda_0)\;}$$

**Sign ambiguity.** $\mathbf v_0$ and $-\mathbf v_0$ are both valid eigenvectors — the PCA does not fix the *orientation*. You orient consistently, e.g. **towards the sensor/viewpoint**: if $\mathbf n_i \cdot (\mathbf p_{\text{view}} - \mathbf p_i) < 0$, flip $\mathbf n_i \mapsto -\mathbf n_i$.

**Curvature as a bonus.** The eigenvalues give a curvature measure (surface variation) for free:
$$\sigma = \frac{\lambda_0}{\lambda_0 + \lambda_1 + \lambda_2}.$$
$\sigma \approx 0$: flat (one eigenvalue tiny); $\sigma \to 1/3$: isotropic/noisy/corner. The **basic project** computes normals and curvature exactly this way by hand.

---

## Intermediate

### 6. Registration and ICP — the central problem

**Registration** means: bring two point clouds $P$ (source) and $Q$ (target), which show the same object from **different viewpoints** (partially overlapping), into alignment by a **rigid transformation** $(\mathbf R, \mathbf t)$. Applications: fusing 3D scans into one model, robot localisation (scan against map), SLAM.

The **Iterative Closest Point (ICP)** (Besl & McKay 1992) is the classic. It decomposes into two alternating steps, each solvable on its own:

```
repeat until convergence:
  (1) CORRESPONDENCE:  for each point p_i in P find the nearest point q_{c(i)} in Q  (kd-tree!)
  (2) TRANSFORMATION:  solve (R,t) = argmin  sum_i || R p_i + t - q_{c(i)} ||^2   (closed form, see below)
  (3) apply (R,t) to P; measure the error; if it barely changes -> done
```

The trick: with *fixed* correspondences, step (2) has a **closed-form solution** — that is the Procrustes/Kabsch problem.

### 7. The SVD solution of the Procrustes problem (Kabsch)

Given correspondence pairs $(\mathbf p_i, \mathbf q_i)$, we seek $(\mathbf R, \mathbf t)$ with $\mathbf R^\top\mathbf R = \mathbf I$, $\det\mathbf R = +1$ (a proper rotation), minimising
$$E(\mathbf R, \mathbf t) = \sum_i \|\mathbf R\mathbf p_i + \mathbf t - \mathbf q_i\|^2.$$
The derivation (builds on module 10):

**Step 1 — compute the translation.** For an optimal $\mathbf R$, $\partial E/\partial \mathbf t = 0$ implies that the centroids are mapped onto each other: $\mathbf t = \bar{\mathbf q} - \mathbf R\bar{\mathbf p}$. So **centre** both clouds: $\tilde{\mathbf p}_i = \mathbf p_i - \bar{\mathbf p}$, $\tilde{\mathbf q}_i = \mathbf q_i - \bar{\mathbf q}$. Then only the rotation remains.

**Step 2 — rotation via SVD.** Minimise $\sum_i\|\mathbf R\tilde{\mathbf p}_i - \tilde{\mathbf q}_i\|^2$. Expanding and dropping constant terms shows: this is **equivalent to $\max_{\mathbf R}\ \mathrm{tr}(\mathbf R^\top \mathbf H)$** with the **cross-covariance matrix**
$$\mathbf H = \sum_i \tilde{\mathbf p}_i\,\tilde{\mathbf q}_i^\top \in \mathbb R^{3\times3}.$$
With the singular value decomposition $\mathbf H = \mathbf U\,\mathbf \Sigma\,\mathbf V^\top$ the optimal rotation is
$$\boxed{\;\mathbf R = \mathbf V\,\mathrm{diag}(1,\,1,\,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top,\qquad \mathbf t = \bar{\mathbf q} - \mathbf R\bar{\mathbf p}\;}$$

**The determinant correction** $\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))$ is essential: without it $\mathbf R = \mathbf V\mathbf U^\top$ could be a **reflection** ($\det = -1$) when the data is very noisy/degenerate. The correction enforces $\det\mathbf R = +1$ (a proper rotation, not a reflection). This is the famous **Kabsch formula** (also: Umeyama, when a scale factor is estimated as well) — the same structure as orthogonal Procrustes from module 10, here for the 3D rigid body. The **medium project** implements exactly this solution and builds ICP on top of it.

### 8. ICP convergence, point-to-plane and robustness

**Convergence.** Every ICP step lowers the error **monotonically** (both sub-steps are optimal for their variable) → ICP is guaranteed to converge. **But only to a local minimum**: if you start from a badly wrong initial pose, ICP finds a wrong alignment (the nearest neighbours are then the wrong points). ICP therefore needs a **good initialisation** — provided by global registration (section 12).

**Point-to-plane.** Instead of the point-to-point distance you minimise the distance **along the target normal**:
$$E_{\perp} = \sum_i \big(\mathbf n_{q_i}\cdot(\mathbf R\mathbf p_i + \mathbf t - \mathbf q_i)\big)^2.$$
This lets points **slide along the surface** instead of snapping point-onto-point — which with non-coincident sampling (points never lie exactly on top of each other) **converges considerably faster** (often in a few iterations) and has a **larger convergence basin**. The price: it needs normals (section 5) and is solved by Gauss-Newton linearisation (small angles).

**Robustness against outliers/partial overlap.** Vanilla ICP takes *all* correspondences equally seriously — a single outlier or the non-overlapping part drags the solution away. Remedies:
- **Distance threshold / trimming**: discard correspondence pairs whose distance is too large (**trimmed ICP**, use only the best $x\%$).
- **Robust weights** (Huber/Tukey instead of the squared loss).
- **Reciprocal correspondences** (keep only if $\mathbf p$ is nearest to $\mathbf q$ *and* vice versa).

The **final project** builds segmentation; ICP's robustness mechanisms are covered in the medium project.

### 9. Segmentation I: RANSAC plane estimation

Many scenes consist of **planes** (floor, walls, tables) plus objects. Planes are found robustly with **RANSAC** (RANdom SAmple Consensus, Fischler & Bolles 1981):

```
repeat N times:
  1. draw 3 random points -> they define a plane (normal = cross product of two edge vectors)
  2. count INLIERS: points with |distance to the plane| < tau
  3. keep the model with the most inliers
afterwards: fit the plane by PCA to ALL inliers (refinement)
```

The distance of a point $\mathbf x$ to the plane with normal $\mathbf n$ (unit vector) through $\mathbf p_0$ is $|\mathbf n\cdot(\mathbf x - \mathbf p_0)|$. RANSAC is robust because a model from an **outlier-free minimal sample** (3 points) suffices and is eventually hit over many trials.

**How many iterations $N$?** With an inlier fraction $w$ and minimal sample size $s$ (here $s=3$), the probability that a sample consists **entirely of inliers** is $w^s$. To draw a clean sample **at least once** with confidence $p$ (e.g. 0.99) you need
$$\boxed{\;N = \frac{\log(1-p)}{\log(1 - w^s)}\;}$$
Example $w=0.5$, $s=3$, $p=0.99$: $N = \log(0.01)/\log(1-0.125) \approx 35$. At $w=0.3$ already $\approx 169$. This formula is the justification for why RANSAC gets by with *few* iterations — and it is verified empirically in the **final project**.

### 10. Segmentation II: clustering the objects

After removing the large plane(s), the **object points** remain. These are separated into individual objects by **clustering** — here the spatial structure is the key:

- **Euclidean clustering / region growing**: start at a point, add all points within radius $r$, grow recursively → one cluster; repeat for unvisited points. At its core this is **DBSCAN** (module 05) with a small `min_samples` — points whose $r$-neighbourhoods touch form one object.
- **DBSCAN** (module 05) directly: robust against noise (outliers are marked as "noise"), finds arbitrarily shaped clusters, needs no cluster count.

The pipeline **remove the RANSAC plane → cluster the rest** is the classic "tabletop segmentation" approach of robotic perception. The **final project** builds it completely and evaluates it against ground-truth labels.

### 11. Feature descriptors

For **global** registration (without a good initialisation) and for object recognition you need **local descriptors** that describe the geometry around a point **invariantly to rotation and translation**:

- **PFH / FPFH** (Fast Point Feature Histograms, Rusu et al.): histograms of the **angular relations between the normals** of neighbouring point pairs in the neighbourhood. FPFH is the fast variant that dominates in practice. Two points with a similar FPFH have similar local geometry → candidates for a correspondence.
- **Spin images** (Johnson & Hebert): project the local neighbourhood into a 2D histogram with respect to the normal axis.

---

## Advanced topics

### 12. The global registration pipeline

ICP needs a good initialisation (section 8). It is provided by **global (coarse) registration**, which works *without* an initial pose:

```
1. downsample (voxel) both clouds
2. compute normals + FPFH descriptors
3. FEATURE MATCHING: for each point in P the FPFH-nearest one in Q -> candidate correspondences
4. RANSAC over correspondences: draw 3 matches, solve Kabsch, count inliers -> coarse (R,t)
5. ICP REFINEMENT (point-to-plane) starting from the coarse pose -> precise (R,t)
```

That is the standard (e.g. Open3D's `global_registration` + `icp`). The core: **features give the coarse alignment (large basin of attraction, but imprecise), ICP refines (precise, but small basin)** — a neat coarse-to-fine division of labour. The more recent alternative **FGR** (Fast Global Registration) drops RANSAC and optimises a robust loss directly.

### 13. Deep learning on point clouds: the permutation invariance problem

An image CNN (module 11) presupposes an **ordered grid** — useless for an unordered point set. The core question: **how do you build a neural network $f$ whose output does not change when you reorder the points?** Formally: $f$ must be **permutation invariant**, $f(\{\mathbf x_1,\dots,\mathbf x_n\}) = f(\{\mathbf x_{\pi(1)},\dots,\mathbf x_{\pi(n)}\})$ for every permutation $\pi$.

**PointNet** (Qi et al. 2017) solves this with a **symmetric function**. The central construction:
$$\boxed{\;f(\{\mathbf x_1,\dots,\mathbf x_n\}) = \gamma\!\Big(\underset{i=1,\dots,n}{\text{MAX}}\ h(\mathbf x_i)\Big)\;}$$
- $h$ is a **shared MLP** that maps **each point individually** into a high-dimensional feature space (identical for all points).
- **MAX** is an **element-wise max pooling** over *all* points → a single global feature vector.
- $\gamma$ is another MLP for the final prediction (classification / segmentation).

**Why does this work?** Because **max (like sum/mean) is symmetric**: the maximum of a set does not depend on the order. So the whole pipeline is permutation invariant — *by construction*. Qi et al. even proved that this form can approximate any continuous set-invariant function (universality) provided $h$ is high-dimensional enough. In addition a **T-Net** (a small network predicting a transformation matrix) aligns the input/features (invariance to rigid transformations).

**Limitation and successors.** PointNet aggregates **globally** (one max over *all* points) and therefore captures **no local structure**. **PointNet++** fixes this **hierarchically**: repeated **farthest point sampling** (section 4) → **grouping** of local neighbourhoods (kd-tree/radius) → PointNet on each local group — exactly like a CNN stacks local convolutions, only on points. Further families: **voxel-based** (3D CNN on voxel grids), **graph-based** (DGCNN — kNN graph + graph convolution, back-reference module 16), and point-based transformers.

> **Practical note (module rule):** Training a PointNet on ModelNet/ShapeNet takes GPU hours — on a laptop neither necessary nor sensible. We understand the **principle** (symmetric function → permutation invariance) and can *demonstrate* it on a **tiny** example (a mini max-pooling network learning a set-invariant property) — that is the actual master-level insight, independent of the scale.

### 14. Surface reconstruction and registration theory (briefly)

- **Surface reconstruction** (point cloud → mesh): **Poisson reconstruction** (solves a Poisson equation from the oriented normals — which is why normals matter so much), **ball pivoting**, **marching cubes** on an implicit function.
- **Global optimality of ICP**: vanilla ICP is only local; **Go-ICP** guarantees the **global** optimum via branch-and-bound over the rotation space $SO(3)$ (more expensive). It shows that the dependence on initialisation is a *solvable* but fundamental problem.

---

## Summary / cheat sheet

**What makes a point cloud hard**: unordered (permutation invariant) · irregular density · no topology · noise/outliers/partial overlap.

**Neighbourhoods**: kNN & radius via a **kd-tree** ($O(\log n)$/query). The engine underneath normals, ICP, clustering.

**Normals (local PCA)**: $\mathbf C=\frac1{|\mathcal N|}\sum(\mathbf q-\bar{\mathbf p})(\mathbf q-\bar{\mathbf p})^\top$; $\mathbf n=\mathbf v_0$ (smallest eigenvalue); sign towards the viewer; curvature $\sigma=\lambda_0/\sum\lambda$.

**Downsampling**: voxel grid (centroid per voxel), FPS (evenly spread selection).

**ICP** (iterate): (1) nearest correspondence (kd-tree), (2) solve Kabsch, (3) apply. Converges **monotonically**, but only **locally** → a good init is needed.

**Kabsch/Procrustes (SVD)**: centre; $\mathbf H=\sum\tilde{\mathbf p}_i\tilde{\mathbf q}_i^\top$; $\mathbf H=\mathbf U\Sigma\mathbf V^\top$; $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$; $\mathbf t=\bar{\mathbf q}-\mathbf R\bar{\mathbf p}$. The **det correction** prevents a reflection.

**point-to-plane**: minimise $(\mathbf n_q\cdot(\mathbf R\mathbf p+\mathbf t-\mathbf q))^2$ → faster, larger basin, needs normals.

**RANSAC plane**: 3 points → plane (normal = cross product); inliers $|\mathbf n\cdot(\mathbf x-\mathbf p_0)|<\tau$; iterations $N=\log(1-p)/\log(1-w^s)$.

**Segmentation pipeline**: remove the RANSAC plane → cluster the rest with DBSCAN/Euclidean clustering.

**Global registration**: voxel → FPFH → feature match → RANSAC (coarse) → ICP (fine). Coarse-to-fine.

**PointNet**: $f=\gamma(\mathrm{MAX}_i\,h(\mathbf x_i))$ — shared per-point MLP + **symmetric max pooling** = **permutation invariance by construction**. Limitation: no local structure → **PointNet++** (FPS + grouping, hierarchical).

---

## Self-test

<details>
<summary><b>1.</b> Why can a standard CNN (module 11) not be applied directly to a point cloud, and what is the fundamental property every point cloud network must satisfy?</summary>

A CNN presupposes a **regular, ordered grid** (convolution over fixed neighbouring pixels). A point cloud is **unordered** and **irregularly sampled** — there is no fixed neighbourhood and no order. The fundamental property is **permutation invariance**: the output must not change when the points are reordered, since the same set is the same cloud.
</details>

<details>
<summary><b>2.</b> How do you estimate a point normal, and which quantity of the local PCA gives it? What about the sign?</summary>

You take the **local neighbourhood**, form its **covariance matrix** $\mathbf C$ and decompose it into eigenvalues/eigenvectors. The surface lies along the two largest eigenvectors; the normal is the **eigenvector of the smallest eigenvalue** $\lambda_0$ (direction of minimal variance, perpendicular to the surface). The **sign** is ambiguous (both $\pm\mathbf v_0$ are eigenvectors) — you orient consistently, usually **towards the sensor/viewer** (flip the sign if $\mathbf n\cdot(\mathbf p_{\text{view}}-\mathbf p)<0$).
</details>

<details>
<summary><b>3.</b> Describe the two steps of an ICP iteration. Why does ICP converge, and why only locally?</summary>

(1) **Correspondence**: for each source point find the nearest target point (kd-tree). (2) **Transformation**: with these fixed correspondences solve the optimal $(\mathbf R,\mathbf t)$ via Kabsch/SVD and apply it. ICP **converges** because both steps lower the error **monotonically** (each is optimal for its variable) and the error is bounded below. Only **locally**, because the correspondences depend on the current pose: if you start far from the solution the "nearest" points are the *wrong* ones → ICP locks into a local minimum. That is why ICP needs a good initialisation.
</details>

<details>
<summary><b>4.</b> Trace the role of the SVD in the Kabsch solution: which matrix is decomposed, and what is the determinant correction for?</summary>

After centring both clouds the problem reduces to $\max_{\mathbf R}\mathrm{tr}(\mathbf R^\top\mathbf H)$ with the **cross-covariance** $\mathbf H=\sum_i\tilde{\mathbf p}_i\tilde{\mathbf q}_i^\top$. With $\mathbf H=\mathbf U\Sigma\mathbf V^\top$ the optimal rotation is $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$. The **determinant correction** ensures $\det\mathbf R=+1$ (a proper rotation). Without it $\mathbf V\mathbf U^\top$ could be a **reflection** ($\det=-1$) for noisy/degenerate data, which is not a physically valid rigid body motion.
</details>

<details>
<summary><b>5.</b> What is the advantage of point-to-plane over point-to-point ICP?</summary>

Point-to-plane minimises the distance **along the target normal** instead of point-onto-point. This lets points **slide along the surface** instead of having to snap exactly onto each other — which is realistic (the sample points of two scans never coincide exactly). Consequence: **faster convergence** (often a few iterations) and a **larger convergence basin**. Price: it needs surface normals and is solved in linearised form (Gauss-Newton).
</details>

<details>
<summary><b>6.</b> A RANSAC plane detector runs at an inlier fraction $w=0.4$. How many iterations for 99 % confidence? Formula and order of magnitude.</summary>

$N=\dfrac{\log(1-p)}{\log(1-w^s)}$ with $p=0.99$, $s=3$, $w=0.4$: $w^s=0.064$, $\log(0.01)/\log(0.936)\approx -4.605/-0.0661\approx \mathbf{70}$ iterations. (The formula grows steeply as $w$ falls — at $w=0.2$ already ~570.) Key point: RANSAC gets by with **surprisingly few** iterations as long as the inlier fraction is not tiny.
</details>

<details>
<summary><b>7.</b> Describe the classic "tabletop" segmentation pipeline.</summary>

(1) Find and remove the **RANSAC plane** (the dominant plane = table/floor). (2) Separate the remaining points into individual **object clusters** with **Euclidean clustering / DBSCAN** (points whose radius neighbourhoods touch belong together). Optionally **voxel downsampling** and **normals** beforehand. Result: the floor is separated off, each object is its own cluster.
</details>

<details>
<summary><b>8.</b> Why is max pooling the key to PointNet's permutation invariance? Sketch the architecture.</summary>

$f(\{\mathbf x_i\}) = \gamma(\mathrm{MAX}_i\,h(\mathbf x_i))$: a **shared MLP** $h$ maps each point **individually** into a feature space; an **element-wise max** over *all* points aggregates into a global vector; an MLP $\gamma$ makes the prediction. The **max is symmetric** — its result does not depend on the order of the inputs — so the whole function is **permutation invariant by construction**. (Sum/mean would do in principle as well; max works best empirically.)
</details>

<details>
<summary><b>9.</b> What is PointNet's main weakness, and how does PointNet++ fix it?</summary>

PointNet aggregates with **one** global max over all points and therefore captures **no local structure** (fine geometric detail, local neighbourhoods). **PointNet++** fixes this **hierarchically**: **farthest point sampling** picks centres, around each a **local neighbourhood is grouped** (radius/kNN) on which a small PointNet computes a local feature; this is stacked over several levels — analogous to the local convolutions of a CNN, only on points.
</details>

<details>
<summary><b>10.</b> Why does ICP need a good initialisation, and where does it come from in practice?</summary>

Because ICP converges only **locally**: with a badly wrong starting pose the nearest-neighbour correspondences are wrong and ICP locks into the wrong minimum. The coarse init comes from **global registration**: voxel downsampling → **FPFH descriptors** → feature matching → **RANSAC** over the matches → coarse $(\mathbf R,\mathbf t)$. This coarse pose (large basin, imprecise) is then **refined by ICP** (precise, small basin) — the coarse-to-fine division of labour.
</details>

---

## Literature & sources

**Textbooks / overviews**
- **Rusu & Cousins, "3D is here: Point Cloud Library (PCL)"**, *ICRA 2011*, along with the **PCL tutorials** (pointclouds.org) — the practical reference for normals, segmentation, registration, FPFH. *Beginner-friendly.*
- **The Open3D documentation** (open3d.org) — excellent, well-explained tutorials on ICP, global registration, RANSAC segmentation. Ideal for mirroring the procedures built from scratch here against a mature library. *Beginner to advanced, free.*

**Key papers (freely findable)**
- **Besl & McKay, "A Method for Registration of 3-D Shapes" (ICP)**, *IEEE TPAMI 1992*. The origin of ICP. *In-depth.*
- **Arun, Huang & Blostein, "Least-Squares Fitting of Two 3-D Point Sets"**, *TPAMI 1987* — the SVD solution (Kabsch/Umeyama family). *Compact, the mathematics of section 7.*
- **Chen & Medioni, "Object modelling by registration of multiple range images"**, 1992 — point-to-plane ICP. *In-depth.*
- **Fischler & Bolles, "Random Sample Consensus (RANSAC)"**, *CACM 1981*. The origin of RANSAC. *Beginner-friendly, a classic.*
- **Rusu, Blodow & Beetz, "Fast Point Feature Histograms (FPFH)"**, *ICRA 2009*. *In-depth.*
- **Qi et al., "PointNet: Deep Learning on Point Sets…"**, *CVPR 2017* and **"PointNet++"**, *NeurIPS 2017*. The deep learning foundations (section 13). *In-depth, but very well written — PointNet's universality proof is worth it.*

**Freely available courses / materials**
- **Open3D "Getting Started" and "Pipelines" tutorials** — registration/segmentation step by step. *Free.*
- **Nicolai Nielsen's and various YouTube series on point cloud registration & ICP** and **Scratchapixel** (geometry). *Free.*
- Lectures on **3D computer vision / photogrammetry** (e.g. Cyrill Stachniss, University of Bonn — freely available on YouTube, excellent on ICP/features/SLAM). *Free, in-depth.*

**For hands-on practice**
- The **three projects** build neighbourhoods+normals (basic), ICP with the Kabsch SVD (medium) and a RANSAC+clustering segmentation pipeline (final) — all from scratch, the best way to deepen the material.

---

> **Next module:** Module 21 "Robotics 1" — foundations of robotics (kinematics, motion planning, sensing). The 3D geometry/transformations (module 19) and point cloud perception (this module) are central building blocks of robot perception.

---
---

# Modul 20 — 3D Point Cloud Processing (deutsche Fassung)

> **Worum geht es?** Ein **LiDAR**, eine **Tiefenkamera** (Kinect/RealSense) oder **Photogrammetrie** liefern die Welt nicht als sauberes Gitter, sondern als **Punktwolke (point cloud)** — eine ungeordnete Menge von 3D-Punkten $\{\mathbf p_1, \dots, \mathbf p_n\}$, jeder mit optionalen Attributen (Farbe, Intensität, Normale). Dieses Modul behandelt, wie man aus dieser rohen Geometrie Bedeutung gewinnt: **Nachbarschaften** finden, **Normalen** schätzen, zwei Scans **registrieren** (das berühmte **ICP**), Ebenen und Objekte **segmentieren**, und wie **Deep Learning** (PointNet) mit der ungeordneten Struktur umgeht.
>
> **Vorkenntnisse**: lineare Algebra (Eigenwerte, SVD, Skalarprodukt/Kreuzprodukt), etwas Statistik. Aus diesem Repo bauen direkt auf: **Modul 05** (PCA — Normalen sind eine lokale PCA; Eigenzerlegung der Kovarianz), **Modul 10** (Procrustes-Alignment — ICP löst pro Iteration ein Procrustes-Problem), **Modul 19** (homogene/rigid Transformationen — das Ergebnis von ICP ist eine $4\times4$-Rigid-Matrix). **Modul 19 ist Pflicht-Vormodul.**

> **Hinweis zum Zuschnitt.** Wie bei den Modulen 15–19 liegt keine offizielle Modulbeschreibung vor; ich habe den Inhalt selbst zugeschnitten, eng an der Standardpraxis (Open3D/PCL-Ökosystem, die Registrierungs- und Segmentierungspipelines der Robotik/autonomen Fahrzeuge) und konsistent mit dem 3D-Block dieses Repos. **Bewusst ohne echte Sensorhardware und ohne die Bibliothek Open3D** (die hier fehlt): Der lehrbare Kern sind die **Algorithmen und ihre Mathematik** — kd-Baum-Nachbarschaften, PCA-Normalen, die **SVD-Lösung von ICP**, die **RANSAC-Statistik**, das **Permutationsinvarianz-Prinzip** von PointNet. Wer `open3d.registration.icp()` aufruft, versteht ICP nicht; wer die Kabsch-Rotation aus der SVD herleitet und den Konvergenz-Basin vermisst, schon. Alle Projekte arbeiten **from scratch** mit reinem `numpy`/`scipy` auf synthetischen, reproduzierbaren Punktwolken — CPU-Sekunden.

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

- erklären können, was eine Punktwolke **fundamental schwierig** macht: **Ungeordnetheit (Permutationsinvarianz)**, irreguläre Dichte, fehlende Topologie/Konnektivität, Rauschen, Teilüberlappung.
- die **Kern-Nachbarschaftsoperationen** (kNN, Radius-Suche) und ihre effiziente Umsetzung per **kd-Baum** verstehen.
- **Normalen und Krümmung** über eine **lokale PCA** (Eigenzerlegung der Kovarianzmatrix) schätzen — inklusive der Vorzeichen-/Orientierungsmehrdeutigkeit.
- **Voxel-Downsampling** und andere Sampling-Strategien (FPS) motivieren und anwenden.
- das **ICP-Verfahren** vollständig herleiten und implementieren können: die **geschlossene SVD-Lösung** des Procrustes-Problems (Kabsch, mit Determinanten-Korrektur gegen Spiegelungen), die Iteration Korrespondenz→Lösung→Anwendung, **point-to-point vs. point-to-plane**, und **warum ICP nur lokal konvergiert**.
- **RANSAC** für robuste Ebenenschätzung herleiten — inklusive der **Iterationszahl-Formel** $N=\log(1-p)/\log(1-w^s)$ — und mit Clustering zu einer **Segmentierungspipeline** kombinieren.
- **Feature-Deskriptoren** (PFH/FPFH, Spin Images) und die **globale Registrierungs-Pipeline** (Feature-Matching → RANSAC → ICP-Verfeinerung) einordnen.
- das **PointNet-Prinzip** erklären: warum eine **symmetrische Funktion (Max-Pooling)** über per-Punkt-MLPs Permutationsinvarianz erzeugt, und wo seine Grenzen liegen (→ PointNet++).

---

## Grundlagen (Basics)

### 1. Was ist eine Punktwolke — und warum ist sie schwer?

Eine Punktwolke ist eine **Menge** $P = \{\mathbf p_1, \dots, \mathbf p_n\}$ mit $\mathbf p_i \in \mathbb R^3$ (oft plus Attribute). Das Wort **Menge** ist der Kern aller Schwierigkeiten:

1. **Ungeordnet / permutationsinvariant.** Es gibt keine „erste" oder „zehnte" Zeile wie bei einem Bild. Zwei Wolken mit denselben Punkten in anderer Reihenfolge sind **identisch**. Jeder Algorithmus (und jedes neuronale Netz!) muss **invariant gegenüber der Punktreihenfolge** sein — das ist die zentrale Designbedingung (Abschnitt 13).
2. **Irreguläre Abtastung / variable Dichte.** LiDAR-Punkte sind nah am Sensor dicht, fern dünn. Kein gleichmäßiges Raster → man kann keine Standard-Faltung (Modul 11) anwenden.
3. **Keine Topologie.** Ein Mesh hat Kanten/Flächen; eine Punktwolke nur Punkte. „Welche Punkte gehören zur selben Fläche?" muss man erst *ableiten* (Nachbarschaften, Normalen, Segmentierung).
4. **Rauschen, Ausreißer, Teilüberlappung, Verdeckung.** Echte Scans sind verrauscht, haben Fehlmessungen (Ausreißer), und ein einzelner Scan sieht immer nur *eine Seite* eines Objekts.

Diese vier Punkte motivieren praktisch das gesamte Modul.

### 2. Woher kommen Punktwolken?

- **LiDAR** (Light Detection and Ranging): misst Laufzeit/Phase von Laserpulsen → Distanz pro Strahlrichtung. Kern der autonomen Fahrzeuge und der Vermessung.
- **Tiefenkameras** (structured light: Kinect v1; time-of-flight: Kinect v2, RealSense): liefern ein Tiefenbild, das per Kameramodell (Modul 19: Rückprojektion mit der inversen Projektionsmatrix) in 3D-Punkte umgerechnet wird.
- **Stereo / Photogrammetrie / SfM**: aus mehreren 2D-Bildern (Modul 11/12) durch Triangulation.
- **Simulation** (wie in diesem Modul): aus bekannten Flächen sampeln — erlaubt *ground truth* für die Evaluation.

### 3. Nachbarschaften und der kd-Baum

Fast jede Punktwolken-Operation braucht **lokale Nachbarschaften** — „gib mir die $k$ nächsten Punkte zu $\mathbf p$" (**kNN**) oder „alle Punkte im Radius $r$" (**Radius-Suche**). Naiv kostet das $O(n)$ pro Anfrage, also $O(n^2)$ für alle Punkte — untragbar bei Millionen Punkten.

Der **kd-Baum (k-dimensional tree)** ist die Standard-Datenstruktur: ein binärer Baum, der den Raum rekursiv entlang alternierender Achsen an Medianen aufteilt. Aufbau in $O(n\log n)$, eine Nachbarschaftsanfrage im Mittel in $O(\log n)$. Bei einer Suche schneidet man Teilbäume ab, deren Region weiter als der bisher beste Kandidat entfernt ist (branch-and-bound). Wir nutzen `scipy.spatial.cKDTree` — die Datenstruktur selbst zu implementieren ist nicht das Lernziel, ihre *Verwendung* für kNN/Radius schon.

> **Merke:** Der kd-Baum ist der stille Motor unter fast allem, was folgt — Normalen-Schätzung (lokale Nachbarschaft), ICP (nächste Korrespondenzen), Clustering (Radius-Nachbarn). Effiziente Nachbarschaft = Voraussetzung für alles.

### 4. Downsampling: Voxel-Grid und Farthest Point Sampling

Rohe Scans sind riesig und ungleichmäßig dicht. **Downsampling** reduziert die Punktzahl *und* vergleichmäßigt die Dichte:

- **Voxel-Grid**: Lege ein 3D-Gitter der Kantenlänge $v$ über die Wolke; **ersetze alle Punkte in einem Voxel durch ihren Schwerpunkt** (centroid). Ergebnis: höchstens ein Punkt pro besetztem Voxel, näherungsweise gleichmäßige Dichte. Der De-facto-Standard-Vorverarbeitungsschritt.
- **Farthest Point Sampling (FPS)**: Wähle iterativ den Punkt, der am **weitesten** von der bisher gewählten Menge entfernt ist. Ergibt eine gut verteilte Teilmenge fester Größe — genutzt in PointNet++ (Abschnitt 14).
- **Uniform/Random**: einfach jeden $k$-ten oder zufällig — schnell, aber respektiert die Dichte nicht.

### 5. Normalen-Schätzung: eine lokale PCA

Eine **Oberflächennormale** $\mathbf n_i$ pro Punkt ist die wichtigste abgeleitete Größe (für Beleuchtung, point-to-plane-ICP, Features, Segmentierung). Ohne Mesh schätzt man sie aus der **lokalen Nachbarschaft** — und das ist **exakt eine PCA** (Rückverweis Modul 05):

1. Nimm die $k$ nächsten Nachbarn $\mathcal N(\mathbf p_i)$.
2. Bilde ihren Schwerpunkt $\bar{\mathbf p}$ und die **Kovarianzmatrix**
   $$\mathbf C = \frac{1}{|\mathcal N|}\sum_{\mathbf q \in \mathcal N} (\mathbf q - \bar{\mathbf p})(\mathbf q - \bar{\mathbf p})^\top \in \mathbb R^{3\times3}.$$
3. Eigenzerlegung $\mathbf C = \sum_j \lambda_j \mathbf v_j \mathbf v_j^\top$ mit $\lambda_0 \le \lambda_1 \le \lambda_2$. Die lokale Fläche spannt sich in Richtung der **zwei großen** Eigenvektoren auf; die Richtung der **kleinsten Varianz** steht **senkrecht** darauf — das ist die **Normale**:
   $$\boxed{\;\mathbf n_i = \mathbf v_0 \quad (\text{Eigenvektor zum kleinsten Eigenwert } \lambda_0)\;}$$

**Vorzeichen-Ambiguität.** $\mathbf v_0$ und $-\mathbf v_0$ sind beide gültige Eigenvektoren — die PCA legt die *Orientierung* nicht fest. Man orientiert konsistent, z. B. **zum Sensor/Betrachterpunkt hin**: ist $\mathbf n_i \cdot (\mathbf p_{\text{view}} - \mathbf p_i) < 0$, drehe $\mathbf n_i \mapsto -\mathbf n_i$.

**Krümmung als Bonus.** Die Eigenwerte liefern gratis ein Krümmungsmaß (surface variation):
$$\sigma = \frac{\lambda_0}{\lambda_0 + \lambda_1 + \lambda_2}.$$
$\sigma \approx 0$: flach (ein Eigenwert winzig); $\sigma \to 1/3$: isotrop/verrauscht/Ecke. Das **Basic-Projekt** rechnet Normalen und Krümmung so von Hand.

---

## Aufbau (Intermediate)

### 6. Registrierung und ICP — das zentrale Problem

**Registrierung** heißt: Zwei Punktwolken $P$ (source) und $Q$ (target), die dasselbe Objekt aus **verschiedenen Blickwinkeln** zeigen (teilüberlappend), durch eine **starre Transformation** $(\mathbf R, \mathbf t)$ zur Deckung bringen. Anwendungen: 3D-Scans zu einem Modell fusionieren, Roboter-Lokalisierung (Scan gegen Karte), SLAM.

Das **Iterative Closest Point (ICP)** (Besl & McKay 1992) ist der Klassiker. Es zerfällt in zwei abwechselnde Schritte, die je für sich lösbar sind:

```
wiederhole bis Konvergenz:
  (1) KORRESPONDENZ:  fuer jeden Punkt p_i in P finde den naechsten Punkt q_{c(i)} in Q  (kd-Baum!)
  (2) TRANSFORMATION: loese (R,t) = argmin  sum_i || R p_i + t - q_{c(i)} ||^2   (geschlossen, s.u.)
  (3) wende (R,t) auf P an; miss den Fehler; wenn kaum noch Aenderung -> fertig
```

Der Trick: Schritt (2) hat bei *festen* Korrespondenzen eine **geschlossene Lösung** — das ist das Procrustes/Kabsch-Problem.

### 7. Die SVD-Lösung des Procrustes-Problems (Kabsch)

Gegeben Korrespondenzpaare $(\mathbf p_i, \mathbf q_i)$, gesucht $(\mathbf R, \mathbf t)$ mit $\mathbf R^\top\mathbf R = \mathbf I$, $\det\mathbf R = +1$ (echte Rotation), die
$$E(\mathbf R, \mathbf t) = \sum_i \|\mathbf R\mathbf p_i + \mathbf t - \mathbf q_i\|^2$$
minimiert. Die Herleitung (baut auf Modul 10):

**Schritt 1 — Translation ausrechnen.** Für optimales $\mathbf R$ ergibt $\partial E/\partial \mathbf t = 0$, dass die Schwerpunkte aufeinander abgebildet werden: $\mathbf t = \bar{\mathbf q} - \mathbf R\bar{\mathbf p}$. Also **zentriere** beide Wolken: $\tilde{\mathbf p}_i = \mathbf p_i - \bar{\mathbf p}$, $\tilde{\mathbf q}_i = \mathbf q_i - \bar{\mathbf q}$. Dann bleibt nur noch die Rotation.

**Schritt 2 — Rotation via SVD.** Minimiere $\sum_i\|\mathbf R\tilde{\mathbf p}_i - \tilde{\mathbf q}_i\|^2$. Ausmultiplizieren und Weglassen konstanter Terme zeigt: das ist **äquivalent zu $\max_{\mathbf R}\ \mathrm{tr}(\mathbf R^\top \mathbf H)$** mit der **Kreuz-Kovarianzmatrix**
$$\mathbf H = \sum_i \tilde{\mathbf p}_i\,\tilde{\mathbf q}_i^\top \in \mathbb R^{3\times3}.$$
Mit der Singulärwertzerlegung $\mathbf H = \mathbf U\,\mathbf \Sigma\,\mathbf V^\top$ ist die optimale Rotation
$$\boxed{\;\mathbf R = \mathbf V\,\mathrm{diag}(1,\,1,\,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top,\qquad \mathbf t = \bar{\mathbf q} - \mathbf R\bar{\mathbf p}\;}$$

**Die Determinanten-Korrektur** $\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))$ ist essenziell: Ohne sie könnte $\mathbf R = \mathbf V\mathbf U^\top$ eine **Spiegelung** ($\det = -1$) sein, wenn die Daten sehr verrauscht/entartet sind. Die Korrektur erzwingt $\det\mathbf R = +1$ (echte Rotation, keine Spiegelung). Das ist die berühmte **Kabsch-Formel** (auch: Umeyama, wenn zusätzlich Skalierung geschätzt wird) — dieselbe Struktur wie das orthogonale Procrustes aus Modul 10, hier für 3D-Rigid-Body. Das **Medium-Projekt** implementiert genau diese Lösung und baut ICP darauf.

### 8. ICP-Konvergenz, point-to-plane und Robustheit

**Konvergenz.** Jeder ICP-Schritt senkt den Fehler **monoton** (beide Teilschritte sind optimal für ihre Variable) → ICP konvergiert garantiert. **Aber nur zu einem lokalen Minimum**: Startet man mit einer stark falschen Ausgangspose, findet ICP eine falsche Deckung (die nächsten Nachbarn sind dann die falschen Punkte). ICP braucht daher eine **gute Initialisierung** — die liefert die globale Registrierung (Abschnitt 12).

**Point-to-plane.** Statt Punkt-zu-Punkt-Distanz minimiert man den Abstand **entlang der Ziel-Normalen**:
$$E_{\perp} = \sum_i \big(\mathbf n_{q_i}\cdot(\mathbf R\mathbf p_i + \mathbf t - \mathbf q_i)\big)^2.$$
Das erlaubt den Punkten, **entlang der Oberfläche zu gleiten**, statt Punkt-auf-Punkt zu rasten — was bei nicht deckungsgleicher Abtastung (Punkte liegen nie exakt aufeinander) **deutlich schneller konvergiert** (oft in wenigen Iterationen) und einen **größeren Konvergenz-Basin** hat. Preis: Es braucht Normalen (Abschnitt 5) und wird per Gauß-Newton linearisiert gelöst (kleine Winkel).

**Robustheit gegen Ausreißer/Teilüberlappung.** Vanilla-ICP nimmt *alle* Korrespondenzen gleich ernst — ein einziger Ausreißer oder der nicht-überlappende Teil zieht die Lösung weg. Gegenmittel:
- **Distanzschwelle / Trimming**: verwirf Korrespondenzpaare, deren Abstand zu groß ist (**Trimmed ICP**, nutze nur die besten $x\%$).
- **Robuste Gewichte** (Huber/Tukey statt quadratischer Verlust).
- **Reziproke Korrespondenzen** (nur behalten, wenn $\mathbf p$ nächster zu $\mathbf q$ *und* umgekehrt).

Das **Final-Projekt** baut Segmentierung; die Robustheitsmechanismen von ICP werden im Medium-Projekt behandelt.

### 9. Segmentierung I: RANSAC-Ebenenschätzung

Viele Szenen bestehen aus **Ebenen** (Boden, Wände, Tische) plus Objekten. Ebenen findet man robust mit **RANSAC** (RANdom SAmple Consensus, Fischler & Bolles 1981):

```
wiederhole N mal:
  1. ziehe zufaellig 3 Punkte -> definieren eine Ebene (Normale = Kreuzprodukt zweier Kantenvektoren)
  2. zaehle INLIER: Punkte mit |Abstand zur Ebene| < tau
  3. merke das Modell mit den meisten Inliern
danach: passe die Ebene per PCA an ALLE Inlier an (Verfeinerung)
```

Der Abstand eines Punktes $\mathbf x$ zur Ebene mit Normale $\mathbf n$ (Einheitsvektor) durch $\mathbf p_0$ ist $|\mathbf n\cdot(\mathbf x - \mathbf p_0)|$. RANSAC ist robust, weil ein Modell aus einem **ausreißerfreien Minimalsample** (3 Punkte) genügt und über viele Versuche irgendwann getroffen wird.

**Wie viele Iterationen $N$?** Bei einem Inlier-Anteil $w$ und Minimalsample-Größe $s$ (hier $s=3$) ist die Wahrscheinlichkeit, dass ein Sample **komplett aus Inliern** besteht, $w^s$. Um mit Konfidenz $p$ (z. B. 0.99) **mindestens einmal** ein sauberes Sample zu ziehen, braucht man
$$\boxed{\;N = \frac{\log(1-p)}{\log(1 - w^s)}\;}$$
Beispiel $w=0.5$, $s=3$, $p=0.99$: $N = \log(0.01)/\log(1-0.125) \approx 35$. Bei $w=0.3$ schon $\approx 169$. Diese Formel ist die Rechtfertigung, warum RANSAC mit *wenigen* Iterationen auskommt — und wird im **Final-Projekt** empirisch verifiziert.

### 10. Segmentierung II: Clustering der Objekte

Nach dem Entfernen der großen Ebene(n) bleiben die **Objekt-Punkte**. Diese trennt man in einzelne Objekte per **Clustering** — hier ist die räumliche Struktur der Schlüssel:

- **Euclidean Clustering / Region Growing**: Starte an einem Punkt, füge alle Punkte im Radius $r$ hinzu, wachse rekursiv → ein Cluster; wiederhole für unbesuchte Punkte. Das ist im Kern **DBSCAN** (Modul 05) mit `min_samples` klein — Punkte, deren $r$-Nachbarschaften sich berühren, bilden ein Objekt.
- **DBSCAN** (Modul 05) direkt: robust gegen Rauschen (Ausreißer werden als „noise" markiert), findet beliebig geformte Cluster, braucht keine Clusterzahl.

Die Pipeline **RANSAC-Ebene entfernen → Rest clustern** ist der klassische „tabletop segmentation"-Ansatz der Robotik-Perzeption. Das **Final-Projekt** baut sie vollständig und evaluiert sie gegen ground-truth-Labels.

### 11. Feature-Deskriptoren

Für die **globale** Registrierung (ohne gute Initialisierung) und für Objekterkennung braucht man **lokale Deskriptoren**, die die Geometrie um einen Punkt **rotations- und translationsinvariant** beschreiben:

- **PFH / FPFH** (Fast Point Feature Histograms, Rusu et al.): Histogramme der **Winkelbeziehungen zwischen den Normalen** benachbarter Punktpaare in der Nachbarschaft. FPFH ist die schnelle, in der Praxis dominante Variante. Zwei Punkte mit ähnlichem FPFH haben ähnliche lokale Geometrie → Kandidaten für eine Korrespondenz.
- **Spin Images** (Johnson & Hebert): projizieren die lokale Nachbarschaft in ein 2D-Histogramm bzgl. der Normalen-Achse.

---

## Advanced-Themen

### 12. Die globale Registrierungs-Pipeline

ICP braucht eine gute Initialisierung (Abschnitt 8). Diese liefert die **globale (grobe) Registrierung**, die *ohne* Anfangspose auskommt:

```
1. Downsampling (Voxel) beider Wolken
2. Normalen + FPFH-Deskriptoren berechnen
3. FEATURE-MATCHING: fuer jeden Punkt in P den FPFH-naechsten in Q -> Kandidaten-Korrespondenzen
4. RANSAC ueber Korrespondenzen: ziehe 3 Matches, loese Kabsch, zaehle Inlier -> grobe (R,t)
5. ICP-VERFEINERUNG (point-to-plane) ausgehend von der groben Pose -> praezise (R,t)
```

Das ist der Standard (z. B. Open3Ds `global_registration` + `icp`). Der Kern: **Features geben die grobe Ausrichtung (großer Einzugsbereich, aber ungenau), ICP verfeinert (präzise, aber kleiner Einzugsbereich)** — eine schöne Arbeitsteilung „grob-zu-fein". Die neuere Alternative **FGR** (Fast Global Registration) verzichtet auf RANSAC und optimiert einen robusten Verlust direkt.

### 13. Deep Learning auf Punktwolken: das Permutationsinvarianz-Problem

Ein Bild-CNN (Modul 11) setzt ein **geordnetes Gitter** voraus — für eine ungeordnete Punktmenge unbrauchbar. Die Kernfrage: **Wie baut man ein neuronales Netz $f$, dessen Ausgabe sich nicht ändert, wenn man die Punkte umordnet?** Formal: $f$ muss **permutationsinvariant** sein, $f(\{\mathbf x_1,\dots,\mathbf x_n\}) = f(\{\mathbf x_{\pi(1)},\dots,\mathbf x_{\pi(n)}\})$ für jede Permutation $\pi$.

**PointNet** (Qi et al. 2017) löst das mit einer **symmetrischen Funktion**. Die zentrale Konstruktion:
$$\boxed{\;f(\{\mathbf x_1,\dots,\mathbf x_n\}) = \gamma\!\Big(\underset{i=1,\dots,n}{\text{MAX}}\ h(\mathbf x_i)\Big)\;}$$
- $h$ ist ein **geteiltes MLP**, das **jeden Punkt einzeln** in einen hochdimensionalen Merkmalsraum abbildet (identisch für alle Punkte).
- **MAX** ist ein **elementweises Max-Pooling** über *alle* Punkte → ein einziger globaler Merkmalsvektor.
- $\gamma$ ist ein weiteres MLP für die finale Vorhersage (Klasse / Segmentierung).

**Warum funktioniert das?** Weil **Max (wie Summe/Mittel) symmetrisch** ist: Das Maximum einer Menge hängt nicht von der Reihenfolge ab. Also ist die ganze Pipeline permutationsinvariant — *by construction*. Qi et al. bewiesen sogar, dass diese Form jede stetige mengeninvariante Funktion approximieren kann (Universalität), wenn $h$ hochdimensional genug ist. Zusätzlich richtet ein **T-Net** (ein kleines Netz, das eine Transformationsmatrix vorhersagt) die Eingabe/Features aus (Invarianz gegen Rigid-Transformationen).

**Grenze und Nachfolger.** PointNet aggregiert **global** (ein Max über *alle* Punkte) und erfasst deshalb **keine lokale Struktur**. **PointNet++** behebt das **hierarchisch**: wiederholt **Farthest Point Sampling** (Abschnitt 4) → **Gruppierung** lokaler Nachbarschaften (kd-Baum/Radius) → PointNet auf jeder lokalen Gruppe — genau wie ein CNN lokale Faltungen stapelt, nur auf Punkten. Weitere Familien: **voxel-basiert** (3D-CNN auf Voxelgittern), **graph-basiert** (DGCNN — kNN-Graph + Graph-Convolution, Rückverweis Modul 16), und punktbasierte Transformer.

> **Praxis-Hinweis (Modul-Regel):** Ein PointNet auf ModelNet/ShapeNet zu trainieren braucht GPU-Stunden — auf einem Laptop weder nötig noch sinnvoll. Wir verstehen das **Prinzip** (symmetrische Funktion → Permutationsinvarianz) und können es an einem **winzigen** Beispiel *demonstrieren* (ein Mini-Max-Pooling-Netz, das eine mengeninvariante Eigenschaft lernt) — das ist die eigentliche Master-Einsicht, unabhängig von der Skalierung.

### 14. Oberflächen-Rekonstruktion und Registrierungstheorie (kurz)

- **Oberflächen-Rekonstruktion** (Punktwolke → Mesh): **Poisson-Rekonstruktion** (löst eine Poisson-Gleichung aus den orientierten Normalen — deshalb sind Normalen so wichtig), **Ball-Pivoting**, **Marching Cubes** auf einer impliziten Funktion.
- **Globale Optimalität von ICP**: Vanilla-ICP ist nur lokal; **Go-ICP** garantiert per Branch-and-Bound über den Rotationsraum $SO(3)$ das **globale** Optimum (teurer). Zeigt, dass die Initialisierungsabhängigkeit ein *lösbares*, aber grundsätzliches Problem ist.

---

## Zusammenfassung / Cheat-Sheet

**Was eine Punktwolke schwer macht**: ungeordnet (permutationsinvariant) · irreguläre Dichte · keine Topologie · Rauschen/Ausreißer/Teilüberlappung.

**Nachbarschaften**: kNN & Radius via **kd-Baum** ($O(\log n)$/Anfrage). Motor unter Normalen, ICP, Clustering.

**Normalen (lokale PCA)**: $\mathbf C=\frac1{|\mathcal N|}\sum(\mathbf q-\bar{\mathbf p})(\mathbf q-\bar{\mathbf p})^\top$; $\mathbf n=\mathbf v_0$ (kleinster Eigenwert); Vorzeichen zum Betrachter; Krümmung $\sigma=\lambda_0/\sum\lambda$.

**Downsampling**: Voxel-Grid (Schwerpunkt/Voxel), FPS (gleichverteilte Auswahl).

**ICP** (iteriere): (1) nächste Korrespondenz (kd-Baum), (2) Kabsch lösen, (3) anwenden. Konvergiert **monoton**, aber nur **lokal** → gute Init nötig.

**Kabsch/Procrustes (SVD)**: zentrieren; $\mathbf H=\sum\tilde{\mathbf p}_i\tilde{\mathbf q}_i^\top$; $\mathbf H=\mathbf U\Sigma\mathbf V^\top$; $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$; $\mathbf t=\bar{\mathbf q}-\mathbf R\bar{\mathbf p}$. **Det-Korrektur** verhindert Spiegelung.

**point-to-plane**: minimiere $(\mathbf n_q\cdot(\mathbf R\mathbf p+\mathbf t-\mathbf q))^2$ → schneller, größerer Basin, braucht Normalen.

**RANSAC-Ebene**: 3 Punkte → Ebene (Normale = Kreuzprodukt); Inlier $|\mathbf n\cdot(\mathbf x-\mathbf p_0)|<\tau$; Iterationen $N=\log(1-p)/\log(1-w^s)$.

**Segmentierungspipeline**: RANSAC-Ebene entfernen → Rest per DBSCAN/Euclidean clustern.

**Globale Registrierung**: Voxel → FPFH → Feature-Match → RANSAC (grob) → ICP (fein). Grob-zu-fein.

**PointNet**: $f=\gamma(\mathrm{MAX}_i\,h(\mathbf x_i))$ — geteiltes per-Punkt-MLP + **symmetrisches Max-Pooling** = **Permutationsinvarianz by construction**. Grenze: keine lokale Struktur → **PointNet++** (FPS + Gruppierung, hierarchisch).

---

## Selbsttest

<details>
<summary><b>1.</b> Warum kann man ein Standard-CNN (Modul 11) nicht direkt auf eine Punktwolke anwenden, und was ist die fundamentale Eigenschaft, die jedes Punktwolken-Netz erfüllen muss?</summary>

Ein CNN setzt ein **reguläres, geordnetes Gitter** voraus (Faltung über feste Nachbar-Pixel). Eine Punktwolke ist **ungeordnet** und **irregulär abgetastet** — es gibt keine feste Nachbarschaft und keine Reihenfolge. Die fundamentale Eigenschaft ist **Permutationsinvarianz**: Die Ausgabe darf sich nicht ändern, wenn man die Punkte umordnet, da dieselbe Menge dieselbe Wolke ist.
</details>

<details>
<summary><b>2.</b> Wie schätzt man eine Punktnormale, und welche Größe der lokalen PCA liefert sie? Was ist mit dem Vorzeichen?</summary>

Man nimmt die **lokale Nachbarschaft**, bildet ihre **Kovarianzmatrix** $\mathbf C$ und zerlegt sie in Eigenwerte/-vektoren. Die Fläche liegt in Richtung der zwei größten Eigenvektoren; die Normale ist der **Eigenvektor zum kleinsten Eigenwert** $\lambda_0$ (Richtung minimaler Varianz, senkrecht zur Fläche). Das **Vorzeichen** ist mehrdeutig ($\pm\mathbf v_0$ sind beide Eigenvektoren) — man orientiert konsistent, meist **zum Sensor/Betrachter hin** (Vorzeichen umdrehen, falls $\mathbf n\cdot(\mathbf p_{\text{view}}-\mathbf p)<0$).
</details>

<details>
<summary><b>3.</b> Beschreibe die zwei Schritte einer ICP-Iteration. Warum konvergiert ICP, und warum nur lokal?</summary>

(1) **Korrespondenz**: Für jeden Quellpunkt den nächsten Zielpunkt suchen (kd-Baum). (2) **Transformation**: Bei diesen festen Korrespondenzen die optimale $(\mathbf R,\mathbf t)$ per Kabsch/SVD lösen und anwenden. ICP **konvergiert**, weil beide Schritte den Fehler **monoton senken** (jeder ist optimal für seine Variable) und der Fehler nach unten beschränkt ist. Nur **lokal**, weil die Korrespondenzen von der aktuellen Pose abhängen: Startet man weit von der Lösung, sind die „nächsten" Punkte die *falschen* → ICP rastet in einem lokalen Minimum ein. Deshalb braucht ICP eine gute Initialisierung.
</details>

<details>
<summary><b>4.</b> Leite die Rolle der SVD in der Kabsch-Lösung nach: Welche Matrix wird zerlegt, und wofür ist die Determinanten-Korrektur?</summary>

Nach dem Zentrieren beider Wolken reduziert sich das Problem auf $\max_{\mathbf R}\mathrm{tr}(\mathbf R^\top\mathbf H)$ mit der **Kreuz-Kovarianz** $\mathbf H=\sum_i\tilde{\mathbf p}_i\tilde{\mathbf q}_i^\top$. Mit $\mathbf H=\mathbf U\Sigma\mathbf V^\top$ ist die optimale Rotation $\mathbf R=\mathbf V\,\mathrm{diag}(1,1,\det(\mathbf V\mathbf U^\top))\,\mathbf U^\top$. Die **Determinanten-Korrektur** stellt sicher, dass $\det\mathbf R=+1$ ist (echte Rotation). Ohne sie könnte $\mathbf V\mathbf U^\top$ bei verrauschten/entarteten Daten eine **Spiegelung** ($\det=-1$) sein, die physikalisch keine gültige Starrkörperbewegung ist.
</details>

<details>
<summary><b>5.</b> Was ist der Vorteil von point-to-plane gegenüber point-to-point ICP?</summary>

Point-to-plane minimiert den Abstand **entlang der Ziel-Normalen** statt Punkt-auf-Punkt. Das lässt die Punkte **entlang der Oberfläche gleiten**, statt exakt aufeinander rasten zu müssen — was realistisch ist (die Abtastpunkte zweier Scans liegen nie exakt übereinander). Folge: **schnellere Konvergenz** (oft wenige Iterationen) und ein **größerer Konvergenz-Basin**. Preis: Es braucht Oberflächennormalen und wird linearisiert (Gauß-Newton) gelöst.
</details>

<details>
<summary><b>6.</b> Ein RANSAC-Ebenendetektor läuft bei Inlier-Anteil $w=0.4$. Wie viele Iterationen für 99 % Konfidenz? Formel und Größenordnung.</summary>

$N=\dfrac{\log(1-p)}{\log(1-w^s)}$ mit $p=0.99$, $s=3$, $w=0.4$: $w^s=0.064$, $\log(0.01)/\log(0.936)\approx -4.605/-0.0661\approx \mathbf{70}$ Iterationen. (Die Formel wächst stark, wenn $w$ sinkt — bei $w=0.2$ schon ~570.) Kernpunkt: RANSAC kommt mit **überraschend wenigen** Iterationen aus, solange der Inlier-Anteil nicht winzig ist.
</details>

<details>
<summary><b>7.</b> Beschreibe die klassische „tabletop"-Segmentierungspipeline.</summary>

(1) **RANSAC-Ebene** finden und entfernen (die dominante Ebene = Tisch/Boden). (2) Die verbleibenden Punkte per **Euclidean Clustering / DBSCAN** in einzelne **Objekt-Cluster** trennen (Punkte, deren Radius-Nachbarschaften sich berühren, gehören zusammen). Optional vorher **Voxel-Downsampling** und **Normalen**. Ergebnis: Boden abgetrennt, jedes Objekt ein eigenes Cluster.
</details>

<details>
<summary><b>8.</b> Warum ist Max-Pooling der Schlüssel zu PointNets Permutationsinvarianz? Skizziere die Architektur.</summary>

$f(\{\mathbf x_i\}) = \gamma(\mathrm{MAX}_i\,h(\mathbf x_i))$: Ein **geteiltes MLP** $h$ bildet jeden Punkt **einzeln** in einen Merkmalsraum ab; ein **elementweises Max** über *alle* Punkte aggregiert zu einem globalen Vektor; ein MLP $\gamma$ macht die Vorhersage. Das **Max ist symmetrisch** — sein Ergebnis hängt nicht von der Reihenfolge der Eingaben ab —, also ist die ganze Funktion **permutationsinvariant by construction**. (Summe/Mittel täten es prinzipiell auch; Max funktioniert empirisch am besten.)
</details>

<details>
<summary><b>9.</b> Was ist PointNets Hauptschwäche, und wie behebt PointNet++ sie?</summary>

PointNet aggregiert mit **einem** globalen Max über alle Punkte und erfasst deshalb **keine lokale Struktur** (feine geometrische Details, lokale Nachbarschaften). **PointNet++** behebt das **hierarchisch**: **Farthest Point Sampling** wählt Zentren, um jedes wird eine **lokale Nachbarschaft gruppiert** (Radius/kNN), auf der ein kleines PointNet ein lokales Merkmal berechnet; das wird über mehrere Ebenen gestapelt — analog zu den lokalen Faltungen eines CNN, nur auf Punkten.
</details>

<details>
<summary><b>10.</b> Warum braucht ICP eine gute Initialisierung, und woher kommt sie in der Praxis?</summary>

Weil ICP nur **lokal** konvergiert: Bei stark falscher Startpose sind die nächsten-Nachbar-Korrespondenzen falsch, und ICP rastet im falschen Minimum ein. Die grobe Init liefert die **globale Registrierung**: Voxel-Downsampling → **FPFH-Deskriptoren** → Feature-Matching → **RANSAC** über die Matches → grobe $(\mathbf R,\mathbf t)$. Diese grobe Pose (großer Einzugsbereich, ungenau) wird dann per **ICP verfeinert** (präzise, kleiner Einzugsbereich) — die Grob-zu-fein-Arbeitsteilung.
</details>

---

## Literatur & Quellen

**Lehrbücher / Übersichten**
- **Rusu & Cousins, „3D is here: Point Cloud Library (PCL)"**, *ICRA 2011*, sowie die **PCL-Tutorials** (pointclouds.org) — die praktische Referenz für Normalen, Segmentierung, Registrierung, FPFH. *Einsteigerfreundlich.*
- **Die Open3D-Dokumentation** (open3d.org) — exzellente, gut erklärte Tutorials zu ICP, globaler Registrierung, RANSAC-Segmentierung. Ideal, um die hier from-scratch gebauten Verfahren gegen eine ausgereifte Bibliothek zu spiegeln. *Einsteiger- bis fortgeschrittenenfreundlich, kostenlos.*

**Schlüssel-Papers (frei auffindbar)**
- **Besl & McKay, „A Method for Registration of 3-D Shapes" (ICP)**, *IEEE TPAMI 1992*. Der ICP-Ursprung. *Vertiefend.*
- **Arun, Huang & Blostein, „Least-Squares Fitting of Two 3-D Point Sets"**, *TPAMI 1987* — die SVD-Lösung (Kabsch/Umeyama-Familie). *Kompakt, die Mathematik aus Abschnitt 7.*
- **Chen & Medioni, „Object modelling by registration of multiple range images"**, 1992 — point-to-plane ICP. *Vertiefend.*
- **Fischler & Bolles, „Random Sample Consensus (RANSAC)"**, *CACM 1981*. Der RANSAC-Ursprung. *Einsteigerfreundlich, klassisch.*
- **Rusu, Blodow & Beetz, „Fast Point Feature Histograms (FPFH)"**, *ICRA 2009*. *Vertiefend.*
- **Qi et al., „PointNet: Deep Learning on Point Sets…"**, *CVPR 2017* und **„PointNet++"**, *NeurIPS 2017*. Die Deep-Learning-Grundlagen (Abschnitt 13). *Vertiefend, aber sehr gut geschrieben — der PointNet-Beweis der Universalität lohnt sich.*

**Frei verfügbare Kurse / Materialien**
- **Open3D „Getting Started"- und „Pipelines"-Tutorials** — Schritt für Schritt Registrierung/Segmentierung. *Kostenlos.*
- **Nicolai Nielsens / diverse YouTube-Serien zu Point Cloud Registration & ICP** und **Scratchapixel** (Geometrie). *Kostenlos.*
- Vorlesungen zu **3D Computer Vision / Photogrammetrie** (z. B. Cyrill Stachniss, Uni Bonn — frei auf YouTube, exzellent zu ICP/Features/SLAM). *Kostenlos, vertiefend.*

**Zum Ausprobieren**
- Die **drei Projekte** bauen Nachbarschaften+Normalen (basic), ICP mit Kabsch-SVD (medium) und eine RANSAC+Clustering-Segmentierungspipeline (final) — alles from scratch, die beste Vertiefung.

---

> **Nächstes Modul:** Modul 21 „Robotics 1" — Grundlagen der Robotik (Kinematik, Bewegungsplanung, Sensorik). Die 3D-Geometrie/Transformationen (Modul 19) und Punktwolken-Perzeption (dieses Modul) sind zentrale Bausteine der Roboter-Wahrnehmung.
