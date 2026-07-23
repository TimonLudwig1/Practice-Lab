# Module 17 — Core XR: Principles of Interactive Systems

> **Language note.** English first, German version (*deutsche Fassung*) below the horizontal rule. The projects themselves are English only.

> **What is this about?** **XR** (extended reality — VR, AR, MR) is the attempt to fool a human
> into believing they are somewhere else. The astonishing thing about it: whether that succeeds is
> hardly decided by the graphics. It is decided by **milliseconds** and by **mathematics** — by
> the question of whether the image reacts fast and correctly enough to a head movement. If it is
> 20 ms too late, the user gets sick. This module covers the **principles** behind that:
> perception, **tracking** and rotation mathematics, **motion-to-photon latency**, interaction
> techniques, **cybersickness** — and how to evaluate interactive systems **empirically**, because
> the only authority that judges XR is a human being.

**Helpful prior knowledge:** linear algebra (matrices, vectors, change of basis), trigonometry,
some statistics.

**Modules you should have done first:**
- **Modules 02/03 (data science)** — for the evaluation part: EDA, hypothesis tests, bootstrap.
  Section 5.2 ties in directly with the A/B test logic from module 03.
- Otherwise **none**. This module starts a new field (block F) and does not build on ML/RL.

> **Note on how the content was scoped.** As with modules 15/16 no official module description was
> available. I scope "Core XR: Principles of Interactive Systems" to the **principles** that hold
> independently of the hardware and that you can genuinely *penetrate* without a VR headset:
> **perception, tracking mathematics, latency, interaction, cybersickness, evaluation**.
>
> **Tooling decision:** in this environment there is **no VR hardware** and **no 3D engine**
> (`open3d`, `trimesh`, `pygame` are missing). That is less of a problem than it sounds: the
> intellectual core of XR is **mathematics and timing behaviour**, and both can be recomputed
> exactly and **tested** with `numpy`/`scipy` — better, in fact, than in an engine, where
> everything disappears behind ready-made functions. `scipy.spatial.transform.Rotation` (incl.
> `Slerp`) is available; where an engine would be needed (rendering, shaders) I explain
> **theoretically**. For practical work: **Unity + OpenXR** is the industry standard.

---

## Learning objectives

After this module you can …

- place XR in the **reality-virtuality continuum** (Milgram) and distinguish **immersion**
  (technical) from **presence** (psychological) — including Slater's *place illusion* /
  *plausibility illusion*;
- name the relevant properties of the **human perceptual system** (FoV, resolution, the
  **vergence-accommodation conflict**, the vestibular system, proprioception) and derive design
  requirements from them;
- classify **tracking**: 3 vs. **6 DoF**, outside-in vs. inside-out, SLAM, IMU sensors — and
  explain why a gyroscope **drifts** and how **sensor fusion** repairs that;
- handle **rotations** confidently: Euler angles and their **gimbal lock**, rotation matrices,
  **quaternions** (with a derivation), **SLERP** — and justify why XR uses quaternions;
- decompose the **motion-to-photon latency** into its links, compute its budget and explain the
  countermeasures (**prediction**, **timewarp/reprojection**);
- compare **interaction techniques** (ray-casting, Go-Go, locomotion) and apply **Fitts' law**;
- explain **cybersickness** via the **sensory conflict theory** and justify countermeasures;
- plan and evaluate a **user study** methodologically soundly: within-/between-subject,
  established questionnaires (**IPQ, SSQ, SUS, NASA-TLX**), appropriate tests, **effect size**,
  multiple comparisons.

---

## 1 · Basics — what makes XR

### 1.1 The reality-virtuality continuum

**Milgram & Kishino (1994)** arrange everything on one axis:

```
Real             Augmented        Augmented         Virtual
environment ───  reality (AR) ─── virtuality  ───   environment (VR)
   |                |                |                 |
 nothing        virtual in       real in           everything
 virtual        real             virtual           virtual
        └──────── Mixed Reality (MR) ────────┘
```

**XR** is the umbrella term for all of it. The difference is not merely gradual — it changes the
**requirements**: in **VR** you have to deliver the whole world, but nobody sees whether it stands
*wrongly* relative to reality. In **AR** reality is the reference — the virtual has to be
**registered** (stay in the right place), and even 1 mm of offset or 5 ms of delay is immediately
noticeable, because the real object lies right next to it. **AR is therefore technically harder
than VR.**

### 1.2 Immersion ≠ presence

The most important conceptual distinction of the module name (**Slater**):

- **Immersion** is an **objective property of the technology**: FoV, resolution, latency, tracking
  accuracy, the number of senses addressed. Measurable, readable from a data sheet.
- **Presence** is the **subjective reaction of the human**: *the feeling of being there*. Not
  measurable except through the user themselves (→ section 5).

Slater decomposes presence further into two illusions that can collapse **independently** of each
other:
- **Place illusion (PI)** — "I am in this place." It arises from **sensorimotor contingency**: I
  move my head and the world reacts **the way it would in reality**. PI depends almost entirely on
  **tracking and latency** — not on the graphics.
- **Plausibility illusion (Psi)** — "What is happening here is really happening." It arises when
  the world **reacts to me** and behaves credibly.

> **The central insight of the module:** a graphically primitive but perfectly tracked, low-latency
> comic-book room produces **more presence** than a photorealistic scene that lags 50 ms behind.
> **That is why this module is about mathematics and milliseconds, not about shaders.** PI
> collapses immediately when sensorimotor contingency is violated — and the body notices that with
> merciless reliability.

### 1.3 The human as a system specification

XR does not build for displays but for a **perceptual system**. Its key figures *are* the
requirements:

| Quantity | Human | typical headset | Consequence |
|---|---|---|---|
| **Field of view (FoV)** | ~200–220° horizontal (binocular ~114° overlapping) | ~90–110° | the "diving mask effect", weakens presence |
| **Resolution** | ~60 pixels/degree (fovea) | ~15–35 pixels/degree | screen-door effect, blurry distance |
| **Temporal resolution** | flicker visible up to ~60–90 Hz, latency noticeable from ~20 ms | 90–120 Hz | **the hard criterion** (section 3) |
| **Stereo depth** | vergence + accommodation **coupled** | accommodation fixed at ~2 m | **the vergence-accommodation conflict** |

**The vergence-accommodation conflict (VAC)** deserves its own explanation, because it is a
**physically unsolvable** problem of conventional headsets:
- **Vergence** = both eyes rotate inwards in order to fixate a near object.
- **Accommodation** = the lens of the eye focuses.

In reality the two are **firmly coupled** (what I fixate, I focus on). In a headset the display
always sits at **the same** optical distance (~1.5–2 m), while the vergence follows the virtual
object — even if it floats 30 cm in front of the nose. **The eyes converge on 30 cm but focus on
2 m.** The brain receives contradictory depth signals → fatigue, headache, blurriness. Hence the
design rule: **no important objects closer than ~50 cm**. Real solutions (light field displays,
varifocal optics) are research.

**The vestibular system & proprioception:** the inner ear measures **acceleration** and
**rotation**, proprioception reports joint positions. Neither can be fooled — a display does not
reach them. That is exactly where section 4.1 comes from.

---

## 2 · Tracking and the mathematics of orientation

### 2.1 Degrees of freedom: 3 DoF vs. 6 DoF

- **3 DoF**: **orientation** only (yaw, pitch, roll). Enough for 360° video. If you lean forward,
  **nothing** happens — the world sticks to your head. That is a direct violation of sensorimotor
  contingency (1.2) and a reliable nausea generator.
- **6 DoF**: orientation **+ position** ($x,y,z$). Only with that can you lean around an object.
  **6 DoF is the lower bound for genuine presence.**

**How is tracking done?**
- **Outside-in**: external base stations/cameras observe the headset (e.g. Lighthouse). Very
  precise, but requires a setup, a limited volume, occlusion possible.
- **Inside-out**: cameras **in** the headset observe the environment and solve **SLAM**
  (*simultaneous localization and mapping*) — build a map and localize yourself in it at the same
  time, a chicken-and-egg problem. No setup, arbitrary volume; in return dependent on light and
  texture (a white wall = no features = loss of tracking). **The standard today.**

**Sensors:** an **IMU** delivers a **gyroscope** (angular velocity, ~1000 Hz) and an
**accelerometer** (acceleration incl. **gravity**). Cameras deliver ~30–60 Hz. That gives a natural
division of labour — and section 2.4 follows from exactly that.

### 2.2 Representing rotations — and why Euler angles fail

An orientation in space has **3 degrees of freedom**. There are several representations:

**Euler angles** $(\text{yaw},\text{pitch},\text{roll})$ — three rotations one after another.
Intuitively readable, compact (3 numbers). Three problems:

1. **The order is a convention, not nature.** "ZYX" ≠ "XYZ". Rotations **do not commute**:
   $R_A R_B \neq R_B R_A$. (Project 01 shows: the same two 90° rotations applied to $\hat z$ yield
   either $[0,-1,0]$ or $[1,0,0]$ depending on the order — different points.) Almost every XR
   interface bug has its root here.
2. **Gimbal lock** — the killer (see below).
3. **Interpolation is broken.** Interpolating linearly between two Euler triples produces
   wobbling, not the shortest rotation.

> ### ⚠️ Gimbal lock — precisely formulated
> If the **pitch is at ±90°** (looking straight up/down), the yaw and the roll axis **coincide**.
> You lose one degree of freedom: **3 DoF → 2 DoF**.
>
> Concretely (convention ZYX, pitch = 90°) the rotation depends **only on the difference
> $(\text{yaw}-\text{roll})$**. All of these combinations produce **exactly the same** rotation:
>
> | yaw | roll | yaw − roll | quaternion |
> |---|---|---|---|
> | 0° | 0° | 0 | $[0,\ 0.7071,\ 0,\ 0.7071]$ |
> | 40° | 40° | 0 | $[0,\ 0.7071,\ 0,\ 0.7071]$ |
> | 90° | 90° | 0 | $[0,\ 0.7071,\ 0,\ 0.7071]$ |
>
> The angular distance between them is **0.000000°** — they are not *similar* but **identical**. A
> user who tilts their head all the way up can no longer control yaw and roll independently; the
> back-conversion `as_euler` has to guess and folds everything into one angle.
> *(A widespread misconception: "(yaw=0, roll=40) and (yaw=40, roll=0) are then equal." **Wrong** —
> their differences differ by 80°, they lie 80° apart. It is the **difference** that survives, not
> the individual value.)*
>
> **Rotation matrices** ($3\times3$, orthogonal, $\det=1$) have no gimbal lock, but need 9 numbers
> for 3 DoF and drift numerically out of orthogonality.

### 2.3 Quaternions — the solution

A **quaternion** $q = w + x\,i + y\,j + z\,k$ with $i^2=j^2=k^2=ijk=-1$. For rotations one uses
**unit quaternions** ($\|q\|=1$). The connection to intuition is the **axis-angle
representation**: a rotation about the unit axis $\hat{\mathbf n}$ by the angle $\theta$:
$$\boxed{\;q = \Big(\cos\tfrac{\theta}{2},\ \hat{\mathbf n}\sin\tfrac{\theta}{2}\Big)\;}$$

A point $\mathbf v$ is rotated by $\mathbf v' = q\,\mathbf v\,q^{-1}$ (with $\mathbf v$ as a pure
quaternion). Composition is simply **multiplication**: $q_{AB} = q_B q_A$.

**Why XR uses them:**
- **No gimbal lock** — the parametrization is regular everywhere.
- **Compact** (4 numbers) and **numerically stable** — drift is corrected by plain
  **normalization**, not by re-orthogonalization.
- **Interpolable** — see SLERP.
- **Cheap composition** (16 multiplications instead of 27 for matrices).

**The curiosity you have to know:** $q$ and $-q$ describe **the same** rotation (the *double
cover* $SU(2)\to SO(3)$; the $\theta/2$ above is the reason). The practical consequence: when
interpolating you have to **check the sign** — otherwise the agent takes the **long way round**
(359° instead of 1°). A classic bug.

**SLERP** (*spherical linear interpolation*) interpolates on the unit sphere along the **great
circle** — the shortest rotation with **constant angular velocity**:
$$\text{Slerp}(q_0,q_1;t)=\frac{\sin\big((1-t)\Omega\big)}{\sin\Omega}\,q_0+\frac{\sin(t\Omega)}{\sin\Omega}\,q_1,
\qquad \cos\Omega = q_0\!\cdot\! q_1$$
Naive **LERP** (averaging component-wise + normalizing) runs along the **chord** instead of the arc
→ the angular velocity **fluctuates**, too fast in the middle. Project 01 measures it: for a
rotation 0°→170° SLERP has exactly constant steps (spread **0.000**), LERP does not (**5.659**).
For small angles the difference is negligible — which is why LERP is quite common for network
interpolation between dense frames.

### 2.4 Sensor fusion: why a gyro alone drifts

The two sensors of an IMU have **complementary** errors:

| | Gyroscope | Accelerometer |
|---|---|---|
| measures | angular **velocity** | acceleration + **gravity** |
| rate | fast (~1000 Hz) | fast |
| short term | **precise, smooth** | **noisy** (every movement disturbs it) |
| long term | **drifts away** | **drift-free** (gravity always points down) |

**Why does the gyro drift?** You need the *angle* but have the *velocity* — so you **integrate**:
$\theta_k=\theta_{k-1}+\omega_k\Delta t$. Every bias, however small, accumulates **without bound**
in the process. A bias of only 0.5 °/s gives an error of **30°** after one minute. (Project 01
measures: RMSE **16.98°**, final error **29.95°** — the virtual world tips away.)

The accelerometer measures an absolute reference via **gravity** ("where is down?"), but is noisy
(RMSE **3.01°**) and unusable during movement.

**The complementary filter** combines both in one line — a high pass on the gyro, a low pass on
the accel:
$$\boxed{\;\theta_k = \alpha\big(\theta_{k-1}+\omega_k\Delta t\big) + (1-\alpha)\,\theta_{\text{accel},k}\;}$$
with $\alpha$ close to 1 (e.g. 0.98). In the short term it follows the smooth gyro, in the long
term the accel pulls it back to the truth. The result: RMSE **0.42°** — **better than either
individual sensor**. A **Kalman filter** does the same thing optimally (with an estimated
uncertainty and bias estimation); the complementary filter is its poor, astonishingly good
relative. Adding a **magnetometer** additionally provides absolute **yaw** (a compass) — which the
accel *cannot* provide, because a rotation about the gravity axis does not change gravity.

---

## 3 · Motion-to-photon: the millisecond budget

### 3.1 The chain

**Motion-to-photon latency** = the time from "the head moves" to "the matching photon hits the
retina". It is **the** critical quantity of XR. The chain:

```
head moves
   → IMU/camera measures       ~1-2 ms
   → sensor fusion/pose        ~1 ms
   → application/physics       ~2-5 ms
   → rendering (GPU)           ~5-11 ms   (at 90 Hz = 11.1 ms per frame)
   → transmission/scanout      ~3-11 ms
   → display (pixel response)  ~1-5 ms
   = motion-to-photon          ~15-40 ms
```

**The target: < 20 ms.** Above that the conflict between the vestibular system and the eyes
becomes noticeable (section 4.1). From ~50 ms on it is unbearable for many. For comparison: a
normal game on a monitor with 60 ms of latency bothers nobody — **because there no vestibular
system contradicts it.**

Note: at **90 Hz** a single frame alone is **11.1 ms**. So the budget is practically used up after
*one* frame plus scanout — there is no room here for "we will optimize that later".

### 3.2 The two countermeasures

Because the chain cannot be shortened arbitrarily, you **cheat** — in two ways:

**1. Prediction.** You do not render for *now* but for the moment at which the photon will appear:
from the current angular velocity you extrapolate the pose forward by the latency $\Delta t$. With
uniform movement that works astonishingly well. The price: at **changes of direction** the
prediction is off → **overshoot**. That is why one predicts only ~20–40 ms ahead, no more.

**2. Timewarp / reprojection (ASW).** The actual trick, and the reason why modern headsets are
bearable: **after** the image has been rendered but **before** it is displayed, you fetch the
**very latest** pose and **shift/warp the finished image** accordingly.
- **Orientational timewarp** is practically free and very effective: a pure *rotation* can be
  corrected on a finished image almost perfectly (you shift the image section).
- **Positional** correction is harder: if the head moves **sideways**, the **occlusion** changes —
  behind the foreground, information would have to appear that was never rendered (disocclusion).
  You get artifacts or have to guess.
- If the GPU is not enough for 90 Hz, you render at 45 Hz and **invent** every second image by
  reprojection (*asynchronous spacewarp*) — visible through artifacts at moving edges, but better
  than stuttering.

> **Note:** timewarp does **not make the latency smaller** — it makes it **invisible for the
> orientation**, the channel to which the vestibular system reacts most sensitively. It is a
> perceptual trick, not a performance fix.

---

## 4 · Interaction and cybersickness

### 4.1 Cybersickness: the sensory conflict theory

**The symptom:** nausea, dizziness, sweating, eye strain, disorientation — for some people after
minutes, still having an effect hours later.

**The leading explanation (sensory conflict / Reason & Brady):** nausea arises when the senses
deliver **contradictory** movement information:
- **The eyes** say: "We are moving" (the world passes by → **vection**, the illusory self-motion).
- **The vestibular system** says: "We are sitting still." It measures real acceleration — and that
  is zero.

The brain cannot resolve the contradiction. The (evolutionarily plausible) hypothesis: such a
conflict naturally arises through **neurotoxins** → the brain infers poisoning → **vomiting**.
**You get sick in VR because the brain believes you have been poisoned.** — Remarkably:
**seasickness is the same conflict with the roles swapped** (the vestibular system reports
movement, the eyes see stillness in the cabin).

**The causes, sorted by effectiveness:**
1. **Latency** — the world lags behind the head movement. The strongest, but also the most easily
   solved lever (section 3).
2. **Artificial locomotion** — movement by stick while the body stands still. The inherent
   conflict.
3. **Acceleration** — constant velocity is relatively harmless; **acceleration**, rotation and
   stairs/ramps are not (the vestibular system measures precisely *acceleration*).
4. **3 DoF instead of 6 DoF**, a wrong interpupillary distance (IPD), a low frame rate.

**Countermeasures — and why they work:**
- **Teleportation** instead of continuous movement: **no** optical flow → **no** conflict. The
  gold standard for comfort, at the price that spatial understanding suffers.
- **Snap turn** (jerky 30° jumps) instead of smooth turning — rotation is the worst case.
- **Vignetting/tunnel vision** during movement: reduces the optical flow in the periphery (which is
  where the sensitivity to vection sits).
- **A static frame of reference** (a virtual cockpit, a nose, a grid): something that moves along
  with the head and confirms "standstill".
- **A high frame rate, low latency** — the basis without which everything else is irrelevant.

> **An honest classification:** the sensory conflict theory explains a lot, but **not everything**
> (for instance not well why susceptibility varies so **massively** between people — women report
> symptoms more often in studies, which is partly attributed to IPD fit). A competing account is
> the **postural instability theory** (Riccio & Stoffregen): nausea follows from a prolonged
> inability to stabilize one's posture. Both are probably partial truths.

### 4.2 Interaction techniques

**Selection & manipulation:**
- **Virtual hand** — grab directly. Natural, but only within arm's reach.
- **Ray-casting** — a ray from the hand, a "laser pointer". Unlimited range; but the **angular
  precision** limits it: at a distance, 1° of hand tremor means many centimetres of offset.
- **Go-Go** — non-linear arm extension: 1:1 up to a threshold, above it the virtual arm grows
  disproportionately. Combines naturalness with reach.
- **Fitts' law** applies here too and quantifies the target acquisition time:
  $$MT = a + b\log_2\!\Big(\frac{D}{W}+1\Big)$$
  ($D$ = distance, $W$ = target width). The term $\log_2(D/W+1)$ is the **index of difficulty**.
  The practical consequence: target size helps **logarithmically** — small targets are expensive,
  and in 3D you use the **angular size**, not the metric one.

**Locomotion** (see 4.1 for the comfort aspect): teleport · continuous (stick) · **room scale**
(real walking — the best comfort, limited by the room) · **redirected walking** (the world is
imperceptibly rotated so that the user walks in a circle but believes they are walking straight —
astonishingly effective, but needs a lot of space).

---

## 5 · Evaluation: XR is an empirical discipline

### 5.1 Why user studies at all?

For "is the latency < 20 ms?" a measurement suffices. But the actual questions — *does this feel
present? does anybody get sick? is it usable?* — can be answered **only on humans**. There is no
offline metric for presence. **That is why XR research is at its core experimental psychology with
technology.**

**Established instruments** (take the validated ones, do not invent your own questionnaires):
- **IPQ** (Igroup Presence Questionnaire) — presence.
- **SSQ** (Simulator Sickness Questionnaire) — cybersickness, with the subscales *nausea*,
  *oculomotor*, *disorientation*. **Important: collect it before *and* after** (the difference is
  what counts).
- **SUS** (System Usability Scale) — 10 items, score 0–100. *(Confusingly: in the presence
  literature "SUS" also means the *Slater-Usoh-Steed* questionnaire. Mind the context.)*
- **NASA-TLX** — subjective workload (mental, physical, temporal, performance, effort,
  frustration).
- **Objectively** alongside: task completion time, error rate, trajectories — and physiologically
  (heart rate, skin conductance) as a sickness correlate.

### 5.2 Study design

- **Within-subject** (every person tests **all** conditions): fewer participants needed, controls
  for individual differences — which in XR are **enormous** (susceptibility, VR experience).
  **In XR usually the right choice.** The price: **order effects** (learning, fatigue, accumulated
  nausea) → **counterbalancing** (e.g. a Latin square) is mandatory.
- **Between-subject** (every person tests **one** condition): no order effects, but far more
  participants needed. Necessary if one condition "spoils" the other (whoever has had 6 DoF once
  rates 3 DoF differently).

**Analysis** — this ties in directly with module 03:
- **Mind the scale level:** questionnaire items are **ordinal** (Likert). For individual items
  non-parametric tests are appropriate: **Wilcoxon signed-rank** (within), **Mann-Whitney U**
  (between). For averaged subscales with many items one often argues for an interval scale →
  a **paired t-test** / ANOVA. *Both are defensible — you just have to justify it.*
- **Report the effect size, not only p.** A significant but tiny effect is irrelevant.
  **Cohen's d** resp. $r=Z/\sqrt{N}$. Rule of thumb: $d\approx0.2$ small, $0.5$ medium, $0.8$
  large.
- **Correct for multiple comparisons.** Whoever tests IPQ, SSQ, SUS, TLX and time individually
  performs 5+ tests — at $\alpha=0.05$ a false alarm is then almost guaranteed (**Bonferroni**:
  $\alpha/m$; or Holm/FDR). *That is the same idea as the base rate discussion in module 15: many
  tests × a small error rate = many false alarms.*
- **Plan the sample size in advance** (a power analysis). N=8 only finds elephants. Typical XR
  studies: N=20–40.
- **Ethics:** cybersickness is a real burden. The possibility to stop without giving a reason,
  informed consent, breaks, no driving home directly after a sickness study.

---

## 6 · Summary / cheat sheet

**Concepts.** The reality-virtuality continuum (Milgram) · **immersion** = the technology
(objective) · **presence** = the experience (subjective) · **place illusion** (tracking+latency!)
+ **plausibility illusion** (Slater).

**The human.** FoV ~200° vs. a headset's ~110° · **VAC**: vergence follows the object,
accommodation sticks at ~2 m → nothing closer than ~50 cm · the vestibular system **cannot be
fooled**.

**Tracking.** 3 DoF (orientation only) vs. **6 DoF** (+ position) · outside-in vs. **inside-out
(SLAM)** · IMU = gyro (fast, **drifts**) + accel (noisy, **drift-free**).

**Rotations.** Euler: intuitive, but **non-commutative** + **gimbal lock** (pitch ±90° ⟹ only
(yaw−roll) survives, 3→2 DoF) · matrices: 9 numbers, they drift · **quaternions**:
$q=(\cos\frac\theta2,\ \hat{\mathbf n}\sin\frac\theta2)$, $\mathbf v'=q\mathbf vq^{-1}$,
$q\equiv-q$ (check the sign!) · **SLERP** = constant angular velocity, LERP is not.

**Fusion.** $\theta_k=\alpha(\theta_{k-1}+\omega_k\Delta t)+(1-\alpha)\theta_{\text{accel},k}$
⟹ better than either sensor alone.

**Latency.** **Motion-to-photon < 20 ms** · 90 Hz ⟹ 11.1 ms per frame · **prediction**
(extrapolate, overshoot at changes of direction) · **timewarp** (correct the finished image;
rotation ~free, position ⟹ **disocclusion**).

**Cybersickness.** **Sensory conflict**: the eyes see movement (**vection**), the vestibular
system does not ⟹ the brain suspects poison. Levers: latency > artificial locomotion >
**acceleration**. Countermeasures: teleport, snap turn, vignetting, a static frame of reference.

**Interaction.** Virtual hand · **ray-casting** (angular precision!) · Go-Go · **Fitts**:
$MT=a+b\log_2(D/W+1)$.

**Evaluation.** **IPQ** (presence) · **SSQ** (sickness, before/after!) · SUS · NASA-TLX ·
**within-subject + counterbalancing** · ordinal ⟹ **Wilcoxon/Mann-Whitney** · **effect size**
(Cohen's d) · **Bonferroni** · plan the power in advance.

---

## 7 · Self-test

<details>
<summary><b>1.</b> Immersion vs. presence — and why is an ugly, fast system better than a beautiful, slow one?</summary>

**Immersion** = an objective property of the technology (FoV, latency, tracking). **Presence** =
the subjective feeling of being there. The **place illusion** arises from **sensorimotor
contingency** — the world has to react to head movement *as it would in reality*. That depends on
**tracking and latency**, not on the graphics. A low-latency comic-book room therefore produces
more presence than a photorealistic scene with a 50 ms delay (which additionally makes you sick).
</details>

<details>
<summary><b>2.</b> What is the vergence-accommodation conflict and what follows from it for design?</summary>

**Vergence** (the eyes rotate inwards) follows the virtual object, **accommodation** (the lens
focuses) sticks at the fixed display distance (~2 m). In reality both are coupled — in the headset
they contradict each other ⟹ fatigue, headache. **Design rule: nothing important closer than
~50 cm.** Real solutions (varifocal/light field optics) are research.
</details>

<details>
<summary><b>3.</b> Explain gimbal lock precisely. Which combinations become indistinguishable?</summary>

At **pitch = ±90°** the yaw and roll axes coincide ⟹ **3 DoF → 2 DoF**. Only the **difference**
$(\text{yaw}-\text{roll})$ survives: (0°,0°), (40°,40°), (90°,90°) produce **exactly the same**
rotation (a distance of 0.000000°). **Not** equal, by contrast, are (0°,40°) and (40°,0°) — their
differences differ by 80°. Quaternions do not have this problem.
</details>

<details>
<summary><b>4.</b> Why $\theta/2$ in the quaternion — and why is $q\equiv-q$ practically relevant?</summary>

Because the rotation acts **twice** as $\mathbf v'=q\mathbf vq^{-1}$ (once $q$, once $q^{-1}$) —
each half contributes $\theta/2$. The consequence is the **double cover**: $q$ and $-q$ are the
same rotation. **Practically:** when interpolating you have to check the sign (if necessary
$q_1 \to -q_1$), otherwise SLERP takes the **long way round** (359° instead of 1°).
</details>

<details>
<summary><b>5.</b> SLERP vs. LERP — what is the difference, and when is LERP fine anyway?</summary>

**SLERP** runs the **great circle** on the unit sphere ⟹ the shortest rotation with **constant
angular velocity**. **LERP** (average + normalize) runs the **chord** ⟹ the angular velocity
fluctuates (measured: a spread of the steps of 0.000 vs. 5.659 for 0°→170°). For **small** angles
the error is negligible — which is why LERP between dense frames is common (it is cheaper).
</details>

<details>
<summary><b>6.</b> Why does a gyroscope drift, and how does the complementary filter repair it?</summary>

The gyro measures angular **velocity**; for the angle you have to **integrate** — and in doing so
every **bias** accumulates without bound (0.5 °/s ⟹ 30° after one minute; measured RMSE 16.98°).
The **accelerometer** delivers an absolute, drift-free but noisy reference via **gravity** (RMSE
3.01°). The filter $\theta_k=\alpha(\theta_{k-1}+\omega_k\Delta t)+(1-\alpha)\theta_{\text{accel}}$
takes the gyro in the short term and the accel in the long term ⟹ RMSE **0.42°**, better than
both.
</details>

<details>
<summary><b>7.</b> What is motion-to-photon latency, where is the limit, and why does 60 ms on a monitor not bother anyone?</summary>

The time from the head movement to the matching photon on the retina; a chain of sensor → fusion →
app → rendering → scanout → display. **The target is < 20 ms.** On a monitor 60 ms does not bother
anyone because **no vestibular system contradicts it** there — in the headset the delay produces
exactly the sensory conflict that makes you sick.
</details>

<details>
<summary><b>8.</b> What does timewarp do — and why does it help more with rotation than with translation?</summary>

It corrects the **finished rendered** image shortly before display using the **latest** pose. A
pure **rotation** can be shifted afterwards almost perfectly (just a different image section). With
**translation** the **occlusion** changes: behind foreground objects, information would have to
appear that was never rendered (**disocclusion**) ⟹ artifacts/guessing. Timewarp does **not**
lower the latency, it makes it **invisible** for the orientation.
</details>

<details>
<summary><b>9.</b> Explain cybersickness via the sensory conflict theory. Why does teleportation help?</summary>

The eyes report self-motion (**vection**), the vestibular system reports standstill. The
contradiction resembles the pattern of a **poisoning** ⟹ nausea. **Teleportation** produces **no
optical flow** ⟹ no conflict. Further levers: lower the latency, avoid **acceleration**, snap turn,
vignetting, a static frame of reference. (Seasickness is the same conflict with the roles swapped.)
</details>

<details>
<summary><b>10.</b> You compare 3 DoF vs. 6 DoF and measure IPQ, SSQ, SUS, TLX and time. Name three methodological obligations.</summary>

Any three: **within-subject with counterbalancing** (order effects: learning, fatigue, accumulated
nausea) · collect the **SSQ before and after** (the difference counts) · **ordinal** Likert items
⟹ **Wilcoxon** instead of a t-test (or justify it) · report the **effect size**, not only p ·
**correct for multiple comparisons** (5 tests ⟹ Bonferroni/Holm) · **plan the power in advance**
(N=20–40) · **ethics** (stopping at any time).
</details>

---

## 8 · Literature & sources

**Standard works:**
- 📗 **LaValle — *Virtual Reality*** (Cambridge; **free online**: lavalle.pl/vr/). *The* reference
  for exactly this module: perception, tracking, rotation mathematics, latency — with mathematics,
  but readable. **The best single source.** Chapters 3 (transformations), 9 (tracking),
  12 (sickness).
- 📗 **Jerald — *The VR Book: Human-Centered Design for VR***. Strong on perception, sickness and
  interaction design. *Beginner-friendly.*
- 📗 **Bowman et al. — *3D User Interfaces: Theory and Practice***. The standard work on
  interaction techniques (→ deepened in module 19). *In depth.*

**Key papers:**
- 📄 **Milgram & Kishino (1994), *A Taxonomy of Mixed Reality Visual Displays*** — the continuum.
- 📄 **Slater (2009), *Place Illusion and Plausibility Illusion*** (Phil. Trans. R. Soc.) — the
  conceptual basis of 1.2. *Free, short, worth reading.*
- 📄 **Reason & Brady (1975), *Motion Sickness*** — the sensory conflict theory.
- 📄 **Riccio & Stoffregen (1991), *An Ecological Theory of Motion Sickness*** — the competing
  postural instability theory.
- 📄 **Shoemake (1985), *Animating Rotation with Quaternion Curves*** (SIGGRAPH) — **SLERP**.
- 📄 **Van Waveren (2016), *The Asynchronous Time Warp for VR on Mobile Hardware***.
- 📄 **Poupyrev et al. (1996), *The Go-Go Interaction Technique***.
- 📄 **Razzaque et al. (2001), *Redirected Walking***.

**Mathematics/practice:**
- 🌐 **3Blue1Brown — *Visualizing quaternions*** (eater.net/quaternions) — interactive, excellent
  if quaternions remain abstract. *Beginner-friendly, free.*
- 🌐 **The scipy `spatial.transform.Rotation` docs** — the API of the projects.
- 🌐 **The OpenXR specification** (khronos.org/openxr) — the open industry standard; the terms
  (pose, space, predicted display time) appear there in exactly this form.
- 🌐 **Unity XR Interaction Toolkit** / **the Oculus developer blog** (Carmack/Abrash on latency) —
  for practical work.

**Evaluation:**
- 📄 **Kennedy et al. (1993), *Simulator Sickness Questionnaire (SSQ)***.
- 📄 **Schubert et al. (2001), *The Experience of Presence: Factor Analytic Insights*** (IPQ;
  igroup.org/pq/ipq — freely available, including the items).
- 📄 **Hart & Staveland (1988), *NASA-TLX***; **Brooke (1996), *SUS***.
- 📗 **For the craft:** *Field — Discovering Statistics* for choosing the test, or module 03 of
  this repo.

---

## Next module

**Module 18 — Multimodal Interfaces** extends interaction beyond the hands (speech, gaze,
gestures, haptics) and asks how modalities are **fused**. **Module 19 — 3D User Interfaces** then
systematically deepens the interaction techniques from 4.2. What you have learned here —
**presence depends on latency and tracking**, quaternions, and that in the end a **human** in an
experiment decides — carries through the entire XR block.

---

# Modul 17 — Core XR: Principles of Interactive Systems (deutsche Fassung)

> **Worum geht es?** **XR** (Extended Reality — VR, AR, MR) ist der Versuch, einem Menschen
> vorzugaukeln, er sei woanders. Das Erstaunliche daran: Ob das gelingt, entscheidet sich kaum
> an der Grafik. Es entscheidet sich an **Millisekunden** und an **Mathematik** — an der Frage,
> ob das Bild schnell und korrekt genug auf eine Kopfbewegung reagiert. Ist es 20 ms zu spät,
> wird dem Nutzer übel. Dieses Modul behandelt die **Prinzipien** dahinter: Wahrnehmung,
> **Tracking** und Rotationsmathematik, **Motion-to-Photon-Latenz**, Interaktionstechniken,
> **Cybersickness** — und wie man interaktive Systeme **empirisch evaluiert**, denn die einzige
> Instanz, die über XR urteilt, ist ein Mensch.

**Hilfreiche Vorkenntnisse:** Lineare Algebra (Matrizen, Vektoren, Basiswechsel), Trigonometrie,
etwas Statistik.

**Diese Module solltest du vorher gemacht haben:**
- **Modul 02/03 (Data Science)** — für den Evaluationsteil: EDA, Hypothesentests, Bootstrap.
  Abschnitt 5.2 knüpft direkt an die A/B-Test-Logik aus Modul 03 an.
- Sonst **keine**. Dieses Modul startet ein neues Feld (Block F) und baut nicht auf ML/RL auf.

> **Hinweis zur Ausgestaltung.** Wie bei Modul 15/16 lag keine offizielle Modulbeschreibung vor.
> Ich schneide „Core XR: Principles of Interactive Systems" auf die **Prinzipien** zu, die
> hardware-unabhängig gelten und die man ohne VR-Brille wirklich *durchdringen* kann:
> **Wahrnehmung, Tracking-Mathematik, Latenz, Interaktion, Cybersickness, Evaluation**.
>
> **Werkzeug-Entscheidung:** In dieser Umgebung gibt es **keine VR-Hardware** und **keine
> 3D-Engine** (`open3d`, `trimesh`, `pygame` fehlen). Das ist weniger schlimm, als es klingt:
> Der intellektuelle Kern von XR ist **Mathematik und Zeitverhalten**, und beides lässt sich mit
> `numpy`/`scipy` exakt nachrechnen und **testen** — besser sogar als in einer Engine, wo alles
> hinter fertigen Funktionen verschwindet. `scipy.spatial.transform.Rotation` (inkl. `Slerp`)
> ist vorhanden; wo eine Engine nötig wäre (Rendering, Shader), erkläre ich **theoretisch**.
> Für die Praxis: **Unity + OpenXR** ist der Industriestandard.

---

## Lernziele

Nach diesem Modul kannst du …

- XR im **Reality-Virtuality-Kontinuum** (Milgram) verorten und **Immersion** (technisch) von
  **Präsenz** (psychologisch) unterscheiden — inkl. Slaters *place illusion* / *plausibility illusion*;
- die relevanten Eigenschaften des **menschlichen Wahrnehmungssystems** benennen (FoV,
  Auflösung, **Vergenz-Akkommodations-Konflikt**, Vestibularsystem, Propriozeption) und daraus
  Design-Anforderungen ableiten;
- **Tracking** einordnen: 3 vs. **6 DoF**, outside-in vs. inside-out, SLAM, IMU-Sensorik — und
  erklären, warum ein Gyroskop **driftet** und wie **Sensorfusion** das repariert;
- **Rotationen** sicher handhaben: Euler-Winkel und ihr **Gimbal Lock**, Rotationsmatrizen,
  **Quaternionen** (mit Herleitung), **SLERP** — und begründen, warum XR Quaternionen nutzt;
- die **Motion-to-Photon-Latenz** in ihre Glieder zerlegen, ihr Budget berechnen und die
  Gegenmittel erklären (**Prediction**, **Timewarp/Reprojection**);
- **Interaktionstechniken** (Ray-Casting, Go-Go, Locomotion) vergleichen und **Fitts' Law**
  anwenden;
- **Cybersickness** über die **Sensory-Conflict-Theorie** erklären und Gegenmaßnahmen begründen;
- eine **Nutzerstudie** methodisch sauber planen und auswerten: Within-/Between-Subject,
  etablierte Fragebögen (**IPQ, SSQ, SUS, NASA-TLX**), passende Tests, **Effektstärke**,
  Mehrfachvergleiche.

---

## 1 · Grundlagen — Was XR ausmacht

### 1.1 Das Reality-Virtuality-Kontinuum

**Milgram & Kishino (1994)** ordnen alles auf einer Achse:

```
Reale          Augmented        Augmented         Virtuelle
Umgebung  ───  Reality (AR) ─── Virtuality  ───   Umgebung (VR)
   |                |                |                 |
 nichts        Virtuelles       Reales in         alles
 virtuell      in real          virtuell          virtuell
        └──────── Mixed Reality (MR) ────────┘
```

**XR** ist der Sammelbegriff für alles davon. Der Unterschied ist nicht bloß graduell — er
ändert die **Anforderungen**: In **VR** muss man die ganze Welt liefern, aber niemand sieht, ob
sie *falsch* zur Realität steht. In **AR** ist die Realität die Referenz — Virtuelles muss
**registriert** sein (an der richtigen Stelle bleiben), und schon 1 mm Versatz oder 5 ms
Verzögerung fallen sofort auf, weil das echte Objekt daneben liegt. **AR ist deshalb technisch
härter als VR.**

### 1.2 Immersion ≠ Präsenz

Die wichtigste begriffliche Unterscheidung des Modulnamens (**Slater**):

- **Immersion** ist eine **objektive Eigenschaft der Technik**: FoV, Auflösung, Latenz,
  Tracking-Genauigkeit, Anzahl angesprochener Sinne. Messbar, in Datenblättern nachlesbar.
- **Präsenz** ist die **subjektive Reaktion des Menschen**: *das Gefühl, dort zu sein*. Nicht
  messbar außer durch den Nutzer selbst (→ Abschnitt 5).

Slater zerlegt Präsenz weiter in zwei Illusionen, die **unabhängig** voneinander kippen können:
- **Place Illusion (PI)** — „Ich bin an diesem Ort." Entsteht aus **sensomotorischer
  Kontingenz**: Ich bewege den Kopf und die Welt reagiert **so, wie sie es in echt täte**. PI
  hängt fast vollständig an **Tracking und Latenz** — nicht an der Grafik.
- **Plausibility Illusion (Psi)** — „Was hier passiert, passiert wirklich." Entsteht, wenn die
  Welt **auf mich reagiert** und sich glaubwürdig verhält.

> **Die zentrale Einsicht des Moduls:** Ein grafisch primitiver, aber perfekt getrackter,
> latenzarmer Comic-Raum erzeugt **mehr Präsenz** als eine fotorealistische Szene, die 50 ms
> hinterherhinkt. **Deshalb geht es in diesem Modul um Mathematik und Millisekunden, nicht um
> Shader.** PI bricht sofort zusammen, wenn die sensomotorische Kontingenz verletzt wird — und
> der Körper bemerkt das gnadenlos zuverlässig.

### 1.3 Der Mensch als Systemspezifikation

XR baut nicht für Displays, sondern für ein **Wahrnehmungssystem**. Dessen Eckdaten *sind* die
Anforderungen:

| Größe | Mensch | typisches Headset | Konsequenz |
|---|---|---|---|
| **Sichtfeld (FoV)** | ~200–220° horizontal (binokular ~114° überlappend) | ~90–110° | „Taucherbrillen-Effekt", schwächt Präsenz |
| **Auflösung** | ~60 Pixel/Grad (Fovea) | ~15–35 Pixel/Grad | Screen-Door-Effekt, unscharfe Ferne |
| **Zeitliche Auflösung** | Flimmern bis ~60–90 Hz sichtbar, Latenz ab ~20 ms spürbar | 90–120 Hz | **das harte Kriterium** (Abschnitt 3) |
| **Stereo-Tiefe** | Vergenz + Akkommodation **gekoppelt** | Akkommodation fix auf ~2 m | **Vergenz-Akkommodations-Konflikt** |

**Der Vergenz-Akkommodations-Konflikt (VAC)** verdient eine eigene Erklärung, weil er ein
**physikalisch unlösbares** Problem herkömmlicher Headsets ist:
- **Vergenz** = beide Augen drehen nach innen, um ein nahes Objekt zu fixieren.
- **Akkommodation** = die Augenlinse stellt scharf.

In der Realität sind beide **fest gekoppelt** (was ich fixiere, stelle ich scharf). Im Headset
sitzt das Display **immer** in derselben optischen Entfernung (~1,5–2 m), während die Vergenz
dem virtuellen Objekt folgt — auch wenn es 30 cm vor der Nase schwebt. **Die Augen konvergieren
auf 30 cm, fokussieren aber auf 2 m.** Das Gehirn bekommt widersprüchliche Tiefensignale →
Ermüdung, Kopfschmerz, Unschärfe. Deshalb die Design-Regel: **keine wichtigen Objekte näher als
~50 cm**. Echte Lösungen (Lichtfeld-Displays, varifokale Optik) sind Forschung.

**Vestibularsystem & Propriozeption:** Das Innenohr misst **Beschleunigung** und **Drehung**,
die Propriozeption meldet Gelenkstellungen. Beide kann man **nicht** täuschen — ein Display
erreicht sie nicht. Genau daraus entsteht Abschnitt 4.1.

---

## 2 · Tracking und die Mathematik der Orientierung

### 2.1 Freiheitsgrade: 3 DoF vs. 6 DoF

- **3 DoF**: nur **Orientierung** (yaw, pitch, roll). Reicht für 360°-Video. Beugt man sich vor,
  passiert **nichts** — die Welt klebt am Kopf. Das ist ein direkter Bruch der sensomotorischen
  Kontingenz (1.2) und ein zuverlässiger Übelkeits-Generator.
- **6 DoF**: Orientierung **+ Position** ($x,y,z$). Erst damit kann man sich um ein Objekt herum
  beugen. **6 DoF ist die Untergrenze für echte Präsenz.**

**Wie wird getrackt?**
- **Outside-in**: externe Basisstationen/Kameras beobachten das Headset (z. B. Lighthouse). Sehr
  präzise, aber Aufbau nötig, begrenztes Volumen, Verdeckung möglich.
- **Inside-out**: Kameras **im** Headset beobachten die Umgebung und lösen **SLAM**
  (*Simultaneous Localization and Mapping*) — Karte bauen und sich gleichzeitig darin
  lokalisieren, ein Henne-Ei-Problem. Kein Aufbau, beliebiges Volumen; dafür abhängig von Licht
  und Textur (weiße Wand = keine Merkmale = Tracking-Verlust). **Heute Standard.**

**Sensorik:** Eine **IMU** liefert **Gyroskop** (Winkelgeschwindigkeit, ~1000 Hz) und
**Accelerometer** (Beschleunigung inkl. **Schwerkraft**). Kameras liefern ~30–60 Hz. Das ergibt
eine natürliche Arbeitsteilung — und genau daraus folgt Abschnitt 2.4.

### 2.2 Rotationen darstellen — und warum Euler-Winkel scheitern

Eine Orientierung im Raum hat **3 Freiheitsgrade**. Es gibt mehrere Darstellungen:

**Euler-Winkel** $(\text{yaw},\text{pitch},\text{roll})$ — drei Drehungen nacheinander. Intuitiv
lesbar, kompakt (3 Zahlen). Drei Probleme:

1. **Reihenfolge ist Konvention, nicht Natur.** „ZYX" ≠ „XYZ". Rotationen **kommutieren nicht**:
   $R_A R_B \neq R_B R_A$. (Projekt 01 zeigt: dieselben zwei 90°-Drehungen auf $\hat z$ ergeben
   je nach Reihenfolge $[0,-1,0]$ oder $[1,0,0]$ — verschiedene Punkte.) Fast jeder
   XR-Schnittstellen-Bug hat hier seine Wurzel.
2. **Gimbal Lock** — der Killer (siehe unten).
3. **Interpolation ist kaputt.** Zwischen zwei Euler-Tripeln linear zu interpolieren erzeugt
   Taumeln, keine kürzeste Drehung.

> ### ⚠️ Gimbal Lock — präzise formuliert
> Steht der **Pitch auf ±90°** (Blick senkrecht nach oben/unten), fallen die yaw- und die
> roll-Achse **zusammen**. Man verliert einen Freiheitsgrad: **3 DoF → 2 DoF**.
>
> Konkret (Konvention ZYX, Pitch = 90°) hängt die Rotation **nur noch von der Differenz
> $(\text{yaw}-\text{roll})$** ab. Alle diese Kombinationen ergeben **exakt dieselbe** Rotation:
>
> | yaw | roll | yaw − roll | Quaternion |
> |---|---|---|---|
> | 0° | 0° | 0 | $[0,\ 0{,}7071,\ 0,\ 0{,}7071]$ |
> | 40° | 40° | 0 | $[0,\ 0{,}7071,\ 0,\ 0{,}7071]$ |
> | 90° | 90° | 0 | $[0,\ 0{,}7071,\ 0,\ 0{,}7071]$ |
>
> Der Winkelabstand zwischen ihnen ist **0,000000°** — sie sind nicht *ähnlich*, sondern
> **identisch**. Ein Nutzer, der den Kopf ganz nach oben legt, kann yaw und roll nicht mehr
> unabhängig steuern; die Rückrechnung `as_euler` muss raten und faltet alles in einen Winkel.
> *(Ein verbreiteter Irrtum: „(yaw=0, roll=40) und (yaw=40, roll=0) sind dann gleich." **Falsch** —
> ihre Differenz unterscheidet sich um 80°, sie liegen 80° auseinander. Es ist die **Differenz**,
> die überlebt, nicht der Einzelwert.)*
>
> **Rotationsmatrizen** ($3\times3$, orthogonal, $\det=1$) haben kein Gimbal Lock, brauchen aber
> 9 Zahlen für 3 DoF und driften numerisch aus der Orthogonalität.

### 2.3 Quaternionen — die Lösung

Ein **Quaternion** $q = w + x\,i + y\,j + z\,k$ mit $i^2=j^2=k^2=ijk=-1$. Für Rotationen nutzt
man **Einheitsquaternionen** ($\|q\|=1$). Der Zusammenhang zur Anschauung ist die
**Achse-Winkel-Darstellung**: Drehung um die Einheitsachse $\hat{\mathbf n}$ um den Winkel $\theta$:
$$\boxed{\;q = \Big(\cos\tfrac{\theta}{2},\ \hat{\mathbf n}\sin\tfrac{\theta}{2}\Big)\;}$$

Ein Punkt $\mathbf v$ wird rotiert durch $\mathbf v' = q\,\mathbf v\,q^{-1}$ (mit $\mathbf v$ als
reinem Quaternion). Verkettung ist schlicht **Multiplikation**: $q_{AB} = q_B q_A$.

**Warum XR sie nutzt:**
- **Kein Gimbal Lock** — die Parametrisierung ist überall regulär.
- **Kompakt** (4 Zahlen) und **numerisch stabil** — Drift korrigiert man durch schlichtes
  **Normieren**, nicht durch Re-Orthogonalisierung.
- **Interpolierbar** — siehe SLERP.
- **Verkettung billig** (16 Multiplikationen statt 27 bei Matrizen).

**Die Kuriosität, die man kennen muss:** $q$ und $-q$ beschreiben **dieselbe** Rotation (die
*doppelte Überdeckung* $SU(2)\to SO(3)$; das $\theta/2$ oben ist der Grund). Praktische Folge:
Beim Interpolieren muss man das **Vorzeichen prüfen** — sonst nimmt der Agent den **langen Weg**
(359° statt 1°). Ein klassischer Bug.

**SLERP** (*Spherical Linear Interpolation*) interpoliert auf der Einheitssphäre entlang des
**Großkreises** — die kürzeste Drehung mit **konstanter Winkelgeschwindigkeit**:
$$\text{Slerp}(q_0,q_1;t)=\frac{\sin\big((1-t)\Omega\big)}{\sin\Omega}\,q_0+\frac{\sin(t\Omega)}{\sin\Omega}\,q_1,
\qquad \cos\Omega = q_0\!\cdot\! q_1$$
Naives **LERP** (komponentenweise mitteln + normieren) läuft die **Sehne** statt des Bogens →
die Winkelgeschwindigkeit **schwankt**, in der Mitte zu schnell. Projekt 01 misst das: bei einer
Drehung 0°→170° hat SLERP exakt konstante Schritte (Streuung **0,000**), LERP nicht (**5,659**).
Bei kleinen Winkeln ist der Unterschied vernachlässigbar — deshalb ist LERP für
Netzwerk-Interpolation zwischen dichten Frames durchaus üblich.

### 2.4 Sensorfusion: warum Gyro allein driftet

Die zwei Sensoren einer IMU haben **komplementäre** Fehler:

| | Gyroskop | Accelerometer |
|---|---|---|
| misst | Winkel**geschwindigkeit** | Beschleunigung + **Schwerkraft** |
| Rate | schnell (~1000 Hz) | schnell |
| kurzfristig | **präzise, glatt** | **verrauscht** (jede Bewegung stört) |
| langfristig | **driftet weg** | **driftfrei** (Schwerkraft zeigt immer nach unten) |

**Warum driftet das Gyro?** Man braucht den *Winkel*, hat aber die *Geschwindigkeit* — also
**integriert** man: $\theta_k=\theta_{k-1}+\omega_k\Delta t$. Jeder noch so kleine **Bias**
summiert sich dabei **unbegrenzt** auf. Ein Bias von nur 0,5 °/s ergibt nach einer Minute **30°**
Fehler. (Projekt 01 misst: RMSE **16,98°**, Endfehler **29,95°** — die virtuelle Welt kippt weg.)

Das Accelerometer misst über die **Schwerkraft** einen absoluten Bezug („wo ist unten?"), ist
aber verrauscht (RMSE **3,01°**) und bei Bewegung unbrauchbar.

**Der Komplementärfilter** kombiniert beide in einer Zeile — Hochpass aufs Gyro, Tiefpass aufs
Accel:
$$\boxed{\;\theta_k = \alpha\big(\theta_{k-1}+\omega_k\Delta t\big) + (1-\alpha)\,\theta_{\text{accel},k}\;}$$
mit $\alpha$ nahe 1 (z. B. 0,98). Kurzfristig folgt er dem glatten Gyro, langfristig zieht ihn
das Accel zurück auf die Wahrheit. Ergebnis: RMSE **0,42°** — **besser als beide Einzelsensoren**.
Ein **Kalman-Filter** macht dasselbe optimal (mit geschätzter Unsicherheit und Bias-Schätzung);
der Komplementärfilter ist sein armer, erstaunlich guter Verwandter. Die **Magnetometer**-Ergänzung
liefert zusätzlich absoluten **yaw** (Kompass) — den das Accel *nicht* liefern kann, weil eine
Drehung um die Schwerkraftachse die Schwerkraft nicht ändert.

---

## 3 · Motion-to-Photon: das Millisekunden-Budget

### 3.1 Die Kette

**Motion-to-Photon-Latenz** = die Zeit von „Kopf bewegt sich" bis „passendes Photon trifft die
Netzhaut". Sie ist **die** kritische Größe von XR. Die Kette:

```
Kopf bewegt sich
   → IMU/Kamera misst        ~1-2 ms
   → Sensorfusion/Pose       ~1 ms
   → Anwendung/Physik        ~2-5 ms
   → Rendering (GPU)         ~5-11 ms   (bei 90 Hz = 11.1 ms pro Frame)
   → Übertragung/Scanout     ~3-11 ms
   → Display (Pixel-Response) ~1-5 ms
   = Motion-to-Photon        ~15-40 ms
```

**Das Ziel: < 20 ms.** Darüber wird der Konflikt zwischen Vestibularsystem und Augen spürbar
(Abschnitt 4.1). Ab ~50 ms ist es für viele unerträglich. Zum Vergleich: Ein normales Spiel am
Monitor mit 60 ms Latenz stört niemanden — **weil dort kein Vestibularsystem widerspricht.**

Man beachte: Bei **90 Hz** ist allein ein Frame **11,1 ms**. Das Budget ist also nach *einem*
Frame plus Scanout praktisch aufgebraucht — hier ist kein Platz für „das optimieren wir später".

### 3.2 Die zwei Gegenmittel

Weil man die Kette nicht beliebig verkürzen kann, **betrügt** man — auf zwei Arten:

**1. Prediction.** Man rendert nicht für *jetzt*, sondern für den Zeitpunkt, zu dem das Photon
erscheinen wird: Aus der aktuellen Winkelgeschwindigkeit extrapoliert man die Pose um die
Latenz $\Delta t$ nach vorn. Bei gleichmäßiger Bewegung funktioniert das verblüffend gut. Der
Preis: Bei **Richtungswechseln** liegt die Vorhersage daneben → **Overshoot**. Deshalb sagt man
nur ~20–40 ms voraus, nicht mehr.

**2. Timewarp / Reprojection (ASW).** Der eigentliche Trick, und der Grund, warum moderne
Headsets erträglich sind: **Nachdem** das Bild gerendert ist, aber **bevor** es angezeigt wird,
holt man die **allerneueste** Pose und **verschiebt/verzerrt das fertige Bild** entsprechend.
- **Orientational Timewarp** ist quasi gratis und sehr wirksam: Eine reine *Drehung* lässt sich
  auf einem fertigen Bild fast perfekt nachkorrigieren (man schiebt den Bildausschnitt).
- **Positionale** Korrektur ist schwerer: Bewegt sich der Kopf **seitlich**, ändert sich die
  **Verdeckung** — hinter dem Vordergrund müsste Information auftauchen, die nie gerendert wurde
  (Disokklusion). Man bekommt Artefakte oder muss raten.
- Reicht die GPU nicht für 90 Hz, rendert man mit 45 Hz und **erfindet** jedes zweite Bild per
  Reprojection dazu (*Asynchronous Spacewarp*) — sichtbar an Artefakten an bewegten Kanten,
  aber besser als Ruckeln.

> **Merke:** Timewarp macht die **Latenz nicht kleiner** — es macht sie **unsichtbar für die
> Orientierung**, den Kanal, auf den das Vestibularsystem am empfindlichsten reagiert. Es ist ein
> Wahrnehmungs-Trick, kein Performance-Fix.

---

## 4 · Interaktion und Cybersickness

### 4.1 Cybersickness: die Sensory-Conflict-Theorie

**Das Symptom:** Übelkeit, Schwindel, Schwitzen, Augenermüdung, Desorientierung — bei manchen
Menschen nach Minuten, noch Stunden nachwirkend.

**Die führende Erklärung (Sensory Conflict / Reason & Brady):** Übelkeit entsteht, wenn die
Sinne **widersprüchliche** Bewegungsinformation liefern:
- **Augen** sagen: „Wir bewegen uns" (die Welt zieht vorbei → **Vection**, die illusorische
  Eigenbewegung).
- **Vestibularsystem** sagt: „Wir sitzen still." Es misst echte Beschleunigung — und die ist null.

Das Gehirn kann den Widerspruch nicht auflösen. Die (evolutionär plausible) Hypothese: Ein
solcher Konflikt entsteht natürlicherweise durch **Neurotoxine** → das Gehirn schließt auf
Vergiftung → **Erbrechen**. **Man wird in VR schlecht, weil das Gehirn glaubt, man sei
vergiftet.** — Bemerkenswert: **Seekrankheit ist derselbe Konflikt mit vertauschten Rollen**
(Vestibularsystem meldet Bewegung, die Augen sehen in der Kabine Stillstand).

**Die Ursachen, nach Wirksamkeit sortiert:**
1. **Latenz** — die Welt hinkt der Kopfbewegung hinterher. Der stärkste, aber auch der am
   besten lösbare Hebel (Abschnitt 3).
2. **Künstliche Fortbewegung** — Bewegung per Stick, während der Körper stillsteht. Der
   inhärente Konflikt.
3. **Beschleunigung** — konstante Geschwindigkeit ist relativ harmlos; **Beschleunigung**,
   Drehung und Treppen/Rampen sind es nicht (das Vestibularsystem misst genau *Beschleunigung*).
4. **3 DoF statt 6 DoF**, falscher Augenabstand (IPD), niedrige Bildrate.

**Gegenmaßnahmen — und warum sie wirken:**
- **Teleportation** statt kontinuierlicher Bewegung: **kein** optischer Fluss → **kein** Konflikt.
  Der Goldstandard für Komfort, mit dem Preis, dass räumliches Verständnis leidet.
- **Snap-Turn** (ruckartige 30°-Sprünge) statt weichem Drehen — Drehung ist der schlimmste Fall.
- **Vignetting/Tunnelblick** während der Bewegung: reduziert den optischen Fluss in der
  Peripherie (dort sitzt die Vection-Empfindlichkeit).
- **Statischer Bezugsrahmen** (virtuelles Cockpit, Nase, Gitter): etwas, das mit dem Kopf
  mitgeht und „Stillstand" bestätigt.
- **Hohe Bildrate, niedrige Latenz** — die Basis, ohne die alles andere egal ist.

> **Ehrliche Einordnung:** Die Sensory-Conflict-Theorie erklärt viel, aber **nicht alles** (etwa
> nicht gut, warum die Anfälligkeit zwischen Menschen so **massiv** streut — Frauen berichten in
> Studien häufiger Symptome, was teils auf IPD-Passform zurückgeführt wird). Konkurrierend ist
> die **Postural-Instability-Theorie** (Riccio & Stoffregen): Übelkeit folgt aus länger
> anhaltender Unfähigkeit, die Körperhaltung zu stabilisieren. Beide sind vermutlich Teilwahrheiten.

### 4.2 Interaktionstechniken

**Selektion & Manipulation:**
- **Virtual Hand** — direkt zugreifen. Natürlich, aber nur in Armreichweite.
- **Ray-Casting** — ein Strahl aus der Hand, „Laserpointer". Reichweite unbegrenzt; aber die
  **Winkelpräzision** limitiert: In der Ferne bedeutet 1° Handzittern viele Zentimeter Versatz.
- **Go-Go** — nichtlineare Armverlängerung: bis zu einer Schwelle 1:1, darüber wächst der
  virtuelle Arm überproportional. Verbindet Natürlichkeit mit Reichweite.
- **Fitts' Law** gilt auch hier und quantifiziert die Zielzeit:
  $$MT = a + b\log_2\!\Big(\frac{D}{W}+1\Big)$$
  ($D$ = Distanz, $W$ = Zielbreite). Der Term $\log_2(D/W+1)$ ist der **Index of Difficulty**.
  Praktische Folge: Zielgröße hilft **logarithmisch** — kleine Ziele sind teuer, und in 3D nutzt
  man die **Winkelgröße**, nicht die metrische.

**Locomotion** (siehe 4.1 für den Komfort-Aspekt): Teleport · kontinuierlich (Stick) ·
**Room-Scale** (echtes Gehen — bester Komfort, begrenzt durch das Zimmer) · **Redirected
Walking** (die Welt wird unmerklich gedreht, sodass der Nutzer im Kreis läuft, aber geradeaus zu
gehen glaubt — verblüffend wirksam, braucht aber viel Platz).

---

## 5 · Evaluation: XR ist eine empirische Disziplin

### 5.1 Warum überhaupt Nutzerstudien?

Für „ist die Latenz < 20 ms?" reicht eine Messung. Aber die eigentlichen Fragen — *Fühlt sich
das präsent an? Wird jemandem übel? Ist es benutzbar?* — sind **nur am Menschen** beantwortbar.
Es gibt keine Offline-Metrik für Präsenz. **Das ist der Grund, warum XR-Forschung im Kern
experimentelle Psychologie mit Technik ist.**

**Etablierte Instrumente** (nimm die validierten, erfinde keine eigenen Fragebögen):
- **IPQ** (Igroup Presence Questionnaire) — Präsenz.
- **SSQ** (Simulator Sickness Questionnaire) — Cybersickness, mit den Subskalen *Nausea*,
  *Oculomotor*, *Disorientation*. **Wichtig: vorher *und* nachher** erheben (Differenz zählt).
- **SUS** (System Usability Scale) — 10 Items, Score 0–100. *(Verwirrend: „SUS" heißt in der
  Präsenzliteratur auch *Slater-Usoh-Steed*-Fragebogen. Kontext beachten.)*
- **NASA-TLX** — subjektive Beanspruchung (mental, körperlich, zeitlich, Leistung, Anstrengung,
  Frustration).
- **Objektiv** daneben: Task-Completion-Time, Fehlerrate, Trajektorien — und physiologisch
  (Herzrate, Hautleitwert) als Sickness-Korrelat.

### 5.2 Studiendesign

- **Within-Subject** (jede Person testet **alle** Bedingungen): weniger Teilnehmer nötig,
  kontrolliert für individuelle Unterschiede — die in XR **riesig** sind (Anfälligkeit,
  VR-Erfahrung). **In XR meist die richtige Wahl.** Preis: **Reihenfolgeeffekte** (Lernen,
  Ermüdung, kumulierte Übelkeit) → **Counterbalancing** (z. B. Latin Square) ist Pflicht.
- **Between-Subject** (jede Person **eine** Bedingung): keine Reihenfolgeeffekte, aber viel mehr
  Teilnehmer nötig. Nötig, wenn eine Bedingung die andere „verdirbt" (wer einmal 6 DoF hatte,
  bewertet 3 DoF anders).

**Auswertung** — das knüpft direkt an Modul 03 an:
- **Skalenniveau beachten:** Fragebogen-Items sind **ordinal** (Likert). Für Einzel-Items sind
  nichtparametrische Tests angebracht: **Wilcoxon signed-rank** (within), **Mann-Whitney U**
  (between). Für gemittelte Subskalen mit vielen Items argumentiert man oft intervallskaliert
  → **paired t-test** / ANOVA. *Beides ist vertretbar — man muss es nur begründen.*
- **Effektstärke berichten, nicht nur p.** Ein signifikanter, aber winziger Effekt ist
  irrelevant. **Cohen's d** bzw. $r=Z/\sqrt{N}$. Faustregel: $d\approx0{,}2$ klein, $0{,}5$
  mittel, $0{,}8$ groß.
- **Mehrfachvergleiche korrigieren.** Wer IPQ, SSQ, SUS, TLX und Zeit einzeln testet, macht 5+
  Tests — bei $\alpha=0{,}05$ ist ein Fehlalarm dann fast garantiert (**Bonferroni**: $\alpha/m$;
  oder Holm/FDR). *Das ist derselbe Gedanke wie die Basisraten-Diskussion in Modul 15: viele
  Tests × kleine Fehlerrate = viele Fehlalarme.*
- **Stichprobengröße vorher** planen (Power-Analyse). N=8 findet nur Elefanten. Typische
  XR-Studien: N=20–40.
- **Ethik:** Cybersickness ist eine reale Belastung. Abbruchmöglichkeit ohne Begründung,
  Aufklärung, Pausen, keine Fahrt nach Hause direkt nach einer Sickness-Studie.

---

## 6 · Zusammenfassung / Cheat-Sheet

**Begriffe.** Reality-Virtuality-Kontinuum (Milgram) · **Immersion** = Technik (objektiv) ·
**Präsenz** = Erleben (subjektiv) · **Place Illusion** (Tracking+Latenz!) + **Plausibility
Illusion** (Slater).

**Mensch.** FoV ~200° vs. Headset ~110° · **VAC**: Vergenz folgt dem Objekt, Akkommodation klebt
auf ~2 m → nichts näher als ~50 cm · Vestibularsystem **kann man nicht täuschen**.

**Tracking.** 3 DoF (nur Orientierung) vs. **6 DoF** (+ Position) · outside-in vs. **inside-out
(SLAM)** · IMU = Gyro (schnell, **driftet**) + Accel (verrauscht, **driftfrei**).

**Rotationen.** Euler: intuitiv, aber **nicht kommutativ** + **Gimbal Lock** (Pitch ±90° ⟹ nur
(yaw−roll) überlebt, 3→2 DoF) · Matrizen: 9 Zahlen, driften · **Quaternionen**:
$q=(\cos\frac\theta2,\ \hat{\mathbf n}\sin\frac\theta2)$, $\mathbf v'=q\mathbf vq^{-1}$,
$q\equiv-q$ (Vorzeichen prüfen!) · **SLERP** = konstante Winkelgeschwindigkeit, LERP nicht.

**Fusion.** $\theta_k=\alpha(\theta_{k-1}+\omega_k\Delta t)+(1-\alpha)\theta_{\text{accel},k}$
⟹ besser als beide Sensoren einzeln.

**Latenz.** **Motion-to-Photon < 20 ms** · 90 Hz ⟹ 11,1 ms je Frame · **Prediction**
(extrapolieren, Overshoot bei Richtungswechsel) · **Timewarp** (fertiges Bild nachkorrigieren;
Rotation ~gratis, Position ⟹ **Disokklusion**).

**Cybersickness.** **Sensory Conflict**: Augen sehen Bewegung (**Vection**), Vestibularsystem
nicht ⟹ Gehirn vermutet Gift. Hebel: Latenz > künstliche Fortbewegung > **Beschleunigung**.
Gegenmittel: Teleport, Snap-Turn, Vignetting, statischer Bezugsrahmen.

**Interaktion.** Virtual Hand · **Ray-Casting** (Winkelpräzision!) · Go-Go · **Fitts**:
$MT=a+b\log_2(D/W+1)$.

**Evaluation.** **IPQ** (Präsenz) · **SSQ** (Sickness, vorher/nachher!) · SUS · NASA-TLX ·
**within-subject + Counterbalancing** · ordinal ⟹ **Wilcoxon/Mann-Whitney** · **Effektstärke**
(Cohen's d) · **Bonferroni** · Power vorher planen.

---

## 7 · Selbsttest

<details>
<summary><b>1.</b> Immersion vs. Präsenz — und warum ist ein hässliches, schnelles System besser als ein schönes, langsames?</summary>

**Immersion** = objektive Technik-Eigenschaft (FoV, Latenz, Tracking). **Präsenz** = subjektives
Gefühl, dort zu sein. Die **Place Illusion** entsteht aus **sensomotorischer Kontingenz** — die
Welt muss auf Kopfbewegung reagieren *wie in echt*. Das hängt an **Tracking und Latenz**, nicht
an der Grafik. Ein latenzarmer Comic-Raum erzeugt daher mehr Präsenz als eine fotorealistische
Szene mit 50 ms Verzug (die zusätzlich krank macht).
</details>

<details>
<summary><b>2.</b> Was ist der Vergenz-Akkommodations-Konflikt und was folgt daraus fürs Design?</summary>

**Vergenz** (Augen drehen nach innen) folgt dem virtuellen Objekt, **Akkommodation** (Linse
stellt scharf) klebt auf der festen Displaydistanz (~2 m). In der Realität sind beide gekoppelt —
im Headset widersprechen sie sich ⟹ Ermüdung, Kopfschmerz. **Design-Regel: nichts Wichtiges
näher als ~50 cm.** Echte Lösungen (varifokale/Lichtfeld-Optik) sind Forschung.
</details>

<details>
<summary><b>3.</b> Erkläre Gimbal Lock präzise. Welche Kombinationen werden ununterscheidbar?</summary>

Bei **Pitch = ±90°** fallen yaw- und roll-Achse zusammen ⟹ **3 DoF → 2 DoF**. Es überlebt nur
die **Differenz** $(\text{yaw}-\text{roll})$: (0°,0°), (40°,40°), (90°,90°) ergeben **exakt
dieselbe** Rotation (0,000000° Abstand). **Nicht** gleich sind dagegen (0°,40°) und (40°,0°) —
ihre Differenzen unterscheiden sich um 80°. Quaternionen haben das Problem nicht.
</details>

<details>
<summary><b>4.</b> Warum $\theta/2$ im Quaternion — und warum ist $q\equiv-q$ praktisch relevant?</summary>

Weil die Rotation als $\mathbf v'=q\mathbf vq^{-1}$ **zweimal** wirkt (einmal $q$, einmal
$q^{-1}$) — jede Hälfte trägt $\theta/2$ bei. Folge ist die **doppelte Überdeckung**: $q$ und
$-q$ sind dieselbe Rotation. **Praktisch:** Beim Interpolieren muss man das Vorzeichen prüfen
(ggf. $q_1 \to -q_1$), sonst nimmt SLERP den **langen Weg** (359° statt 1°).
</details>

<details>
<summary><b>5.</b> SLERP vs. LERP — was ist der Unterschied, und wann ist LERP trotzdem ok?</summary>

**SLERP** läuft den **Großkreis** auf der Einheitssphäre ⟹ kürzeste Drehung mit **konstanter
Winkelgeschwindigkeit**. **LERP** (mitteln + normieren) läuft die **Sehne** ⟹ die
Winkelgeschwindigkeit schwankt (gemessen: Streuung der Schritte 0,000 vs. 5,659 bei 0°→170°).
Bei **kleinen** Winkeln ist der Fehler vernachlässigbar — deshalb ist LERP zwischen dichten
Frames üblich (billiger).
</details>

<details>
<summary><b>6.</b> Warum driftet ein Gyroskop, und wie repariert der Komplementärfilter das?</summary>

Das Gyro misst Winkel**geschwindigkeit**; für den Winkel muss man **integrieren** — dabei
summiert sich jeder **Bias** unbegrenzt auf (0,5 °/s ⟹ 30° nach einer Minute; gemessen RMSE
16,98°). Das **Accelerometer** liefert über die **Schwerkraft** einen absoluten, driftfreien,
aber verrauschten Bezug (RMSE 3,01°). Der Filter
$\theta_k=\alpha(\theta_{k-1}+\omega_k\Delta t)+(1-\alpha)\theta_{\text{accel}}$ nimmt kurzfristig
das Gyro, langfristig das Accel ⟹ RMSE **0,42°**, besser als beide.
</details>

<details>
<summary><b>7.</b> Was ist Motion-to-Photon-Latenz, wo liegt die Grenze, und warum stört 60 ms am Monitor nicht?</summary>

Zeit von der Kopfbewegung bis zum passenden Photon auf der Netzhaut; Kette aus Sensor →
Fusion → App → Rendering → Scanout → Display. **Ziel < 20 ms.** Am Monitor stört 60 ms nicht,
weil dort **kein Vestibularsystem widerspricht** — im Headset erzeugt die Verzögerung genau den
sensorischen Konflikt, der übel macht.
</details>

<details>
<summary><b>8.</b> Was macht Timewarp — und warum hilft es bei Drehung besser als bei Translation?</summary>

Es korrigiert das **fertig gerenderte** Bild kurz vor der Anzeige anhand der **neuesten** Pose.
Eine reine **Drehung** lässt sich fast perfekt nachschieben (nur ein anderer Bildausschnitt). Bei
**Translation** ändert sich die **Verdeckung**: hinter Vordergrundobjekten müsste Information
auftauchen, die nie gerendert wurde (**Disokklusion**) ⟹ Artefakte/Raten. Timewarp senkt die
Latenz **nicht**, es macht sie für die Orientierung **unsichtbar**.
</details>

<details>
<summary><b>9.</b> Erkläre Cybersickness über die Sensory-Conflict-Theorie. Warum hilft Teleportation?</summary>

Augen melden Eigenbewegung (**Vection**), das Vestibularsystem meldet Stillstand. Der Widerspruch
ähnelt dem Muster einer **Vergiftung** ⟹ Übelkeit. **Teleportation** erzeugt **keinen optischen
Fluss** ⟹ kein Konflikt. Weitere Hebel: Latenz senken, **Beschleunigung** vermeiden, Snap-Turn,
Vignetting, statischer Bezugsrahmen. (Seekrankheit ist derselbe Konflikt mit vertauschten Rollen.)
</details>

<details>
<summary><b>10.</b> Du vergleichst 3 DoF vs. 6 DoF und misst IPQ, SSQ, SUS, TLX und Zeit. Nenne drei methodische Pflichten.</summary>

Beliebige drei: **Within-Subject mit Counterbalancing** (Reihenfolgeeffekte: Lernen, Ermüdung,
kumulierte Übelkeit) · **SSQ vorher und nachher** erheben (Differenz zählt) · **ordinale**
Likert-Items ⟹ **Wilcoxon** statt t-Test (oder begründen) · **Effektstärke** berichten, nicht nur
p · **Mehrfachvergleiche korrigieren** (5 Tests ⟹ Bonferroni/Holm) · **Power vorher** planen
(N=20–40) · **Ethik** (Abbruch jederzeit).
</details>

---

## 8 · Literatur & Quellen

**Standardwerke:**
- 📗 **LaValle — *Virtual Reality*** (Cambridge; **frei online**: lavalle.pl/vr/). *Die*
  Referenz für genau dieses Modul: Wahrnehmung, Tracking, Rotationsmathematik, Latenz — mit
  Mathematik, aber lesbar. **Beste Einzelquelle.** Kap. 3 (Transformationen), 9 (Tracking),
  12 (Sickness).
- 📗 **Jerald — *The VR Book: Human-Centered Design for VR***. Stark auf Wahrnehmung,
  Sickness und Interaktionsdesign. *Einsteigerfreundlich.*
- 📗 **Bowman et al. — *3D User Interfaces: Theory and Practice***. Das Standardwerk zu
  Interaktionstechniken (→ vertieft in Modul 19). *Vertiefend.*

**Schlüsselpaper:**
- 📄 **Milgram & Kishino (1994), *A Taxonomy of Mixed Reality Visual Displays*** — das Kontinuum.
- 📄 **Slater (2009), *Place Illusion and Plausibility Illusion*** (Phil. Trans. R. Soc.) —
  die begriffliche Grundlage von 1.2. *Frei, kurz, lesenswert.*
- 📄 **Reason & Brady (1975), *Motion Sickness*** — Sensory-Conflict-Theorie.
- 📄 **Riccio & Stoffregen (1991), *An Ecological Theory of Motion Sickness*** — die
  konkurrierende Postural-Instability-Theorie.
- 📄 **Shoemake (1985), *Animating Rotation with Quaternion Curves*** (SIGGRAPH) — **SLERP**.
- 📄 **Van Waveren (2016), *The Asynchronous Time Warp for VR on Mobile Hardware***.
- 📄 **Poupyrev et al. (1996), *The Go-Go Interaction Technique***.
- 📄 **Razzaque et al. (2001), *Redirected Walking***.

**Mathematik/Praxis:**
- 🌐 **3Blue1Brown — *Visualizing quaternions*** (eater.net/quaternions) — interaktiv,
  hervorragend, wenn Quaternionen abstrakt bleiben. *Einsteigerfreundlich, frei.*
- 🌐 **scipy `spatial.transform.Rotation`-Doku** — die API der Projekte.
- 🌐 **OpenXR-Spezifikation** (khronos.org/openxr) — der offene Industriestandard; die
  Begriffe (Pose, Space, Predicted Display Time) tauchen dort genau so auf.
- 🌐 **Unity XR Interaction Toolkit** / **Oculus Developer Blog** (Carmack/Abrash zu Latenz) —
  für die Praxis.

**Evaluation:**
- 📄 **Kennedy et al. (1993), *Simulator Sickness Questionnaire (SSQ)***.
- 📄 **Schubert et al. (2001), *The Experience of Presence: Factor Analytic Insights*** (IPQ;
  igroup.org/pq/ipq — frei verfügbar, inkl. Items).
- 📄 **Hart & Staveland (1988), *NASA-TLX***; **Brooke (1996), *SUS***.
- 📗 **Hyndman-artig fürs Handwerk:** *Field — Discovering Statistics* für die Testwahl, oder
  Modul 03 dieses Repos.

---

## Nächstes Modul

**Modul 18 — Multimodal Interfaces** erweitert die Interaktion über Hände hinaus (Sprache,
Blick, Gesten, Haptik) und fragt, wie man Modalitäten **fusioniert**. **Modul 19 — 3D User
Interfaces** vertieft dann systematisch die Interaktionstechniken aus 4.2. Was du hier gelernt
hast — **Präsenz hängt an Latenz und Tracking**, Quaternionen, und dass am Ende ein **Mensch**
im Experiment entscheidet — trägt durch den gesamten XR-Block.
