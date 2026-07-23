"""Mobile navigation: RRT planning + particle filter localisation + path following.
(Reference solution P03-final)   Module 21 — Robotics 1.

The complete sense-plan-act cycle in 2D, from scratch (numpy only):
  PLAN  : RRT in the configuration space (a point robot -> C-space = workspace) + shortcut smoothing
  SENSE : unicycle odometry (drifts!) + range measurements to known landmarks
          -> Monte Carlo localisation (particle filter with systematic resampling)
  ACT   : pure pursuit controller on the ESTIMATED pose
"""
import numpy as np


# ==========================================================================
# World
# ==========================================================================
class World:
    """2D world with circular obstacles."""

    def __init__(self, obstacles, bounds=(0.0, 10.0, 0.0, 10.0)):
        self.obstacles = [np.asarray(o, float) for o in obstacles]  # (cx, cy, r)
        self.bounds = bounds

    def in_bounds(self, p):
        x0, x1, y0, y1 = self.bounds
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def in_collision(self, p, margin=0.0):
        p = np.asarray(p, float)[:2]
        if not self.in_bounds(p):
            return True
        for o in self.obstacles:
            if np.hypot(*(p - o[:2])) <= o[2] + margin:
                return True
        return False

    def segment_free(self, a, b, step=0.05, margin=0.0):
        """Collision test along the segment a->b (dense sampling)."""
        a = np.asarray(a, float)[:2]
        b = np.asarray(b, float)[:2]
        d = np.linalg.norm(b - a)
        n = max(2, int(np.ceil(d / step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            if self.in_collision(a + t * (b - a), margin):
                return False
        return True


def default_world():
    """The standard test world: the obstacles form a bottleneck in the middle."""
    return World([(3.0, 2.0, 1.1), (3.0, 6.5, 1.4), (6.5, 4.0, 1.3),
                  (7.5, 8.0, 1.0), (1.5, 8.5, 0.9)])


LANDMARKS = np.array([[0.5, 0.5], [9.5, 0.5], [0.5, 9.5], [9.5, 9.5], [5.0, 5.0]])


# ==========================================================================
# PLAN — RRT
# ==========================================================================
def rrt(start, goal, world, step=0.5, goal_bias=0.08, max_iter=5000, goal_tol=0.5,
        rng=None, margin=0.0):
    """Rapidly-exploring random tree. Returns (path, nodes, parents); path=None on failure."""
    rng = np.random.default_rng(0) if rng is None else rng
    start = np.asarray(start, float)[:2]
    goal = np.asarray(goal, float)[:2]
    nodes = [start]
    parents = [-1]
    x0, x1, y0, y1 = world.bounds
    for _ in range(max_iter):
        # 1. sample (with goal bias)
        if rng.random() < goal_bias:
            q_rand = goal
        else:
            q_rand = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        # 2. nearest tree node
        arr = np.array(nodes)
        i_near = int(np.argmin(np.linalg.norm(arr - q_rand, axis=1)))
        q_near = nodes[i_near]
        # 3. step of length `step` towards q_rand
        d = q_rand - q_near
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            continue
        q_new = q_near + (min(step, nrm) / nrm) * d
        # 4. collision test of the edge
        if not world.segment_free(q_near, q_new, margin=margin):
            continue
        nodes.append(q_new)
        parents.append(i_near)
        # 5. goal reached?
        if np.linalg.norm(q_new - goal) <= goal_tol and world.segment_free(q_new, goal, margin=margin):
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            # recover the path by tracing back the parents
            path = []
            i = len(nodes) - 1
            while i != -1:
                path.append(nodes[i])
                i = parents[i]
            return np.array(path[::-1]), np.array(nodes), parents
    return None, np.array(nodes), parents


def shortcut(path, world, rng=None, iters=300, margin=0.0):
    """Path smoothing: repeatedly try to connect two points directly."""
    if path is None or len(path) < 3:
        return path
    rng = np.random.default_rng(0) if rng is None else rng
    P = [p for p in path]
    for _ in range(iters):
        if len(P) < 3:
            break
        i = rng.integers(0, len(P) - 2)
        j = rng.integers(i + 2, len(P))
        if world.segment_free(P[i], P[j], margin=margin):
            P = P[:i + 1] + P[j:]
    return np.array(P)


def path_length(path):
    if path is None or len(path) < 2:
        return np.inf
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))


def densify(path, ds=0.1):
    """Break the path into fine, evenly spaced waypoints.

    After the smoothing the path consists of few, widely spaced corners. Pure pursuit,
    however, needs a lookahead point ON the path — otherwise the target jumps and the
    controller oscillates. Therefore interpolate between the corners.
    """
    P = np.asarray(path, float)
    if len(P) < 2:
        return P
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        d = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(d / ds)))
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.array(out)


# ==========================================================================
# SENSE — motion and measurement model
# ==========================================================================
def unicycle_step(state, v, omega, dt, rng=None, noise=(0.0, 0.0)):
    """Unicycle model: x += v cos(theta) dt, y += v sin(theta) dt, theta += omega dt.
    noise = (sigma_v, sigma_omega) as additive noise on the control inputs."""
    x, y, th = state
    if rng is not None and (noise[0] > 0 or noise[1] > 0):
        v = v + rng.normal(0, noise[0])
        omega = omega + rng.normal(0, noise[1])
    return np.array([x + v * np.cos(th) * dt,
                     y + v * np.sin(th) * dt,
                     _wrap(th + omega * dt)])


def _wrap(a):
    """Normalise an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def measure_ranges(state, landmarks=LANDMARKS, sigma_r=0.25, rng=None):
    """Ranges to all (known) landmarks, optionally noisy."""
    d = np.linalg.norm(landmarks - np.asarray(state)[:2], axis=1)
    if rng is not None and sigma_r > 0:
        d = d + rng.normal(0, sigma_r, d.shape)
    return d


# ==========================================================================
# SENSE — particle filter (Monte Carlo localisation)
# ==========================================================================
class ParticleFilter:
    """Particle filter with systematic resampling and the N_eff criterion."""

    def __init__(self, n_particles, init_pose, init_spread=(0.5, 0.5, 0.3), rng=None):
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.M = n_particles
        p = np.asarray(init_pose, float)
        self.particles = np.column_stack([
            self.rng.normal(p[0], init_spread[0], n_particles),
            self.rng.normal(p[1], init_spread[1], n_particles),
            _wrap(self.rng.normal(p[2], init_spread[2], n_particles))])
        self.weights = np.full(n_particles, 1.0 / n_particles)

    def predict(self, v, omega, dt, noise):
        """Motion model on every particle (with noise -> the belief becomes blurrier)."""
        vs = v + self.rng.normal(0, noise[0], self.M)
        ws = omega + self.rng.normal(0, noise[1], self.M)
        th = self.particles[:, 2]
        self.particles[:, 0] += vs * np.cos(th) * dt
        self.particles[:, 1] += vs * np.sin(th) * dt
        self.particles[:, 2] = _wrap(th + ws * dt)

    def update(self, z, landmarks=LANDMARKS, sigma_r=0.25):
        """Measurement model: w *= p(z | x) under the Gaussian assumption (belief sharpens)."""
        d = np.linalg.norm(landmarks[None, :, :] - self.particles[:, None, :2], axis=2)
        # log likelihood (numerically stable), then normalise
        ll = -0.5 * np.sum(((z[None, :] - d) / sigma_r) ** 2, axis=1)
        ll -= ll.max()
        w = self.weights * np.exp(ll)
        s = w.sum()
        self.weights = np.full(self.M, 1.0 / self.M) if s <= 1e-300 else w / s

    def n_eff(self):
        return 1.0 / np.sum(self.weights ** 2)

    def resample_if_needed(self, threshold_ratio=0.5):
        """Systematic resampling only on particle depletion (N_eff < M/2)."""
        if self.n_eff() >= threshold_ratio * self.M:
            return False
        positions = (self.rng.random() + np.arange(self.M)) / self.M
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.clip(idx, 0, self.M - 1)
        self.particles = self.particles[idx]
        self.weights = np.full(self.M, 1.0 / self.M)
        return True

    def estimate(self):
        """Weighted mean; the orientation via unit vectors (circular mean)."""
        x = self.weights @ self.particles[:, 0]
        y = self.weights @ self.particles[:, 1]
        c = self.weights @ np.cos(self.particles[:, 2])
        s = self.weights @ np.sin(self.particles[:, 2])
        return np.array([x, y, np.arctan2(s, c)])


# ==========================================================================
# ACT — pure pursuit path following
# ==========================================================================
def pure_pursuit(path, pose, lookahead=0.8, k_omega=2.0, v_max=0.8, omega_max=1.5):
    """Control commands (v, omega) to follow the path.

    Finds the most advanced path point within the lookahead radius and turns proportionally to
    the heading error towards it (a P controller on the orientation).

    IMPORTANT: omega is CLAMPED to omega_max. Without saturation the robot turns too far per time
    step (omega*dt) at a large heading error, overshoots and oscillates.
    The path should be refined with densify() beforehand.
    """
    P = np.asarray(path, float)
    pos = np.asarray(pose)[:2]
    d = np.linalg.norm(P - pos, axis=1)
    i_close = int(np.argmin(d))
    target = P[-1]
    for i in range(i_close, len(P)):
        if np.linalg.norm(P[i] - pos) >= lookahead:
            target = P[i]
            break
    heading_err = _wrap(np.arctan2(*(target - pos)[::-1]) - pose[2])
    omega = float(np.clip(k_omega * heading_err, -omega_max, omega_max))
    # drive more slowly at a large heading error (turn first, then drive)
    v = v_max * max(0.15, np.cos(np.clip(heading_err, -np.pi/2, np.pi/2)))
    return v, omega
