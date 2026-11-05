"""
Physics-aware 3D ball tracking demo
-----------------------------------
A small, self-contained simulation of the problem behind EA SPORTS TRACAB's
"Physics-Aware 3D Ball Tracking" thesis:

1. Simulate a football in flight (gravity + quadratic air drag + bounces).
2. Observe it with two calibrated cameras (pinhole model) with pixel noise
   and occlusions (frames where a camera, or both, cannot see the ball).
3. Estimate the 3D trajectory two ways:
     a) frame-by-frame triangulation (only works when BOTH cameras see the ball)
     b) an Extended Kalman Filter (EKF) that uses the physics model as its
        prediction step and fuses whatever camera measurements are available.
4. Compare the errors and plot the results.

Run:  python ball_tracking_demo.py
Needs: numpy, matplotlib   (pip install numpy matplotlib)
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------- Physics model ----------------
G = np.array([0.0, 0.0, -9.81])          # gravity [m/s^2]
RHO, CD, R, M = 1.2, 0.25, 0.11, 0.43    # air density, drag coeff, ball radius [m], mass [kg]
K_DRAG = 0.5 * RHO * CD * np.pi * R**2 / M   # quadratic drag constant [1/m]
E_BOUNCE, MU_GROUND = 0.7, 0.8           # vertical restitution, horizontal speed kept after bounce


def accel(v):
    """Acceleration from gravity and quadratic air drag."""
    return G - K_DRAG * np.linalg.norm(v) * v


def step(x, dt):
    """Advance state x = [px, py, pz, vx, vy, vz] by dt (semi-implicit Euler + bounce)."""
    p, v = x[:3].copy(), x[3:].copy()
    v = v + accel(v) * dt
    p = p + v * dt
    if p[2] < 0.0:                       # ground contact -> bounce
        p[2] = -p[2]
        v[2] = -E_BOUNCE * v[2]
        v[:2] *= MU_GROUND
    return np.concatenate([p, v])


# ---------------- Ground truth trajectory ----------------
FPS = 50
DT = 1.0 / FPS
N = 150                                   # 3 seconds
x_true = np.zeros((N, 6))
x_true[0] = [0.0, 0.0, 0.3, 18.0, 4.0, 9.0]   # a lofted pass
for k in range(1, N):
    x = x_true[k - 1]
    for _ in range(20):                   # fine substeps for an accurate "truth"
        x = step(x, DT / 20)
    x_true[k] = x

# ---------------- Cameras ----------------
def look_at(cam_pos, target):
    """Rotation matrix (world -> camera) for a camera at cam_pos looking at target."""
    z = target - cam_pos; z /= np.linalg.norm(z)          # optical axis
    x = np.cross(z, np.array([0, 0, 1.0])); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])

K_INT = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1.0]])   # 1920x1080 camera
cams = []
for pos in ([-15.0, -45.0, 18.0], [55.0, -45.0, 18.0]):
    c = np.array(pos)
    Rm = look_at(c, np.array([20.0, 3.0, 2.0]))
    t = -Rm @ c
    cams.append(K_INT @ np.hstack([Rm, t[:, None]]))   # 3x4 projection matrix


def project(P, p):
    h = P @ np.append(p, 1.0)
    return h[:2] / h[2]


PIX_NOISE = 2.0                           # pixel noise (std)
visible = np.ones((N, 2), dtype=bool)
visible[40:55, 0] = False                 # camera 1 occluded by a player
visible[90:110, :] = False                # ball hidden from BOTH cameras (e.g. a crowd of players)
meas = np.full((N, 2, 2), np.nan)
for k in range(N):
    for c in range(2):
        if visible[k, c]:
            meas[k, c] = project(cams[c], x_true[k, :3]) + rng.normal(0, PIX_NOISE, 2)


# ---------------- Method A: triangulation (DLT) ----------------
def triangulate(uv1, uv2):
    A = []
    for P, (u, v) in ((cams[0], uv1), (cams[1], uv2)):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]

tri = np.full((N, 3), np.nan)
for k in range(N):
    if visible[k].all():
        tri[k] = triangulate(meas[k, 0], meas[k, 1])


# ---------------- Method B: physics-aware EKF ----------------
def num_jacobian(f, x, eps=1e-5):
    fx = f(x)
    J = np.zeros((fx.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x); dx[i] = eps
        J[:, i] = (f(x + dx) - fx) / eps
    return J

Q = np.diag([1e-4] * 3 + [5e-2] * 3)      # process noise: model is good but not perfect (spin, wind)
x_est = np.zeros((N, 6))
P_est = np.zeros((N, 6, 6))
x = np.concatenate([tri[0], [15.0, 0.0, 8.0]])   # rough initial guess (velocity unknown)
P = np.diag([1.0] * 3 + [25.0] * 3)
for k in range(N):
    if k > 0:                                       # predict with physics
        F = num_jacobian(lambda s: step(s, DT), x)
        x = step(x, DT)
        P = F @ P @ F.T + Q
    cams_k = [c for c in range(2) if visible[k, c]]
    if cams_k:                                      # update with available cameras
        h = lambda s: np.concatenate([project(cams[c], s[:3]) for c in cams_k])
        z = np.concatenate([meas[k, c] for c in cams_k])
        H = num_jacobian(h, x)
        Rn = np.eye(len(z)) * PIX_NOISE**2
        S = H @ P @ H.T + Rn
        Kg = P @ H.T @ np.linalg.inv(S)
        x = x + Kg @ (z - h(x))
        P = (np.eye(6) - Kg @ H) @ P
    x_est[k], P_est[k] = x, P

# ---------------- Results ----------------
err_tri = np.linalg.norm(tri - x_true[:, :3], axis=1)
err_ekf = np.linalg.norm(x_est[:, :3] - x_true[:, :3], axis=1)
both = visible.all(axis=1)
gap = ~visible.any(axis=1)
one = visible.any(axis=1) & ~both
print(f"Drag constant k = {K_DRAG:.4f} 1/m")
print(f"Frames: {N} | both cameras: {both.sum()} | one camera: {one.sum()} | none: {gap.sum()}")
print(f"Triangulation RMSE (only when both see the ball): {np.sqrt(np.nanmean(err_tri[both]**2)):.3f} m")
print(f"EKF RMSE when both cameras see the ball:           {np.sqrt(np.mean(err_ekf[both]**2)):.3f} m")
print(f"EKF RMSE with only one camera:                     {np.sqrt(np.mean(err_ekf[one]**2)):.3f} m")
print(f"EKF RMSE during full occlusion (prediction only):  {np.sqrt(np.mean(err_ekf[gap]**2)):.3f} m")
print(f"Triangulation coverage: {both.mean()*100:.0f}% of frames | EKF coverage: 100% of frames")

t = np.arange(N) * DT
sigma = np.sqrt(P_est[:, 2, 2])
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(t, x_true[:, 2], "k-", label="true height")
ax[0].plot(t, tri[:, 2], "o", ms=3, alpha=0.6, label="triangulation")
ax[0].plot(t, x_est[:, 2], "r-", label="physics-aware EKF")
ax[0].fill_between(t, x_est[:, 2] - 2 * sigma, x_est[:, 2] + 2 * sigma, color="r", alpha=0.15, label="EKF ±2σ")
ax[0].axvspan(90 * DT, 110 * DT, color="grey", alpha=0.2, label="ball hidden from both cameras")
ax[0].set_ylabel("height z [m]"); ax[0].legend(loc="upper right"); ax[0].set_title("3D ball tracking: height over time")
ax[1].plot(t, err_tri, "o", ms=3, alpha=0.6, label="triangulation error")
ax[1].plot(t, err_ekf, "r-", label="EKF error")
ax[1].axvspan(90 * DT, 110 * DT, color="grey", alpha=0.2)
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("3D position error [m]"); ax[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig("ball_tracking_results.png", dpi=130)
print("Saved plot: ball_tracking_results.png")
