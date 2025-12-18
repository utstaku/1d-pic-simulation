import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 

# -------------------------
# パラメータ
# -------------------------
q_over_m = -1.0              # q/m（電子を仮想的に -1 と正規化）
E = np.array([0.0, 1.0, 0.0])   # 背景磁場
B = np.array([0.0, 0.0, 1.0])

dt = 0.05       # タイムステップ
steps = 400     # ステップ数

# -------------------------
# 初期条件（粒子は 1 個）
# -------------------------
x = np.array([0.0, 0.0, 0.0])   # 位置ベクトル (x,y,z)
v = np.array([2.0, 0.0, 1.0])   # 速度ベクトル (vx,vy,vz)

# -------------------------
# Buneman–Boris 法（単粒子）
# -------------------------
def boris_push(v, q_over_m, E, B, dt):
    """
    v : (3,)  速度ベクトル
    E : (3,)  一様電場
    B : (3,)  一様磁場
    """
    v_minus = v + 0.5 * q_over_m * dt * E

    v0 = v_minus + 0.5 * q_over_m * dt * np.cross(v_minus, B)

    v_plus = v_minus + 1/(1 + (q_over_m * dt*0.5)**2 * (np.dot(B, B))) * (q_over_m) * np.cross(v0, B) * dt

    v_new = v_plus + 0.5 * q_over_m * dt * E

    return v_new

traj = np.zeros((steps, 3))  # 各ステップの位置 (x,y,z)

for n in range(steps):
    x += v * dt * 0.5 #xを半ステップ進める
    v = boris_push(v, q_over_m, E, B, dt) #Boris法で速度を更新
    x += v * dt * 0.5 #xを半ステップ進める
    traj[n] = x #位置を記録

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 軌道から表示範囲を決める
xmin, xmax = traj[:,0].min(), traj[:,0].max()
ymin, ymax = traj[:,1].min(), traj[:,1].max()
zmin, zmax = traj[:,2].min(), traj[:,2].max()

# ちょっとマージンをつける
mx = 0.2 * (xmax - xmin + 1e-9)
my = 0.2 * (ymax - ymin + 1e-9)
mz = 0.2 * (zmax - zmin + 1e-9)

ax.set_xlim(xmin - mx, xmax + mx)
ax.set_ylim(ymin - my, ymax + my)
ax.set_zlim(zmin - mz, zmax + mz)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Single particle in 3D (B along z)')

# 軌跡（線）と現在位置（点）の Artist を作成
line, = ax.plot([], [], [], lw=1)      # 通った軌跡
point, = ax.plot([], [], [], 'o')      # 現在位置

# 初期化関数
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

# フレームごとの更新
def update(frame):
    # 0〜frame までの軌跡
    line.set_data(traj[:frame, 0], traj[:frame, 1])     # x, y
    line.set_3d_properties(traj[:frame, 2])             # z

    # 現在位置（長さ1の配列にするのがポイント）
    point.set_data([traj[frame, 0]], [traj[frame, 1]])
    point.set_3d_properties([traj[frame, 2]])

    return line, point

ani = animation.FuncAnimation(
    fig, update, frames=steps,
    init_func=init, blit=True, interval=30
)

plt.show()
ani.save("single_particle_orbit_BZvertical.gif", writer="ffmpeg")
