import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

# ============================================================
# OUTPUT
# ============================================================
OUT = "Fisiologia_Respiratoria_2_AulaBasica_Master_Spontanea_RC_v2_FIX.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 20
DURATION_S = 60

# ============================================================
# STEPWISE CYCLE (7s)
# CRF hold 1s -> Insp 2s -> End-insp hold 1s -> Exp 2s -> End-exp hold 1s
# ============================================================
T_HOLD_EE  = 1.0
T_INSP     = 2.0
T_HOLD_EI  = 1.0
T_EXP      = 2.0
T_HOLD_EE2 = 1.0
T_CYCLE    = T_HOLD_EE + T_INSP + T_HOLD_EI + T_EXP + T_HOLD_EE2  # 7s

# ============================================================
# Baselines / targets (cmH2O)
# ============================================================
PPL_EE = -5.0
PPL_EI = -7.0
PMUS_PEAK = abs(PPL_EI - PPL_EE)  # 2 cmH2O

# ============================================================
# R-C model parameters (didactic)
# Units:
#   R in cmH2O / (L/s)
#   C in L / cmH2O
# ============================================================
R = 5.0
C = 0.10

# ============================================================
# Didactic VT target for visuals (L)
# ============================================================
VT_TARGET = 0.5

# ============================================================
# Helpers
# ============================================================
def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5*np.cos(np.pi*x)

def phase_in_cycle(tau: float):
    a = T_HOLD_EE
    b = a + T_INSP
    c = b + T_HOLD_EI
    d = c + T_EXP
    if tau < a:
        return "hold_ee", tau / max(T_HOLD_EE, 1e-6)
    if tau < b:
        return "insp", (tau - a) / max(T_INSP, 1e-6)
    if tau < c:
        return "hold_ei", (tau - b) / max(T_HOLD_EI, 1e-6)
    if tau < d:
        return "exp", (tau - c) / max(T_EXP, 1e-6)
    return "hold_ee2", (tau - d) / max(T_HOLD_EE2, 1e-6)

def pmus_of_tau(tau: float) -> float:
    """
    Pmus (cmH2O): 0 em CRF. Sobe durante inspiração, mantém, e relaxa na expiração.
    """
    ph, x = phase_in_cycle(tau)
    if ph in ("hold_ee", "hold_ee2"):
        return 0.0
    if ph == "insp":
        return PMUS_PEAK * smoothstep(x)
    if ph == "hold_ei":
        return PMUS_PEAK
    return PMUS_PEAK * (1.0 - smoothstep(x))

def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def make_history(t: float, window: float, fps: int):
    t0 = max(0.0, t - window)
    n = int(max(90, min(int(window*fps), int((t - t0)*fps + 1))))
    return np.linspace(t0, t, n)

# ============================================================
# Precompute one-cycle signals with RC model (steady-state)
# ============================================================
N_PRE = 5000
tau_one = np.linspace(0.0, T_CYCLE, N_PRE)
dt = tau_one[1] - tau_one[0]

# Simulate multiple cycles to remove drift (use last cycle)
N_CYCLES_SIM = 4
tau_all = np.linspace(0.0, T_CYCLE * N_CYCLES_SIM, N_PRE * N_CYCLES_SIM)
dt_all = tau_all[1] - tau_all[0]

pmus_all = np.array([pmus_of_tau(tau % T_CYCLE) for tau in tau_all])

V_all = np.zeros_like(tau_all)      # L above FRC
Flow_all = np.zeros_like(tau_all)   # L/s

# Dynamics: 0 = R*Flow + V/C - Pmus -> Flow = (Pmus - V/C)/R
for k in range(1, len(tau_all)):
    pm = pmus_all[k-1]
    flow = (pm - V_all[k-1] / C) / R
    V_all[k] = V_all[k-1] + flow * dt_all
    Flow_all[k] = flow

# Extract last cycle
start = (N_CYCLES_SIM - 1) * N_PRE
end = N_CYCLES_SIM * N_PRE
tau_grid = np.linspace(0.0, T_CYCLE, N_PRE)
V = V_all[start:end].copy()
Flow = Flow_all[start:end].copy()
pmus_grid = pmus_all[start:end].copy()

# Remove tiny mismatch at cycle edges by forcing periodic closure (volume only)
# (this prevents subtle end-start inconsistencies when looping)
V = V - np.linspace(0.0, V[-1] - V[0], V.size)

# Compute pressures
Ppl = (PPL_EE - pmus_grid).copy()
# Airway opening pressure = 0 => Palv = -R*Flow (sign gives:
# inspiration Palv<0 => inflow>0)
Palv = (-R * Flow).copy()
PL = Palv - Ppl

# Scale volume to VT_TARGET and scale flow consistently (NO np.gradient -> no "degrau")
Vmin, Vmax = float(np.min(V)), float(np.max(V))
if (Vmax - Vmin) > 1e-9:
    scale_k = VT_TARGET / (Vmax - Vmin)
    V_scaled = (V - Vmin) * scale_k
    Flow_scaled = Flow * scale_k
else:
    V_scaled = V.copy()
    Flow_scaled = Flow.copy()

Palv_scaled = -R * Flow_scaled
Ppl_scaled = Ppl.copy()
PL_scaled = Palv_scaled - Ppl_scaled

def interp_cycle(arr, tau):
    tau = tau % T_CYCLE
    return float(np.interp(tau, tau_grid, arr))

def ppl_of_tau(tau: float) -> float:
    return interp_cycle(Ppl_scaled, tau)

def palv_of_tau(tau: float) -> float:
    return interp_cycle(Palv_scaled, tau)

def pl_of_tau(tau: float) -> float:
    return interp_cycle(PL_scaled, tau)

def flow_of_tau(tau: float) -> float:
    return interp_cycle(Flow_scaled, tau)

def volume_of_tau(tau: float) -> float:
    return interp_cycle(V_scaled, tau)

# ============================================================
# Drawing: anatomical lungs + vessels
# ============================================================
def _mix(c1, c2, a):
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return tuple(np.clip((1-a)*c1 + a*c2, 0, 1))

def draw_lungs_anatomic(ax, center=(0.43, 0.64), scale=1.0, inflate=0.0):
    cx, cy = center
    s = scale * (1.0 + 0.12 * inflate)

    pink = (0.93, 0.62, 0.68)
    pink_dark = (0.78, 0.36, 0.46)
    highlight = (0.98, 0.86, 0.88)

    fill = _mix(pink, highlight, 0.30 + 0.25*inflate)
    edge = _mix(pink_dark, (0.55, 0.20, 0.28), 0.20)

    lobe_w = 0.20 * s
    lobe_h = 0.30 * s
    left_center = (cx - 0.12*s, cy)
    right_center = (cx + 0.12*s, cy)

    left = Ellipse(left_center, width=lobe_w, height=lobe_h, angle=10,
                   facecolor=fill, edgecolor=edge, linewidth=4.0, alpha=0.98)
    right = Ellipse(right_center, width=lobe_w, height=lobe_h, angle=-10,
                    facecolor=fill, edgecolor=edge, linewidth=4.0, alpha=0.98)
    ax.add_patch(left); ax.add_patch(right)
    left.set_clip_on(True); right.set_clip_on(True)

    for side in [-1, 1]:
        x0 = cx + side*0.12*s
        y0 = cy + 0.06*s
        xs = np.linspace(x0 - 0.06*s*side, x0 + 0.02*s*side, 120)
        ys = y0 - 0.05*s*np.sin(np.linspace(0, np.pi, 120))
        ax.plot(xs, ys, color=_mix(edge, (0,0,0), 0.15), lw=1.1, alpha=0.32, clip_on=True)

    notch = Ellipse((cx - 0.08*s, cy - 0.01*s), width=0.11*s, height=0.16*s, angle=25,
                    facecolor=ax.get_facecolor(), edgecolor=ax.get_facecolor(), linewidth=0)
    ax.add_patch(notch); notch.set_clip_on(True)

    # trachea inside thorax
    tr_w = 0.040*s
    tr_h = 0.095*s
    tr_x = cx - tr_w/2
    tr_y = cy + 0.135*s
    tr = FancyBboxPatch((tr_x, tr_y), tr_w, tr_h,
                        boxstyle="round,pad=0.006,rounding_size=0.012",
                        facecolor="#111827", edgecolor="#111827", linewidth=0)
    ax.add_patch(tr); tr.set_clip_on(True)

    bronchi_top = cy + 0.135*s
    bronchi_mid = cy + 0.090*s
    bronchi_low = cy + 0.050*s
    bronchi_verts = [
        (cx, bronchi_top),
        (cx, bronchi_mid),
        (cx - 0.06*s, bronchi_low),
        (cx, bronchi_mid),
        (cx + 0.06*s, bronchi_low),
    ]
    bronchi_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.MOVETO, Path.LINETO]
    bronchi = PathPatch(Path(bronchi_verts, bronchi_codes),
                        edgecolor="#111827", linewidth=3.6, facecolor="none", capstyle="round")
    ax.add_patch(bronchi); bronchi.set_clip_on(True)

    # vessels
    vs = (1.0 + 0.10*inflate)
    red = (0.86, 0.18, 0.18)
    blue = (0.12, 0.45, 0.78)

    def branch_path(points):
        verts, codes = [], []
        for j, (x,y) in enumerate(points):
            verts.append((x,y))
            codes.append(Path.MOVETO if j==0 else Path.LINETO)
        return Path(verts, codes)

    rv = [(cx + 0.02*s, cy + 0.10*s),
          (cx + 0.08*s*vs, cy + 0.07*s),
          (cx + 0.13*s*vs, cy + 0.02*s),
          (cx + 0.14*s*vs, cy - 0.05*s)]
    lv = [(cx - 0.02*s, cy + 0.10*s),
          (cx - 0.08*s*vs, cy + 0.07*s),
          (cx - 0.13*s*vs, cy + 0.02*s),
          (cx - 0.14*s*vs, cy - 0.05*s)]

    for pts in (rv, lv):
        ax.add_patch(PathPatch(branch_path(pts), edgecolor=red, linewidth=2.6,
                               facecolor="none", alpha=0.85, capstyle="round"))
        pts2 = [(x, y - 0.015*s) for (x,y) in pts]
        ax.add_patch(PathPatch(branch_path(pts2), edgecolor=blue, linewidth=2.2,
                               facecolor="none", alpha=0.75, capstyle="round"))

# ============================================================
# Mini alveolus (reduced + moved to never collide with formula box)
# ============================================================
def draw_alveolus(ax, origin=(0.07, 0.365), inflate=0.0):
    ox, oy = origin
    w = 0.135 * (0.95 + 0.25*inflate)
    h = 0.135 * (0.80 + 0.35*inflate)
    edge = "#7c3aed"
    fill = (0.93, 0.90, 0.99)

    e = Ellipse((ox + 0.075, oy), width=w, height=h,
                facecolor=fill, edgecolor=edge, linewidth=2.3, alpha=0.96, zorder=10)
    ax.add_patch(e); e.set_clip_on(True)

    # label ABOVE the alveolus, with white background for readability
    ax.text(
        ox, oy + 0.125, "Mini-alvéolo",
        fontsize=9.1, weight="bold", color=edge, zorder=30, clip_on=True,
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white", alpha=0.85, edgecolor="none")
    )

def draw_pl_thermometer(ax, x=0.84, y=0.34, w=0.10, h=0.48, pl_value=6.0):
    pl_min, pl_max = 4.0, 10.0
    plv = float(np.clip(pl_value, pl_min, pl_max))
    frac = (plv - pl_min) / (pl_max - pl_min)

    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=2.0, edgecolor="#111827", alpha=0.85))

    def y_of(pl):
        return y + h * (pl - pl_min) / (pl_max - pl_min)

    y_low_top  = y_of(5.5)
    y_phys_top = y_of(8.0)

    ax.add_patch(Rectangle((x, y), w, y_low_top - y, facecolor="#e5e7eb", edgecolor="none", alpha=0.95))
    ax.add_patch(Rectangle((x, y_low_top), w, y_phys_top - y_low_top, facecolor="#bbf7d0", edgecolor="none", alpha=0.95))
    ax.add_patch(Rectangle((x, y_phys_top), w, y + h - y_phys_top, facecolor="#fde68a", edgecolor="none", alpha=0.95))

    ax.add_patch(Rectangle((x, y), w, h*frac, facecolor="#7c3aed", edgecolor="none", alpha=0.55))
    ax.plot([x-0.02, x+w+0.02], [y + h*frac, y + h*frac], color="#111827", lw=2.0, clip_on=True)

    ax.text(x + w/2, y + h + 0.02, "PL", ha="center", fontsize=10, weight="bold", color="#111827")
    ax.text(x + w/2, y - 0.045, f"{pl_value:.1f}", ha="center", fontsize=10, weight="bold", color="#7c3aed")

def gradient_semaphore(ax, palv_value, base_y=0.235):
    tol = 0.10
    if palv_value < -tol:
        title = "Palv < Patm  →  ar entra"
        color = "#16a34a"
        box = "#dcfce7"
    elif palv_value > tol:
        title = "Palv > Patm  →  ar sai"
        color = "#16a34a"
        box = "#dcfce7"
    else:
        title = "Palv = Patm  →  fluxo = 0"
        color = "#6b7280"
        box = "#f3f4f6"

    ax.text(0.04, base_y + 0.06, "Semáforo do gradiente:", fontsize=10, weight="bold",
            color="#111827", clip_on=True)
    ax.text(0.04, base_y, title, fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=box, edgecolor=box, alpha=0.95),
            color=color, clip_on=True)

# ============================================================
# RENDER
# ============================================================
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "24"]
)

HIST = 12.0
total_frames = int(DURATION_S * FPS)

for i in range(total_frames):
    t = i / FPS
    tau = t % T_CYCLE
    ph, _ = phase_in_cycle(tau)

    ppl_now = ppl_of_tau(tau)
    palv_now = palv_of_tau(tau)
    pl_now = pl_of_tau(tau)
    flow_now = flow_of_tau(tau)
    vol_now = volume_of_tau(tau)

    th = make_history(t, HIST, FPS)
    tau_h = np.array([tt % T_CYCLE for tt in th])

    ppl_h  = np.array([ppl_of_tau(ta) for ta in tau_h])
    palv_h = np.array([palv_of_tau(ta) for ta in tau_h])
    pl_h   = np.array([pl_of_tau(ta) for ta in tau_h])
    flow_h = np.array([flow_of_tau(ta) for ta in tau_h]) * 60.0  # L/min
    vol_h  = np.array([volume_of_tau(ta) for ta in tau_h])

    # cycle boundaries (shading)
    t_cycle_start = t - (t % T_CYCLE)
    a = t_cycle_start + T_HOLD_EE
    b = a + T_INSP
    c = b + T_HOLD_EI
    d = c + T_EXP
    e = d + T_HOLD_EE2

    fig.clf()
    gs = fig.add_gridspec(
        3, 3,
        left=0.04, right=0.985, top=0.92, bottom=0.06,
        wspace=0.28, hspace=0.42,
        width_ratios=[1.28, 1.55, 1.10],
        height_ratios=[1.45, 1.45, 0.78]
    )

    ax_anim = fig.add_subplot(gs[:, 0])
    ax_p    = fig.add_subplot(gs[0:2, 1])
    ax_f    = fig.add_subplot(gs[2, 1])
    ax_txt  = fig.add_subplot(gs[0:2, 2])
    ax_v    = fig.add_subplot(gs[2, 2])

    # =========================
    # (A) Pulmão + diafragma
    # =========================
    ax_anim.set_title("Pulmões (anatómico) + Diafragma", fontsize=11.5, weight="bold")
    ax_anim.set_xlim(0, 1)
    ax_anim.set_ylim(0, 1)
    ax_anim.axis("off")

    thorax = Rectangle((0.06, 0.12), 0.74, 0.78, fill=False, lw=3,
                       edgecolor="#111827", alpha=0.70)
    ax_anim.add_patch(thorax)

    ax_anim.text(
        0.43, 0.875, "Caixa torácica",
        ha="center", va="center",
        fontsize=9.2, color="#111827", zorder=60,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
        clip_on=True
    )

    # Inflate tied to volume (0..VT_TARGET)
    pl_norm = float(np.clip(vol_now / max(VT_TARGET, 1e-6), 0.0, 1.0))
    draw_lungs_anatomic(ax_anim, center=(0.43, 0.64), scale=1.0, inflate=pl_norm)

    # airflow arrow (bigger when |flow| bigger)
    arrow_mag = float(np.clip(abs(flow_now) / 0.6, 0.0, 1.0))
    if flow_now > 1e-6:
        ax_anim.annotate("", xy=(0.43, 0.64), xytext=(0.88, 0.64),
                         arrowprops=dict(arrowstyle="->", lw=3 + 4*arrow_mag, color="#dc2626"))
        ax_anim.text(0.90, 0.72, "Ar entra", fontsize=9, color="#dc2626", ha="center",
                     bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.80),
                     clip_on=True)
    elif flow_now < -1e-6:
        ax_anim.annotate("", xy=(0.88, 0.64), xytext=(0.43, 0.64),
                         arrowprops=dict(arrowstyle="->", lw=3 + 4*arrow_mag, color="#16a34a"))
        ax_anim.text(0.90, 0.72, "Ar sai", fontsize=9, color="#16a34a", ha="center",
                     bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.80),
                     clip_on=True)
    else:
        ax_anim.text(0.90, 0.71, "Fluxo = 0", fontsize=9, color="#6b7280", ha="center",
                     bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.80),
                     clip_on=True)

    # diaphragm motion tied to Pmus
    pm_now = pmus_of_tau(tau)
    dia_norm = float(np.clip(pm_now / max(PMUS_PEAK, 1e-6), 0.0, 1.0))
    dia_y = 0.22 - 0.10 * dia_norm

    xs = np.linspace(0.10, 0.78, 240)
    arch = dia_y + 0.06 * np.sin(np.pi * (xs - 0.10) / (0.78 - 0.10))
    ax_anim.plot(xs, arch, lw=7, color="#111827", clip_on=True)

    ax_anim.text(
        0.82, dia_y + 0.02, "Diafragma",
        fontsize=9, color="#111827",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
        clip_on=True
    )

    phase_badge = {
        "hold_ee":  "CRF (pausa)",
        "insp":     "INSPIRAÇÃO",
        "hold_ei":  "Fim INSP (pausa)",
        "exp":      "EXPIRAÇÃO",
        "hold_ee2": "CRF (pausa)",
    }
    ph_label = phase_badge.get(ph, "")
    ax_anim.text(
        0.06, 0.05,
        f"{ph_label} — Ppl={ppl_now:.1f} | Palv={palv_now:.1f}",
        fontsize=8.7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="#e5e7eb"),
        clip_on=True
    )

    # =========================
    # (B) Pressões
    # =========================
    ax_p.set_title("Pressões (cmH₂O)", fontsize=11.5, weight="bold")
    ax_p.plot(th, ppl_h,  lw=3.1, color="#111827", label="Ppl")
    ax_p.plot(th, palv_h, lw=3.1, color="#2563eb", label="Palv")
    ax_p.plot(th, pl_h,   lw=3.1, color="#7c3aed", label="PL")
    ax_p.axhline(0, color="#9ca3af", lw=1.2)
    ax_p.set_ylim(-10, +10)
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax_p.set_xlabel("Tempo (s)")
    ax_p.set_ylabel("cmH₂O")

    ax_p.scatter([t], [ppl_now], s=52, color="#111827", zorder=6)
    ax_p.scatter([t], [palv_now], s=52, color="#2563eb", zorder=6)
    ax_p.scatter([t], [pl_now], s=52, color="#7c3aed", zorder=6)

    ax_p.axvspan(t_cycle_start, a, color="#f3f4f6", alpha=0.20)
    ax_p.axvspan(a, b, color="#fecaca", alpha=0.14)
    ax_p.axvspan(b, c, color="#f3f4f6", alpha=0.20)
    ax_p.axvspan(c, d, color="#bbf7d0", alpha=0.10)
    ax_p.axvspan(d, e, color="#f3f4f6", alpha=0.20)

    # =========================
    # (C) Fluxo (agora com aceleração inicial + desaceleração)
    # =========================
    ax_f.set_title("Fluxo (L/min)", fontsize=10.8, weight="bold")
    ax_f.plot(th, flow_h, lw=2.6, color="#dc2626", label="Fluxo")
    ax_f.axhline(0, color="#9ca3af", lw=1.1)
    ax_f.set_ylim(-60, 60)
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(loc="upper right", fontsize=8.8, frameon=True)
    ax_f.set_xlabel("Tempo (s)")
    ax_f.set_ylabel("L/min")

    ax_f.axvspan(t_cycle_start, a, color="#f3f4f6", alpha=0.20)
    ax_f.axvspan(a, b, color="#fecaca", alpha=0.14)
    ax_f.axvspan(b, c, color="#f3f4f6", alpha=0.20)
    ax_f.axvspan(c, d, color="#bbf7d0", alpha=0.10)
    ax_f.axvspan(d, e, color="#f3f4f6", alpha=0.20)

    # =========================
    # (D) Painel direito (sem sobreposições)
    # =========================
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")

    ax_txt.text(0.04, 0.95, "Leituras (agora)", fontsize=13.5, weight="bold",
                color="#111827", clip_on=True)

    ax_txt.text(0.04, 0.88, f"Ppl  = {ppl_now:.1f} cmH₂O", fontsize=12.2, weight="bold",
                color="#111827", clip_on=True)
    ax_txt.text(0.04, 0.82, f"Palv = {palv_now:.1f} cmH₂O", fontsize=12.2, weight="bold",
                color="#2563eb", clip_on=True)
    ax_txt.text(0.04, 0.76, f"PL   = {pl_now:.1f} cmH₂O", fontsize=12.2, weight="bold",
                color="#7c3aed", clip_on=True)

    ax_txt.text(0.04, 0.70, f"Fluxo = {flow_now*60:.0f} L/min", fontsize=12.2, weight="bold",
                color=("#dc2626" if flow_now > 0 else "#16a34a" if flow_now < 0 else "#6b7280"),
                clip_on=True)

    ax_txt.text(0.04, 0.64, f"VT ≈ {vol_now*1000:.0f} mL", fontsize=12.2, weight="bold",
                color="#b91c1c", clip_on=True)

    # Formula box placed higher, guaranteed not to collide with alveolus label
    ax_txt.text(
        0.04, 0.600,
        "Pressão transpulmonar:\nPL = Palv − Ppl",
        fontsize=11.2, weight="bold", color="#6d28d9",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#ede9fe", edgecolor="#6d28d9", alpha=0.98),
        clip_on=True, zorder=40
    )

    draw_alveolus(ax_txt, origin=(0.07, 0.365), inflate=pl_norm)
    draw_pl_thermometer(ax_txt, x=0.84, y=0.34, w=0.10, h=0.48, pl_value=pl_now)

    gradient_semaphore(ax_txt, palv_now, base_y=0.235)

    ax_txt.text(
        0.04, 0.02,
        "Passo-a-passo (modelo R–C):\n"
        "Pmus↑ → Ppl↓ → Palv<0 → fluxo entra\n"
        "V↑ → V/C↑ → ΔP diminui → fluxo desacelera → 0\n"
        "Pmus↓ → recoil domina → Palv>0 → ar sai\n\n"
        "Humor: é elástico + resistência. Nada místico.",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.33", facecolor="#fff7ed", alpha=0.97, edgecolor="#fed7aa"),
        clip_on=True
    )

    # =========================
    # (E) Volume vs time
    # =========================
    ax_v.set_title("Volume vs Tempo (VT sobre CRF)", fontsize=10.8, weight="bold")
    ax_v.plot(th, vol_h, lw=2.6, color="#b91c1c", label="Volume (L)")
    ax_v.axhline(0, color="#9ca3af", lw=1.1)
    ax_v.set_ylim(-0.05, VT_TARGET*1.25)
    ax_v.grid(True, alpha=0.25)
    ax_v.set_xlabel("Tempo (s)")
    ax_v.set_ylabel("L")
    ax_v.legend(loc="upper right", fontsize=8.2, frameon=True)
    ax_v.scatter([t], [vol_now], s=40, color="#2563eb", zorder=6)

    ax_v.axvspan(t_cycle_start, a, color="#f3f4f6", alpha=0.20)
    ax_v.axvspan(a, b, color="#fecaca", alpha=0.14)
    ax_v.axvspan(b, c, color="#f3f4f6", alpha=0.20)
    ax_v.axvspan(c, d, color="#bbf7d0", alpha=0.10)
    ax_v.axvspan(d, e, color="#f3f4f6", alpha=0.20)

    fig.suptitle(
        "Fisiologia Respiratória 2 — Respiração Espontânea (Modelo R–C)",
        fontsize=15, weight="bold", y=0.985
    )

    plt.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
