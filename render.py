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
OUT = "Fisiologia_Respiratoria_2_AulaBasica_Master_v3_1_CURVAS_OK.mp4"

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
# TARGETS (cmH2O)
# ============================================================
PPL_EE  = -5.0
PPL_EI  = -7.0

PALV_MIN_INSP = -1.0
PALV_EXP_PEAK = +1.0

# Flow peaks (L/s)
FLOW_PEAK_INSP = 0.45
FLOW_PEAK_EXP  = -0.35

# ============================================================
# BASIC VT (L)
# ============================================================
VT_L = 0.5  # 500 mL

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

def ppl_of_tau(tau: float) -> float:
    ph, x = phase_in_cycle(tau)
    if ph in ("hold_ee", "hold_ee2"):
        return PPL_EE
    if ph == "insp":
        return PPL_EE + (PPL_EI - PPL_EE) * smoothstep(x)
    if ph == "hold_ei":
        return PPL_EI
    return PPL_EI + (PPL_EE - PPL_EI) * smoothstep(x)

# ✅ CORRIGIDO: Palv com “onda suave” (cai durante insp e volta a 0 no fim insp)
def palv_of_tau(tau: float) -> float:
    ph, x = phase_in_cycle(tau)
    if ph in ("hold_ee", "hold_ei", "hold_ee2"):
        return 0.0
    if ph == "insp":
        # 0 -> -1 -> 0 (clássico)
        return PALV_MIN_INSP * np.sin(np.pi * x)
    # exp: 0 -> +1 -> 0 (clássico)
    return PALV_EXP_PEAK * np.sin(np.pi * x)

def flow_of_tau(tau: float) -> float:
    ph, x = phase_in_cycle(tau)
    if ph in ("hold_ee", "hold_ee2", "hold_ei"):
        return 0.0
    if ph == "insp":
        # desacelerante
        return FLOW_PEAK_INSP * np.exp(-2.0 * x)
    # exp desacelerante
    return FLOW_PEAK_EXP * np.exp(-2.2 * x)

def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def trapezoid_integral(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * dx))

# ============================================================
# Precompute volume(tau) from flow(tau) for one cycle
# ============================================================
N_PRE = 2400
tau_grid = np.linspace(0.0, T_CYCLE, N_PRE)
flow_grid = np.array([flow_of_tau(tau) for tau in tau_grid])  # L/s

vol_raw = np.zeros_like(tau_grid)
for k in range(1, len(tau_grid)):
    vol_raw[k] = trapezoid_integral(flow_grid[:k+1], tau_grid[:k+1])

vol_raw = vol_raw - np.min(vol_raw)
peak = np.max(vol_raw)
scale = VT_L / peak if peak > 1e-9 else 1.0
vol_grid = vol_raw * scale  # L

def volume_of_tau(tau: float) -> float:
    tau = tau % T_CYCLE
    return float(np.interp(tau, tau_grid, vol_grid))

# ============================================================
# Drawing: anatomical lungs + vessels
# ============================================================
def _mix(c1, c2, a):
    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)
    return tuple(np.clip((1-a)*c1 + a*c2, 0, 1))

def draw_lungs_anatomic(ax, center=(0.43, 0.64), scale=1.0, inflate=0.0):
    cx, cy = center
    s = scale * (1.0 + 0.10 * inflate)

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

    # fissures
    for side in [-1, 1]:
        x0 = cx + side*0.12*s
        y0 = cy + 0.06*s
        xs = np.linspace(x0 - 0.06*s*side, x0 + 0.02*s*side, 120)
        ys = y0 - 0.05*s*np.sin(np.linspace(0, np.pi, 120))
        ax.plot(xs, ys, color=_mix(edge, (0,0,0), 0.15), lw=1.1, alpha=0.32, clip_on=True)

    notch = Ellipse((cx - 0.08*s, cy - 0.01*s), width=0.11*s, height=0.16*s, angle=25,
                    facecolor=ax.get_facecolor(), edgecolor=ax.get_facecolor(), linewidth=0)
    ax.add_patch(notch); notch.set_clip_on(True)

    # trachea inside box
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

    vs = (1.0 + 0.10*inflate)
    red = (0.86, 0.18, 0.18)
    blue = (0.12, 0.45, 0.78)

    def branch_path(points):
        verts, codes = [], []
        for j, (x,y) in enumerate(points):
            verts.append((x,y))
            codes.append(Path.MOVETO if j==0 else Path.LINETO)
        return Path(verts, codes)

    rv_base = [(cx + 0.02*s, cy + 0.10*s),
               (cx + 0.08*s*vs, cy + 0.07*s),
               (cx + 0.13*s*vs, cy + 0.02*s),
               (cx + 0.14*s*vs, cy - 0.05*s)]
    lv_base = [(cx - 0.02*s, cy + 0.10*s),
               (cx - 0.08*s*vs, cy + 0.07*s),
               (cx - 0.13*s*vs, cy + 0.02*s),
               (cx - 0.14*s*vs, cy - 0.05*s)]

    for pts in [rv_base, lv_base]:
        p1 = PathPatch(branch_path(pts), edgecolor=red, linewidth=2.6,
                       facecolor="none", alpha=0.85, capstyle="round")
        ax.add_patch(p1); p1.set_clip_on(True)

        pts2 = [(x, y - 0.015*s) for (x,y) in pts]
        p2 = PathPatch(branch_path(pts2), edgecolor=blue, linewidth=2.2,
                       facecolor="none", alpha=0.75, capstyle="round")
        ax.add_patch(p2); p2.set_clip_on(True)

# ============================================================
# Mini alveolus (label with bbox + zorder)
# ============================================================
def draw_alveolus(ax, origin=(0.04, 0.41), inflate=0.0):
    ox, oy = origin
    w = 0.20 * (0.85 + 0.30*inflate)
    h = 0.20 * (0.60 + 0.55*inflate)
    edge = "#7c3aed"
    fill = (0.93, 0.90, 0.99)
    e = Ellipse((ox + 0.10, oy), width=w, height=h,
                facecolor=fill, edgecolor=edge, linewidth=2.8, alpha=0.95)
    ax.add_patch(e); e.set_clip_on(True)
    ax.text(
        ox, oy + 0.15, "Mini-alvéolo",
        fontsize=9.2, weight="bold", color=edge, clip_on=True, zorder=25,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none")
    )

def draw_pl_thermometer(ax, x=0.84, y=0.34, w=0.10, h=0.48, pl_value=6.0):
    pl_min = 4.0
    pl_max = 9.0
    plv = float(np.clip(pl_value, pl_min, pl_max))
    frac = (plv - pl_min) / (pl_max - pl_min)

    frame = Rectangle((x, y), w, h, fill=False, lw=2.0, edgecolor="#111827", alpha=0.85)
    ax.add_patch(frame); frame.set_clip_on(True)

    def y_of(pl):
        return y + h * (pl - pl_min) / (pl_max - pl_min)

    y_low_top  = y_of(5.5)
    y_phys_top = y_of(7.5)

    ax.add_patch(Rectangle((x, y), w, y_low_top - y, facecolor="#e5e7eb", edgecolor="none", alpha=0.95))
    ax.add_patch(Rectangle((x, y_low_top), w, y_phys_top - y_low_top, facecolor="#bbf7d0", edgecolor="none", alpha=0.95))
    ax.add_patch(Rectangle((x, y_phys_top), w, y + h - y_phys_top, facecolor="#fde68a", edgecolor="none", alpha=0.95))

    ax.add_patch(Rectangle((x, y), w, h*frac, facecolor="#7c3aed", edgecolor="none", alpha=0.55))
    ax.plot([x-0.02, x+w+0.02], [y + h*frac, y + h*frac], color="#111827", lw=2.0, clip_on=True)

    ax.text(x + w/2, y + h + 0.02, "PL", ha="center", fontsize=10, weight="bold", color="#111827", clip_on=True)
    ax.text(x + w/2, y - 0.045, f"{pl_value:.1f}", ha="center", fontsize=10, weight="bold", color="#7c3aed", clip_on=True)

def gradient_semaphore(ax, palv_value, base_y=0.25):
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

    ax.text(0.04, base_y + 0.05, "Semáforo do gradiente:", fontsize=10, weight="bold",
            color="#111827", clip_on=True)
    ax.text(0.04, base_y, title, fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.30", facecolor=box, edgecolor=box, alpha=0.95),
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

total_frames = int(DURATION_S * FPS)

# ✅ Em vez de histórico deslizante: mostramos sempre 1 ciclo fixo 0–7 s
T_VIEW = T_CYCLE
N_VIEW = 500
t_view = np.linspace(0.0, T_VIEW, N_VIEW)

for i in range(total_frames):
    t = i / FPS
    tau = t % T_CYCLE
    ph, _ = phase_in_cycle(tau)

    ppl_now = ppl_of_tau(tau)
    palv_now = palv_of_tau(tau)
    pl_now = palv_now - ppl_now
    flow_now = flow_of_tau(tau)
    vol_now = volume_of_tau(tau)

    # fixed-cycle traces
    ppl_c  = np.array([ppl_of_tau(tt) for tt in t_view])
    palv_c = np.array([palv_of_tau(tt) for tt in t_view])
    pl_c   = palv_c - ppl_c
    flow_c = np.array([flow_of_tau(tt) for tt in t_view]) * 60.0
    vol_c  = np.array([volume_of_tau(tt) for tt in t_view])

    # phase boundaries in cycle-time
    a = T_HOLD_EE
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
    # Lung panel
    # =========================
    ax_anim.set_title("Pulmões (anatómico) + Diafragma", fontsize=11.5, weight="bold")
    ax_anim.set_xlim(0, 1)
    ax_anim.set_ylim(0, 1)
    ax_anim.axis("off")

    thorax = Rectangle((0.06, 0.12), 0.74, 0.78, fill=False, lw=3,
                       edgecolor="#111827", alpha=0.70)
    ax_anim.add_patch(thorax)

    # ✅ Texto da caixa torácica SEM ficar por baixo da linha
    ax_anim.text(
        0.43, 0.905, "Caixa torácica",
        ha="center", va="center", fontsize=9, color="#111827",
        zorder=30, clip_on=True,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.75, edgecolor="none")
    )

    pl_min_ref = (0.0 - PPL_EE)  # 5
    pl_max_ref = (0.0 - PPL_EI)  # 7
    pl_norm = float(np.clip((pl_now - pl_min_ref) / max(pl_max_ref - pl_min_ref, 1e-6), 0.0, 1.0))

    draw_lungs_anatomic(ax_anim, center=(0.43, 0.64), scale=1.0, inflate=pl_norm)

    # airflow arrow + label with bbox
    arrow_mag = float(np.clip(abs(flow_now) / 0.45, 0.0, 1.0))
    if flow_now > 1e-6:
        ax_anim.annotate("", xy=(0.43, 0.64), xytext=(0.88, 0.64),
                         arrowprops=dict(arrowstyle="->", lw=3 + 4*arrow_mag, color="#dc2626"))
        ax_anim.text(0.88, 0.68, "Ar entra", fontsize=9, color="#dc2626", ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
                     zorder=30, clip_on=True)
    elif flow_now < -1e-6:
        ax_anim.annotate("", xy=(0.88, 0.64), xytext=(0.43, 0.64),
                         arrowprops=dict(arrowstyle="->", lw=3 + 4*arrow_mag, color="#16a34a"))
        ax_anim.text(0.88, 0.68, "Ar sai", fontsize=9, color="#16a34a", ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
                     zorder=30, clip_on=True)
    else:
        ax_anim.text(0.88, 0.66, "Fluxo = 0", fontsize=9, color="#6b7280", ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
                     zorder=30, clip_on=True)

    # diaphragm motion + label moved slightly up/right
    dia_norm = float(np.clip((PPL_EE - ppl_now) / (PPL_EE - PPL_EI), 0.0, 1.0))
    dia_y = 0.22 - 0.10 * dia_norm

    xs = np.linspace(0.10, 0.78, 240)
    arch = dia_y + 0.06 * np.sin(np.pi * (xs - 0.10) / (0.78 - 0.10))
    ax_anim.plot(xs, arch, lw=7, color="#111827", clip_on=True)

    ax_anim.text(
        0.79, dia_y + 0.06, "Diafragma",
        fontsize=9, color="#111827", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        zorder=30, clip_on=True
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
    # Pressures
    # =========================
    ax_p.set_title("Pressões (cmH₂O)", fontsize=11.5, weight="bold")
    ax_p.plot(t_view, ppl_c,  lw=3.2, color="#111827", label="Ppl")
    ax_p.plot(t_view, palv_c, lw=3.2, color="#2563eb", label="Palv")
    ax_p.plot(t_view, pl_c,   lw=3.2, color="#7c3aed", label="PL")
    ax_p.axhline(0, color="#9ca3af", lw=1.2)
    ax_p.set_xlim(0, T_CYCLE)
    ax_p.set_ylim(-10, +10)
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax_p.set_xlabel("Tempo (s)")
    ax_p.set_ylabel("cmH₂O")

    # shading phases
    ax_p.axvspan(0, a, color="#f3f4f6", alpha=0.20)
    ax_p.axvspan(a, b, color="#fecaca", alpha=0.14)
    ax_p.axvspan(b, c, color="#f3f4f6", alpha=0.20)
    ax_p.axvspan(c, d, color="#bbf7d0", alpha=0.10)
    ax_p.axvspan(d, e, color="#f3f4f6", alpha=0.20)

    ax_p.scatter([tau], [ppl_now], s=55, color="#111827", zorder=6)
    ax_p.scatter([tau], [palv_now], s=55, color="#2563eb", zorder=6)
    ax_p.scatter([tau], [pl_now], s=55, color="#7c3aed", zorder=6)

    # =========================
    # Flow
    # =========================
    ax_f.set_title("Fluxo (L/min)", fontsize=10.8, weight="bold")
    ax_f.plot(t_view, flow_c, lw=2.4, color="#dc2626", label="Fluxo")
    ax_f.axhline(0, color="#9ca3af", lw=1.1)
    ax_f.set_xlim(0, T_CYCLE)
    ax_f.set_ylim(-60, 60)
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(loc="upper right", fontsize=8.8, frameon=True)
    ax_f.set_xlabel("Tempo (s)")
    ax_f.set_ylabel("L/min")

    ax_f.axvspan(0, a, color="#f3f4f6", alpha=0.20)
    ax_f.axvspan(a, b, color="#fecaca", alpha=0.14)
    ax_f.axvspan(b, c, color="#f3f4f6", alpha=0.20)
    ax_f.axvspan(c, d, color="#bbf7d0", alpha=0.10)
    ax_f.axvspan(d, e, color="#f3f4f6", alpha=0.20)

    ax_f.scatter([tau], [flow_now*60], s=40, color="#111827", zorder=6)

    # =========================
    # Right panel
    # =========================
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")

    ax_txt.text(0.04, 0.95, "Leituras (agora)", fontsize=13.5, weight="bold",
                color="#111827", clip_on=True)

    ax_txt.text(0.04, 0.88, f"Ppl  = {ppl_now:.1f} cmH₂O", fontsize=12.4, weight="bold",
                color="#111827", clip_on=True)
    ax_txt.text(0.04, 0.82, f"Palv = {palv_now:.1f} cmH₂O", fontsize=12.4, weight="bold",
                color="#2563eb", clip_on=True)
    ax_txt.text(0.04, 0.76, f"PL   = {pl_now:.1f} cmH₂O", fontsize=12.4, weight="bold",
                color="#7c3aed", clip_on=True)

    ax_txt.text(0.04, 0.70, f"Fluxo = {flow_now*60:.0f} L/min", fontsize=12.4, weight="bold",
                color=("#dc2626" if flow_now > 0 else "#16a34a" if flow_now < 0 else "#6b7280"),
                clip_on=True)

    ax_txt.text(0.04, 0.64, f"VT ≈ {vol_now*1000:.0f} mL", fontsize=12.4, weight="bold",
                color="#b91c1c", clip_on=True)

    ax_txt.text(
        0.04, 0.54,
        "Pressão transpulmonar:\nPL = Palv − Ppl",
        fontsize=12.0, weight="bold", color="#6d28d9",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#ede9fe", edgecolor="#6d28d9", alpha=0.97),
        clip_on=True
    )

    # Mini-alvéolo desenhado depois (fica por cima)
    draw_alveolus(ax_txt, origin=(0.04, 0.41), inflate=pl_norm)
    draw_pl_thermometer(ax_txt, x=0.84, y=0.34, w=0.10, h=0.48, pl_value=pl_now)

    gradient_semaphore(ax_txt, palv_now, base_y=0.25)

    ax_txt.text(
        0.04, 0.02,
        "Passo-a-passo:\n"
        "1) CRF: Palv=0 → fluxo=0\n"
        "2) Insp: Ppl↓ → Palv↓ → ar entra\n"
        "3) Fim insp: Palv=0 → fluxo=0\n"
        "4) Exp: Palv↑ → ar sai\n"
        "5) CRF: equilíbrio\n\n"
        "Humor clínico: não é magia — é ΔP.",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.33", facecolor="#fff7ed", alpha=0.97, edgecolor="#fed7aa"),
        clip_on=True
    )

    # =========================
    # Volume vs time
    # =========================
    ax_v.set_title("Volume vs Tempo (VT sobre CRF)", fontsize=10.8, weight="bold")
    ax_v.plot(t_view, vol_c, lw=2.4, color="#b91c1c", label="Volume (L)")
    ax_v.axhline(0, color="#9ca3af", lw=1.1)
    ax_v.set_xlim(0, T_CYCLE)
    ax_v.set_ylim(-0.05, VT_L*1.25)
    ax_v.grid(True, alpha=0.25)
    ax_v.set_xlabel("Tempo (s)")
    ax_v.set_ylabel("L")
    ax_v.legend(loc="upper right", fontsize=8.2, frameon=True)
    ax_v.scatter([tau], [vol_now], s=42, color="#2563eb", zorder=6)

    ax_v.axvspan(0, a, color="#f3f4f6", alpha=0.20)
    ax_v.axvspan(a, b, color="#fecaca", alpha=0.14)
    ax_v.axvspan(b, c, color="#f3f4f6", alpha=0.20)
    ax_v.axvspan(c, d, color="#bbf7d0", alpha=0.10)
    ax_v.axvspan(d, e, color="#f3f4f6", alpha=0.20)

    fig.suptitle(
        "Fisiologia Respiratória 2 — Mecânica Básica da Respiração",
        fontsize=15, weight="bold", y=0.985
    )

    plt.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
