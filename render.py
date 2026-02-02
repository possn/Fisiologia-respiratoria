import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ============================================================
# OUTPUT
# ============================================================
OUT = "Respiracao_Normal_Base_Aula_Internos_EOM_60s.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 15          # <- mais suave
DURATION_S = 60

# ============================================================
# Robust trapezoid integration (NumPy compat)
# ============================================================
def integrate_trapezoid(y, x):
    """
    NumPy 2.x: np.trapezoid exists; np.trapz may not.
    This wrapper works across versions.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    # Fallback: manual trapezoid
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return np.sum((y[:-1] + y[1:]) * 0.5 * dx)

# ============================================================
# LUNG VOLUMES (adulto típico, mL) — valores didácticos aproximados
# ============================================================
VT  = 500
IRV = 3000
ERV = 1100
RV  = 1200

FRC = ERV + RV
TLC = RV + ERV + VT + IRV

# ============================================================
# BREATH PHYSIO (normal quiet breathing)
# ============================================================
T_CYCLE = 5.0  # ~12 rpm
INSP = 2.0
EXP  = 3.0

# Pressões alvo (cmH2O)
PPL_BASE = -5.0
PPL_INSP = -8.0

PALV_BASE = 0.0
PALV_MIN  = -1.0
PALV_EXP  = +1.0

# Fluxo (L/s) — desacelerante (modelo didático)
FLOW_PEAK_INSP = 0.45
FLOW_PEAK_EXP  = -0.35

# ============================================================
# Equação do Movimento (didáctico)
# Paw(t) = R*Vdot + E*V + baseline
# baseline aqui é conceptual (no ventilador = PEEP)
# ============================================================
R_AW = 8.0               # cmH2O/(L/s) (didático)
C_RS = 0.060             # L/cmH2O     (60 mL/cmH2O)
E_RS = 1.0 / C_RS        # cmH2O/L

# ============================================================
# Visual scaling (common bars)
# ============================================================
BAR_MAX_CM = 15.0        # <- escala comum para R·V’ e E·V

# ============================================================
# Helpers
# ============================================================
def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def breath_phase(t):
    t0 = t % T_CYCLE
    if t0 < INSP:
        return t0, "insp"
    return t0 - INSP, "exp"

def smoothstep(x):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(x, 0, 1))

def volume_waveform(t):
    """VT(t): 0→VT na insp; VT→0 na exp."""
    tp, ph = breath_phase(t)
    if ph == "insp":
        x = tp / INSP
        return VT * smoothstep(x)
    else:
        tau = 0.9
        return VT * np.exp(-tp / tau)

def ppl_waveform(t):
    """Ppl: -5 → -8 na insp; volta a -5 na exp."""
    tp, ph = breath_phase(t)
    if ph == "insp":
        x = tp / INSP
        return PPL_BASE + (PPL_INSP - PPL_BASE) * smoothstep(x)
    else:
        x = tp / EXP
        return PPL_INSP + (PPL_BASE - PPL_INSP) * smoothstep(x)

def palv_waveform(t):
    """Palv: 0→-1 na insp; bossa expiratória +1→0 na exp."""
    tp, ph = breath_phase(t)
    if ph == "insp":
        x = tp / INSP
        return PALV_BASE + (PALV_MIN - PALV_BASE) * smoothstep(x)
    else:
        x = tp / EXP
        return PALV_EXP * np.exp(-3.0*x)

def flow_waveform(t):
    """Fluxo desacelerante: + na insp, − na exp."""
    tp, ph = breath_phase(t)
    if ph == "insp":
        x = tp / INSP
        return FLOW_PEAK_INSP * np.exp(-2.0*x)
    else:
        x = tp / EXP
        return FLOW_PEAK_EXP * np.exp(-2.2*x)

# ============================================================
# Render params
# ============================================================
HIST_SEC = 12.0
ZOOM_SEC = 6.0

def make_history(t, fps, window_sec):
    t_start = max(0.0, t - window_sec)
    n = int(max(10, min((t - t_start) * fps + 1, window_sec * fps)))
    return np.linspace(t_start, t, n)

# ============================================================
# RENDER
# ============================================================
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "28"]
)

total_frames = int(DURATION_S * FPS)

for i in range(total_frames):
    t = i / FPS
    tp, ph = breath_phase(t)

    # Instant signals
    v_tidal_ml = volume_waveform(t)     # mL
    v_tidal_L  = v_tidal_ml / 1000.0    # L
    ppl  = ppl_waveform(t)
    palv = palv_waveform(t)
    flow_Ls = flow_waveform(t)          # L/s
    pl   = palv - ppl
    vol_abs = FRC + v_tidal_ml

    # Equation of motion terms (didactic)
    P_res = R_AW * flow_Ls              # cmH2O (±)
    P_el  = E_RS * v_tidal_L            # cmH2O (>=0)

    # Histories
    t_hist = make_history(t, FPS, HIST_SEC)
    t_zoom = make_history(t, FPS, ZOOM_SEC)

    ppl_hist  = np.array([ppl_waveform(tt) for tt in t_hist])
    palv_hist = np.array([palv_waveform(tt) for tt in t_hist])
    pl_hist   = palv_hist - ppl_hist
    flow_hist = np.array([flow_waveform(tt) for tt in t_hist])      # L/s
    flow_hist_lmin = flow_hist * 60.0

    # VT integration concept (current cycle, inspiratory positive area)
    t_cycle_start = t - (t % T_CYCLE)
    t_cycle_grid = np.linspace(t_cycle_start, t, max(2, int((t - t_cycle_start) * FPS) + 1))
    flow_cycle = np.array([flow_waveform(tt) for tt in t_cycle_grid])  # L/s
    flow_pos = np.clip(flow_cycle, 0, None)
    vt_integrated_ml = integrate_trapezoid(flow_pos, t_cycle_grid) * 1000.0  # L->mL

    # Zoom
    ppl_zoom  = np.array([ppl_waveform(tt) for tt in t_zoom])
    palv_zoom = np.array([palv_waveform(tt) for tt in t_zoom])
    pl_zoom   = palv_zoom - ppl_zoom

    # ============================================================
    # LAYOUT
    # ============================================================
    fig.clf()
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.05, 1.05, 1.10, 1.00],
        height_ratios=[1.0, 1.0]
    )

    ax_volumes = fig.add_subplot(gs[:, 0])
    ax_mech    = fig.add_subplot(gs[0, 1])
    ax_traces  = fig.add_subplot(gs[1, 1])
    ax_integr  = fig.add_subplot(gs[:, 2])
    ax_zoom    = fig.add_subplot(gs[:, 3])

    # ------------------------------------------------------------
    # (A) Volumes
    # ------------------------------------------------------------
    ax_volumes.set_title("Volumes Pulmonares — VT a oscilar sobre a FRC", fontsize=12, weight="bold")
    x0 = 0.5
    width = 0.35

    def rect(y0, h, label, color, alpha=0.9):
        ax_volumes.add_patch(plt.Rectangle((x0 - width/2, y0), width, h, color=color, alpha=alpha))
        ax_volumes.text(x0, y0 + h/2, label, ha="center", va="center",
                        fontsize=10, color="white", weight="bold")

    rect(0, RV,  "RV\nVolume Residual",            "#374151", 0.95)
    rect(RV, ERV, "ERV\nReserva Expiratória",      "#6b7280", 0.95)
    rect(RV + ERV, VT, "VT\nVolume Corrente",      "#2563eb", 0.75)
    rect(RV + ERV + VT, IRV, "IRV\nReserva Insp.", "#0f766e", 0.90)

    ax_volumes.axhline(FRC, color="#111827", lw=2, alpha=0.65)
    ax_volumes.text(0.05, FRC, f" FRC = ERV+RV = {FRC} mL", va="bottom", fontsize=10)

    ax_volumes.axhline(TLC, color="#111827", lw=2, alpha=0.65)
    ax_volumes.text(0.05, TLC, f" TLC = {TLC} mL", va="bottom", fontsize=10)

    ax_volumes.plot([x0 - width/2, x0 + width/2], [vol_abs, vol_abs], color="#dc2626", lw=4)
    ax_volumes.text(x0 + 0.32, vol_abs, f"V(t)={int(vol_abs)} mL", color="#dc2626", fontsize=10, va="center")

    ax_volumes.set_xlim(0, 1)
    ax_volumes.set_ylim(0, TLC * 1.05)
    ax_volumes.set_xticks([])
    ax_volumes.set_ylabel("Volume (mL)")

    ax_volumes.text(
        0.05, 0.98 * TLC,
        "Definições:\nFRC = ERV + RV\nTLC = RV + ERV + VT + IRV",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#e5e7eb")
    )

    # ------------------------------------------------------------
    # (B) Mecânica + Equação do Movimento
    # ------------------------------------------------------------
    ax_mech.set_title("Mecânica + Equação do Movimento (ligada ao que vês)", fontsize=11, weight="bold")
    ax_mech.set_xlim(0, 1)
    ax_mech.set_ylim(0, 1)
    ax_mech.axis("off")

    lung_r = 0.18 + 0.10*(v_tidal_ml / VT)
    ax_mech.add_patch(plt.Circle((0.28, 0.62), lung_r, fill=False, lw=6, color="#2563eb"))
    ax_mech.text(0.28, 0.62, "Pulmão", ha="center", va="center", fontsize=10, weight="bold")

    dia_y = 0.25 - 0.10*(v_tidal_ml / VT)
    xs = np.linspace(0.08, 0.52, 200)
    arch = dia_y + 0.08*np.sin(np.pi*(xs-0.08)/0.44)
    ax_mech.plot(xs, arch, lw=6, color="#111827")
    ax_mech.text(0.54, dia_y + 0.02, "Diafragma", fontsize=9, va="center")

    if ph == "insp":
        ax_mech.annotate("", xy=(0.60, dia_y-0.08), xytext=(0.60, dia_y+0.08),
                         arrowprops=dict(arrowstyle="->", lw=3, color="#dc2626"))
        phase_color = "#dc2626"
        phase_word = "INSP"
    else:
        ax_mech.annotate("", xy=(0.60, dia_y+0.08), xytext=(0.60, dia_y-0.08),
                         arrowprops=dict(arrowstyle="->", lw=3, color="#16a34a"))
        phase_color = "#16a34a"
        phase_word = "EXP"

    ax_mech.text(0.70, 0.86, f"Fase: {phase_word}", fontsize=11, weight="bold", color=phase_color)
    ax_mech.text(0.70, 0.78, f"Ppl ≈ {ppl:.1f} cmH₂O", fontsize=11, weight="bold", color="#111827")
    ax_mech.text(0.70, 0.70, f"Palv ≈ {palv:.1f} cmH₂O", fontsize=11, weight="bold", color="#2563eb")
    ax_mech.text(0.70, 0.62, f"PL=Palv−Ppl ≈ {pl:.1f} cmH₂O", fontsize=11, weight="bold", color="#7c3aed")
    ax_mech.text(0.70, 0.54, f"Fluxo ≈ {flow_Ls*60:.0f} L/min", fontsize=11, weight="bold", color="#dc2626")

    # Highlight terms depending on flow/volume
    res_level = min(abs(flow_Ls) / max(abs(FLOW_PEAK_INSP), 1e-6), 1.0)
    el_level  = min(v_tidal_ml / VT, 1.0)

    res_alpha = 0.25 + 0.60*res_level
    el_alpha  = 0.25 + 0.60*el_level

    ax_mech.text(
        0.05, 0.46,
        "Equação do movimento:\n"
        r"$P_{aw}(t)=R\cdot\dot V(t) + E\cdot V(t) + \mathrm{baseline}$",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, edgecolor="#e5e7eb")
    )

    ax_mech.text(
        0.06, 0.30,
        r"$R\cdot\dot V$ (resistivo)",
        fontsize=10, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fecaca", alpha=res_alpha, edgecolor="#fecaca")
    )
    ax_mech.text(
        0.06, 0.22,
        r"$E\cdot V$ (elástico)",
        fontsize=10, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ddd6fe", alpha=el_alpha, edgecolor="#ddd6fe")
    )
    ax_mech.text(
        0.06, 0.14,
        "baseline (no ventilador = PEEP)",
        fontsize=10, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#e5e7eb", alpha=0.85, edgecolor="#e5e7eb")
    )

    # Bars on common scale 0–15 cmH2O
    P_res_mag = min(abs(P_res), BAR_MAX_CM)
    P_el_mag  = min(P_el,       BAR_MAX_CM)

    # Bar frames
    ax_mech.add_patch(plt.Rectangle((0.78, 0.10), 0.06, 0.30, fill=False, lw=2, color="#9ca3af"))
    ax_mech.add_patch(plt.Rectangle((0.88, 0.10), 0.06, 0.30, fill=False, lw=2, color="#9ca3af"))

    # Fill bars
    ax_mech.add_patch(plt.Rectangle((0.78, 0.10), 0.06, 0.30*(P_res_mag/BAR_MAX_CM), color="#ef4444", alpha=0.85))
    ax_mech.add_patch(plt.Rectangle((0.88, 0.10), 0.06, 0.30*(P_el_mag/BAR_MAX_CM),  color="#7c3aed", alpha=0.70))

    ax_mech.text(0.81, 0.42, "R·V’", ha="center", fontsize=9, weight="bold", color="#b91c1c")
    ax_mech.text(0.91, 0.42, "E·V",  ha="center", fontsize=9, weight="bold", color="#6d28d9")
    ax_mech.text(0.81, 0.06, f"{abs(P_res):.1f}", ha="center", fontsize=9, color="#b91c1c")
    ax_mech.text(0.91, 0.06, f"{P_el:.1f}",       ha="center", fontsize=9, color="#6d28d9")
    ax_mech.text(0.845, 0.02, "cmH₂O", ha="center", fontsize=8, color="#6b7280")

    # ------------------------------------------------------------
    # (C) Traçados
    # ------------------------------------------------------------
    ax_traces.set_title("Traçados: Ppl, Palv, PL e Fluxo", fontsize=11, weight="bold")
    ax_traces.plot(t_hist, ppl_hist,  lw=2.4, color="#111827", label="Ppl")
    ax_traces.plot(t_hist, palv_hist, lw=2.4, color="#2563eb", label="Palv")
    ax_traces.plot(t_hist, pl_hist,   lw=2.4, color="#7c3aed", label="PL")

    ax_traces.axhline(0, color="#9ca3af", lw=1.2)
    ax_traces.set_ylabel("Pressão (cmH₂O)")
    ax_traces.set_ylim(-10, +10)
    ax_traces.grid(True, alpha=0.25)

    ax2 = ax_traces.twinx()
    ax2.plot(t_hist, flow_hist_lmin, lw=2.4, color="#dc2626", label="Fluxo (L/min)")
    ax2.set_ylabel("Fluxo (L/min)")
    ax2.set_ylim(-60, 60)

    l1, lab1 = ax_traces.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax_traces.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8, frameon=True)
    ax_traces.set_xlabel("Tempo (s)")

    # Shade phases in last cycle
    ax_traces.axvspan(t_cycle_start, t_cycle_start + INSP, color="#fecaca", alpha=0.18)
    ax_traces.axvspan(t_cycle_start + INSP, t_cycle_start + T_CYCLE, color="#bbf7d0", alpha=0.10)

    # ------------------------------------------------------------
    # (D) Integração: VT = ∫ Fluxo dt
    # ------------------------------------------------------------
    ax_integr.set_title("Integração: VT ≈ ∫ Fluxo(t) dt (conceito)", fontsize=12, weight="bold")
    ax_integr.grid(True, alpha=0.25)

    t_c = np.linspace(t_cycle_start, t_cycle_start + T_CYCLE, int(T_CYCLE * FPS))
    flow_c = np.array([flow_waveform(tt) for tt in t_c]) * 60.0

    ax_integr.plot(t_c, flow_c, lw=2.6, color="#dc2626", label="Fluxo (L/min)")
    ax_integr.axhline(0, color="#9ca3af", lw=1.2)

    t_fill = t_c[t_c <= t]
    flow_fill = np.array([flow_waveform(tt) for tt in t_fill]) * 60.0
    flow_fill_pos = np.clip(flow_fill, 0, None)
    ax_integr.fill_between(t_fill, 0, flow_fill_pos, color="#fecaca", alpha=0.45, label="Área inspiratória")

    ax_integr.set_ylabel("Fluxo (L/min)")
    ax_integr.set_xlabel("Tempo (s)")
    ax_integr.set_ylim(-60, 60)
    ax_integr.legend(loc="upper right", fontsize=9, frameon=True)

    ax_integr.text(
        0.02, 0.05,
        f"VT acumulado ≈ {vt_integrated_ml:.0f} mL\n(área sob o fluxo inspiratório)",
        transform=ax_integr.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#e5e7eb")
    )

    ax_integr.text(
        0.02, 0.90,
        "Pressão necessária = atrito + distensão + baseline",
        transform=ax_integr.transAxes,
        fontsize=10, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#e5e7eb")
    )

    # ------------------------------------------------------------
    # (E) Zoom
    # ------------------------------------------------------------
    ax_zoom.set_title("Zoom: amplitudes pequenas (cmH₂O) → grandes efeitos", fontsize=12, weight="bold")
    ax_zoom.plot(t_zoom, ppl_zoom,  lw=2.8, color="#111827", label="Ppl")
    ax_zoom.plot(t_zoom, palv_zoom, lw=2.8, color="#2563eb", label="Palv")
    ax_zoom.plot(t_zoom, pl_zoom,   lw=2.8, color="#7c3aed", label="PL")

    ax_zoom.axhline(0, color="#9ca3af", lw=1.2)
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.set_ylim(-10, +10)
    ax_zoom.set_ylabel("Pressão (cmH₂O)")
    ax_zoom.set_xlabel("Tempo (s)")

    ax_zoom.axhline(PPL_BASE, color="#111827", lw=1.0, alpha=0.25)
    ax_zoom.axhline(PPL_INSP, color="#111827", lw=1.0, alpha=0.25)
    ax_zoom.axhline(PALV_BASE, color="#2563eb", lw=1.0, alpha=0.25)
    ax_zoom.axhline(PALV_MIN,  color="#2563eb", lw=1.0, alpha=0.25)

    ax_zoom.annotate("ΔPpl≈3\n(−5→−8)", xy=(t, PPL_INSP), xytext=(t_zoom[0], -9),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#111827"),
                     fontsize=10, color="#111827")

    ax_zoom.annotate("ΔPalv≈1\n(0→−1)", xy=(t, PALV_MIN), xytext=(t_zoom[0], 2.5),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#2563eb"),
                     fontsize=10, color="#2563eb")

    ax_zoom.legend(loc="upper right", fontsize=9, frameon=True)

    # ============================================================
    # Global title
    # ============================================================
    fig.suptitle(
        f"Respiração Normal — Base para Internos | Fase: {ph.upper()} | Equação do Movimento ligada às curvas",
        fontsize=13, weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
