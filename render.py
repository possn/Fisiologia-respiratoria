import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ============================================================
# OUTPUT
# ============================================================
OUT = "Respiracao_Normal_Volumes_Pressoes_Fluxo_PL_60s.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 12
DURATION_S = 60
W, H = 1280, 720

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
PALV_EXP  = +1.0  # pequena pressão positiva expiratória (passiva)

# Fluxo (L/s) — desacelerante (modelo didático)
FLOW_PEAK_INSP = 0.45    # ~27 L/min
FLOW_PEAK_EXP  = -0.35   # ~-21 L/min

# ============================================================
# Helpers
# ============================================================
def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def breath_phase(t):
    """Retorna fase dentro do ciclo e se está em inspiração/expiração."""
    t0 = t % T_CYCLE
    if t0 < INSP:
        return t0, "insp"
    return t0 - INSP, "exp"

def smoothstep(x):
    """0→1 suave (cosine easing)."""
    return 0.5 - 0.5*np.cos(np.pi*np.clip(x, 0, 1))

def volume_waveform(t):
    """
    Volume "relativo" do VT (0→VT na insp; VT→0 na exp),
    com inspiração suave e expiração exponencial (passiva).
    """
    tp, ph = breath_phase(t)
    if ph == "insp":
        x = tp / INSP
        return VT * smoothstep(x)
    else:
        tau = 0.9
        return VT * np.exp(-tp / tau)

def ppl_waveform(t):
    """
    Ppl: -5 → -8 durante insp; volta para -5 na exp.
    """
    tp, ph = breath_phase(t)
    if ph == "insp":
        x = tp / INSP
        return PPL_BASE + (PPL_INSP - PPL_BASE) * smoothstep(x)
    else:
        x = tp / EXP
        return PPL_INSP + (PPL_BASE - PPL_INSP) * smoothstep(x)

def palv_waveform(t):
    """
    Palv: 0 → -1 na inspiração; na expiração sobe ligeiro +1 e volta a 0.
    """
    tp, ph = breath_phase(t)
    if ph == "insp":
        x = tp / INSP
        return PALV_BASE + (PALV_MIN - PALV_BASE) * smoothstep(x)
    else:
        x = tp / EXP
        return PALV_EXP * np.exp(-3.0*x)

def flow_waveform(t):
    """
    Fluxo desacelerante:
    - inspiração: começa alto e desacelera até 0
    - expiração: começa negativo e desacelera até 0
    """
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
HIST_SEC = 12.0  # mostrar ~2 ciclos
ZOOM_SEC = 6.0   # zoom último 6s (didático)

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

    # Instant signals
    v_tidal = volume_waveform(t)         # 0..VT
    ppl = ppl_waveform(t)                # -5..-8
    palv = palv_waveform(t)              # 0..-1..+1
    flow = flow_waveform(t)              # L/s
    pl = palv - ppl                      # transpulmonary

    vol_abs = FRC + v_tidal              # absolute lung volume around FRC

    # Histories
    t_hist = make_history(t, FPS, HIST_SEC)
    t_zoom = make_history(t, FPS, ZOOM_SEC)

    ppl_hist  = np.array([ppl_waveform(tt) for tt in t_hist])
    palv_hist = np.array([palv_waveform(tt) for tt in t_hist])
    pl_hist   = palv_hist - ppl_hist
    flow_hist = np.array([flow_waveform(tt) for tt in t_hist]) * 60.0  # L/min

    ppl_zoom  = np.array([ppl_waveform(tt) for tt in t_zoom])
    palv_zoom = np.array([palv_waveform(tt) for tt in t_zoom])
    pl_zoom   = palv_zoom - ppl_zoom

    # =======================
    # LAYOUT
    # =======================
    fig.clf()
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.05, 1.10, 1.10],
        height_ratios=[1.0, 1.0]
    )

    ax_volumes = fig.add_subplot(gs[:, 0])      # left full height
    ax_mech    = fig.add_subplot(gs[0, 1])      # top middle
    ax_traces  = fig.add_subplot(gs[1, 1])      # bottom middle
    ax_zoom    = fig.add_subplot(gs[:, 2])      # right full height zoom panel

    # ------------------------------------------------------------
    # (A) Volumes pulmonares (diagrama empilhado)
    # ------------------------------------------------------------
    ax_volumes.set_title("Volumes Pulmonares — VT a oscilar sobre a FRC", fontsize=12, weight="bold")

    x0 = 0.5
    width = 0.35

    def rect(y0, h, label, color, alpha=0.9):
        ax_volumes.add_patch(plt.Rectangle((x0 - width/2, y0), width, h, color=color, alpha=alpha))
        ax_volumes.text(x0, y0 + h/2, label, ha="center", va="center", fontsize=10, color="white", weight="bold")

    rect(0, RV,  "RV\nVolume Residual",            "#374151", 0.95)
    rect(RV, ERV, "ERV\nReserva Expiratória",      "#6b7280", 0.95)
    rect(RV + ERV, VT, "VT\nVolume Corrente",      "#2563eb", 0.75)
    rect(RV + ERV + VT, IRV, "IRV\nReserva Insp.", "#0f766e", 0.90)

    # Reference lines
    ax_volumes.axhline(FRC, color="#111827", lw=2, alpha=0.65)
    ax_volumes.text(0.05, FRC, f" FRC = ERV+RV = {FRC} mL", va="bottom", fontsize=10)

    ax_volumes.axhline(TLC, color="#111827", lw=2, alpha=0.65)
    ax_volumes.text(0.05, TLC, f" TLC = {TLC} mL", va="bottom", fontsize=10)

    # Dynamic marker (absolute volume)
    ax_volumes.plot([x0 - width/2, x0 + width/2], [vol_abs, vol_abs], color="#dc2626", lw=4)
    ax_volumes.text(x0 + 0.32, vol_abs, f"V(t)={int(vol_abs)} mL", color="#dc2626", fontsize=10, va="center")

    ax_volumes.set_xlim(0, 1)
    ax_volumes.set_ylim(0, TLC * 1.05)
    ax_volumes.set_xticks([])
    ax_volumes.set_ylabel("Volume (mL)")

    # Fórmulas (overlay)
    ax_volumes.text(0.05, 0.98 * TLC,
                    "Definições:\nFRC = ERV + RV\nTLC = RV + ERV + VT + IRV",
                    fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#e5e7eb"))

    # ------------------------------------------------------------
    # (B) Mecânica: pulmão + diafragma
    # ------------------------------------------------------------
    ax_mech.set_title("Mecânica: diafragma desce → Ppl fica mais negativa → Palv cai → ar entra", fontsize=11, weight="bold")
    ax_mech.set_xlim(0, 1)
    ax_mech.set_ylim(0, 1)
    ax_mech.axis("off")

    # Lung schematic (size linked to V(t))
    lung_r = 0.18 + 0.10*(v_tidal / VT)
    ax_mech.add_patch(plt.Circle((0.30, 0.60), lung_r, fill=False, lw=6, color="#2563eb"))
    ax_mech.text(0.30, 0.60, "Pulmão", ha="center", va="center", fontsize=10, weight="bold")

    # Diaphragm arc (descends in inspiration)
    dia_y = 0.25 - 0.10*(v_tidal / VT)
    xs = np.linspace(0.10, 0.55, 200)
    arch = dia_y + 0.08*np.sin(np.pi*(xs-0.10)/0.45)
    ax_mech.plot(xs, arch, lw=6, color="#111827")
    ax_mech.text(0.56, dia_y + 0.02, "Diafragma", fontsize=9, va="center")

    # Direction arrows
    if breath_phase(t)[1] == "insp":
        ax_mech.annotate("", xy=(0.62, dia_y-0.08), xytext=(0.62, dia_y+0.08),
                         arrowprops=dict(arrowstyle="->", lw=3, color="#dc2626"))
        ax_mech.text(0.65, dia_y, "contrai\n(desce)", color="#dc2626", fontsize=9, va="center")
    else:
        ax_mech.annotate("", xy=(0.62, dia_y+0.08), xytext=(0.62, dia_y-0.08),
                         arrowprops=dict(arrowstyle="->", lw=3, color="#16a34a"))
        ax_mech.text(0.65, dia_y, "relaxa\n(sobe)", color="#16a34a", fontsize=9, va="center")

    # Instant readouts
    ax_mech.text(0.72, 0.82, f"Ppl ≈ {ppl:.1f} cmH₂O", fontsize=11, weight="bold", color="#111827")
    ax_mech.text(0.72, 0.74, f"Palv ≈ {palv:.1f} cmH₂O", fontsize=11, weight="bold", color="#2563eb")
    ax_mech.text(0.72, 0.66, f"PL = Palv−Ppl ≈ {pl:.1f} cmH₂O", fontsize=11, weight="bold", color="#7c3aed")
    ax_mech.text(0.72, 0.58, f"Fluxo ≈ {flow*60:.0f} L/min", fontsize=11, weight="bold", color="#dc2626")

    # Mini bullets
    ax_mech.text(0.72, 0.46, "Inspiração normal:", fontsize=10, weight="bold")
    ax_mech.text(0.72, 0.40, "Ppl: −5 → −8", fontsize=10)
    ax_mech.text(0.72, 0.34, "Palv: 0 → −1", fontsize=10)
    ax_mech.text(0.72, 0.28, "Fluxo: desacelera → 0", fontsize=10)

    # ------------------------------------------------------------
    # (C) Traçados: Ppl, Palv, PL e Fluxo (histórico)
    # ------------------------------------------------------------
    ax_traces.set_title("Traçados fisiológicos (normal): Pressões e fluxo", fontsize=11, weight="bold")

    ax_traces.plot(t_hist, ppl_hist,  lw=2.4, color="#111827", label="Ppl (pleural)")
    ax_traces.plot(t_hist, palv_hist, lw=2.4, color="#2563eb", label="Palv (alveolar)")
    ax_traces.plot(t_hist, pl_hist,   lw=2.4, color="#7c3aed", label="PL = Palv−Ppl")

    ax_traces.axhline(0, color="#9ca3af", lw=1.2)
    ax_traces.set_ylabel("Pressão (cmH₂O)")
    ax_traces.set_ylim(-10, +10)
    ax_traces.grid(True, alpha=0.25)

    # Flow on secondary axis
    ax2 = ax_traces.twinx()
    ax2.plot(t_hist, flow_hist, lw=2.4, color="#dc2626", label="Fluxo (L/min)")
    ax2.set_ylabel("Fluxo (L/min)")
    ax2.set_ylim(-60, 60)

    # Combined legend
    l1, lab1 = ax_traces.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax_traces.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8, frameon=True)

    ax_traces.set_xlabel("Tempo (s)")

    # Overlay formulas
    ax_traces.text(0.01, 0.02,
                   "Fórmulas:\nPL = Palv − Ppl\nFluxo(t): desacelerante (normal)\n∫Fluxo dt = VT (conceito)",
                   transform=ax_traces.transAxes,
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#e5e7eb"))

    # ------------------------------------------------------------
    # (D) Zoom didático (últimos 6 s): mostrar setas -5→-8 e 0→-1
    # ------------------------------------------------------------
    ax_zoom.set_title("Zoom didático (últimos ~6 s): amplitudes pequenas mas decisivas", fontsize=12, weight="bold")

    ax_zoom.plot(t_zoom, ppl_zoom,  lw=2.8, color="#111827", label="Ppl")
    ax_zoom.plot(t_zoom, palv_zoom, lw=2.8, color="#2563eb", label="Palv")
    ax_zoom.plot(t_zoom, pl_zoom,   lw=2.8, color="#7c3aed", label="PL")

    ax_zoom.axhline(0, color="#9ca3af", lw=1.2)
    ax_zoom.set_ylabel("Pressão (cmH₂O)")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.set_ylim(-10, +10)
    ax_zoom.set_xlabel("Tempo (s)")

    # Annotate target deltas (approximate)
    # Place arrows in the current window, using fixed y-values for clarity
    ax_zoom.annotate("ΔPpl ≈ 3 cmH₂O\n(−5→−8)",
                     xy=(t, PPL_INSP), xytext=(t_zoom[0], -9),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#111827"),
                     fontsize=10, color="#111827")

    ax_zoom.annotate("ΔPalv ≈ 1 cmH₂O\n(0→−1)",
                     xy=(t, PALV_MIN), xytext=(t_zoom[0], 2.5),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#2563eb"),
                     fontsize=10, color="#2563eb")

    ax_zoom.legend(loc="upper right", fontsize=9, frameon=True)

    # ============================================================
    # Global title
    # ============================================================
    ph = breath_phase(t)[1]
    fig.suptitle(
        f"Respiração Normal (ciclo: {ph}) — Volumes + Mecânica + Pressões (Ppl/Palv/PL) + Fluxo",
        fontsize=13, weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Frame
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
