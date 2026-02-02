import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ============================================================
# OUTPUT
# ============================================================
OUT = "Spirograma_Volume_Tempo_Com_Mecanica_Pressoes_60s.mp4"

# ============================================================
# VIDEO SETTINGS
# ============================================================
FPS = 15
DURATION_S = 60

# ============================================================
# Robust trapezoid integration (NumPy compat)
# ============================================================
def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return np.sum((y[:-1] + y[1:]) * 0.5 * dx)

# ============================================================
# LUNG VOLUMES (adulto típico, mL) — valores didácticos
# ============================================================
VT  = 500
IRV = 3000
ERV = 1100
RV  = 1200

FRC = ERV + RV
TLC = RV + ERV + VT + IRV
VC  = TLC - RV  # capacidade vital

# Para desenhar a curva, definimos "níveis alvo" úteis
V_FRC = FRC
V_TIDAL_MID = FRC + 0.35 * VT       # centro da respiração corrente (didático)
V_TIDAL_MIN = V_TIDAL_MID - 0.5*VT  # fim expiração normal ~FRC
V_TIDAL_MAX = V_TIDAL_MID + 0.5*VT  # fim inspiração normal

# ============================================================
# BREATH PHYSIO (normal quiet breathing)
# ============================================================
T_CYCLE = 5.0   # ~12 rpm
INSP = 2.0
EXP  = 3.0

# Pressões alvo (cmH2O) para o "capítulo fisiologia"
PPL_BASE = -5.0
PPL_INSP = -8.0
PALV_BASE = 0.0
PALV_MIN  = -1.0
PALV_EXP  = +1.0

# Fluxo (L/s) — desacelerante (didático)
FLOW_PEAK_INSP = 0.45
FLOW_PEAK_EXP  = -0.35

# ============================================================
# Equação do Movimento (didáctico)
# Paw(t)=R*Vdot + E*V + baseline (baseline conceptual = PEEP no ventilador)
# ============================================================
R_AW = 8.0
C_RS = 0.060
E_RS = 1.0 / C_RS
BAR_MAX_CM = 15.0

# ============================================================
# Helpers
# ============================================================
def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return 0.5 - 0.5*np.cos(np.pi*x)

def breath_phase(t):
    t0 = t % T_CYCLE
    if t0 < INSP:
        return t0, "insp"
    return t0 - INSP, "exp"

def ppl_waveform(t):
    tp, ph = breath_phase(t)
    if ph == "insp":
        return PPL_BASE + (PPL_INSP - PPL_BASE) * smoothstep(tp/INSP)
    return PPL_INSP + (PPL_BASE - PPL_INSP) * smoothstep(tp/EXP)

def palv_waveform(t):
    tp, ph = breath_phase(t)
    if ph == "insp":
        return PALV_BASE + (PALV_MIN - PALV_BASE) * smoothstep(tp/INSP)
    x = tp/EXP
    return PALV_EXP * np.exp(-3.0*x)

def flow_waveform(t):
    tp, ph = breath_phase(t)
    if ph == "insp":
        return FLOW_PEAK_INSP * np.exp(-2.0*(tp/INSP))
    return FLOW_PEAK_EXP * np.exp(-2.2*(tp/EXP))

# ============================================================
# Spirometry-like Volume-Time curve
# - Base: tidal breathing
# - At scheduled moments: big manoeuvres to TLC / RV
# ============================================================
# Eventos (em segundos) para o minuto: (start, type)
# type: "max_insp" (subir até TLC), "forced_exp" (descer até RV), "recover" (voltar a tidal)
EVENTS = [
    (15.0, "max_insp"),
    (20.0, "forced_exp"),
    (25.0, "recover"),
    (40.0, "max_insp"),
    (45.0, "forced_exp"),
    (50.0, "recover"),
]

def tidal_component(t):
    """Pequena oscilação tidal em torno de V_TIDAL_MID."""
    tp, ph = breath_phase(t)
    if ph == "insp":
        frac = smoothstep(tp/INSP)
        return V_TIDAL_MIN + (V_TIDAL_MAX - V_TIDAL_MIN) * frac
    else:
        # expiração passiva: queda exponencial até V_TIDAL_MIN
        tau = 0.9
        return V_TIDAL_MIN + (V_TIDAL_MAX - V_TIDAL_MIN) * np.exp(-(tp)/tau)

def apply_event_target(t):
    """
    Define um alvo de volume (target) durante eventos para simular manobras grandes.
    Retorna (target_volume, blend) onde blend 0..1 indica "força" do evento.
    """
    # Por defeito: sem evento
    target = None
    blend = 0.0

    # janelas de evento simples (3–4s) para produzir picos/vales como a figura
    for (t0, typ) in EVENTS:
        if typ == "max_insp":
            # 0–3s: rampa suave até TLC
            if t0 <= t < t0 + 3.0:
                x = (t - t0) / 3.0
                target = TLC
                blend = smoothstep(x)
                return target, blend
        if typ == "forced_exp":
            # 0–3s: rampa suave até RV (expiração forçada)
            if t0 <= t < t0 + 3.0:
                x = (t - t0) / 3.0
                target = RV
                blend = smoothstep(x)
                return target, blend
        if typ == "recover":
            # 0–4s: volta ao tidal (blend decrescente)
            if t0 <= t < t0 + 4.0:
                x = (t - t0) / 4.0
                target = tidal_component(t)
                blend = smoothstep(x)
                return target, blend

    return target, blend

def volume_spiro(t):
    """
    Volume absoluto (mL) ao longo do tempo:
    - tidal breathing base
    - durante eventos, mistura para TLC/RV e regressa
    """
    base = tidal_component(t)
    target, blend = apply_event_target(t)
    if target is None:
        return base
    # Mistura para o alvo durante o evento
    return (1.0 - blend) * base + blend * target

# ============================================================
# Render windows
# ============================================================
HIST_SEC = 20.0   # spirograma precisa de mais tempo visível
ZOOM_SEC = 6.0

def make_history(t, fps, window_sec):
    t_start = max(0.0, t - window_sec)
    n = int(max(30, min((t - t_start) * fps + 1, window_sec * fps)))
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

    # Signals
    vol_abs = volume_spiro(t)  # mL
    # Para ligação à EOM: precisamos de "V acima de baseline" (tomamos baseline=FRC)
    v_tidal_ml = np.clip(vol_abs - FRC, 0, VT)  # apenas para o módulo mecânico (didático)
    v_tidal_L  = v_tidal_ml / 1000.0

    ppl  = ppl_waveform(t)
    palv = palv_waveform(t)
    flow_Ls = flow_waveform(t)
    pl = palv - ppl

    # EOM terms
    P_res = R_AW * flow_Ls
    P_el  = E_RS * v_tidal_L

    # Histories
    t_hist = make_history(t, FPS, HIST_SEC)
    vol_hist = np.array([volume_spiro(tt) for tt in t_hist])

    ppl_hist  = np.array([ppl_waveform(tt) for tt in t_hist])
    palv_hist = np.array([palv_waveform(tt) for tt in t_hist])
    pl_hist   = palv_hist - ppl_hist
    flow_hist = np.array([flow_waveform(tt) for tt in t_hist]) * 60.0  # L/min

    # VT integration concept (current cycle)
    t_cycle_start = t - (t % T_CYCLE)
    t_cycle_grid = np.linspace(t_cycle_start, t, max(2, int((t - t_cycle_start) * FPS) + 1))
    flow_cycle = np.array([flow_waveform(tt) for tt in t_cycle_grid])  # L/s
    flow_pos = np.clip(flow_cycle, 0, None)
    vt_integrated_ml = integrate_trapezoid(flow_pos, t_cycle_grid) * 1000.0

    # Zoom
    t_zoom = make_history(t, FPS, ZOOM_SEC)
    ppl_zoom  = np.array([ppl_waveform(tt) for tt in t_zoom])
    palv_zoom = np.array([palv_waveform(tt) for tt in t_zoom])
    pl_zoom   = palv_zoom - ppl_zoom

    # ============================================================
    # LAYOUT
    # ============================================================
    fig.clf()
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.35, 1.05, 1.05, 0.95],
        height_ratios=[1.0, 1.0]
    )

    ax_spiro  = fig.add_subplot(gs[:, 0])   # <- curva Volume-Tempo tipo o teu exemplo
    ax_mech   = fig.add_subplot(gs[0, 1])
    ax_traces = fig.add_subplot(gs[1, 1])
    ax_integr = fig.add_subplot(gs[:, 2])
    ax_zoom   = fig.add_subplot(gs[:, 3])

    # ------------------------------------------------------------
    # (A) SPIROGRAM: Volume vs Tempo (como a imagem)
    # ------------------------------------------------------------
    ax_spiro.set_title("Spirograma: Volume pulmonar ao longo do tempo", fontsize=12, weight="bold")
    ax_spiro.plot(t_hist, vol_hist, lw=2.8, color="#dc2626")

    # Linhas de referência
    ax_spiro.axhline(RV,  color="#111827", lw=1.6, alpha=0.75)
    ax_spiro.axhline(FRC, color="#111827", lw=1.6, alpha=0.55)
    ax_spiro.axhline(TLC, color="#111827", lw=1.6, alpha=0.75)

    # Etiquetas laterais (como o esquema clássico)
    ax_spiro.text(t_hist[0], TLC + 80, "Capacidade Pulmonar Total (TLC)", fontsize=9, color="#111827")
    ax_spiro.text(t_hist[0], FRC + 80, "Capacidade Residual Funcional (FRC)", fontsize=9, color="#111827")
    ax_spiro.text(t_hist[0], RV  + 80, "Volume Residual (RV)", fontsize=9, color="#111827")

    # Marcador do instante actual
    ax_spiro.scatter([t], [vol_abs], s=50, color="#2563eb", zorder=5)
    ax_spiro.text(t, vol_abs + 120, f"{int(vol_abs)} mL", fontsize=9, color="#2563eb", ha="center")

    # Setas para IRV / ERV / VT / VC (didático)
    # VT: entre V_TIDAL_MIN e V_TIDAL_MAX
    xA = t_hist[0] + 1.0
    ax_spiro.annotate("", xy=(xA, V_TIDAL_MAX), xytext=(xA, V_TIDAL_MIN),
                      arrowprops=dict(arrowstyle="<->", lw=2, color="#2563eb"))
    ax_spiro.text(xA + 0.2, (V_TIDAL_MIN + V_TIDAL_MAX)/2, "VT\n(Volume corrente)",
                  fontsize=9, color="#2563eb", va="center")

    # ERV: entre FRC e RV
    xB = t_hist[0] + 3.0
    ax_spiro.annotate("", xy=(xB, FRC), xytext=(xB, RV),
                      arrowprops=dict(arrowstyle="<->", lw=2, color="#6b7280"))
    ax_spiro.text(xB + 0.2, (FRC + RV)/2, "ERV\n(Reserva exp.)", fontsize=9, color="#6b7280", va="center")

    # IRV: entre TLC e V_TIDAL_MAX (aprox)
    xC = t_hist[0] + 5.0
    ax_spiro.annotate("", xy=(xC, TLC), xytext=(xC, V_TIDAL_MAX),
                      arrowprops=dict(arrowstyle="<->", lw=2, color="#0f766e"))
    ax_spiro.text(xC + 0.2, (TLC + V_TIDAL_MAX)/2, "IRV\n(Reserva insp.)", fontsize=9, color="#0f766e", va="center")

    # VC: TLC a RV
    xD = t_hist[-1] - 2.0
    ax_spiro.annotate("", xy=(xD, TLC), xytext=(xD, RV),
                      arrowprops=dict(arrowstyle="<->", lw=2.2, color="#111827"))
    ax_spiro.text(xD - 0.2, (TLC + RV)/2, "VC\n(Capacidade vital)", fontsize=9, color="#111827",
                  va="center", ha="right")

    ax_spiro.set_ylabel("Volume (mL)")
    ax_spiro.set_xlabel("Tempo (s)")
    ax_spiro.set_ylim(0, TLC * 1.08)
    ax_spiro.grid(True, alpha=0.20)

    # ------------------------------------------------------------
    # (B) Mecânica + EOM
    # ------------------------------------------------------------
    ax_mech.set_title("Mecânica + Equação do Movimento (ponte para ventilador)", fontsize=11, weight="bold")
    ax_mech.set_xlim(0, 1)
    ax_mech.set_ylim(0, 1)
    ax_mech.axis("off")

    # Pulmão (tamanho ligado ao componente tidal apenas — não à manobra)
    lung_r = 0.18 + 0.10*(v_tidal_ml / max(VT, 1e-6))
    ax_mech.add_patch(plt.Circle((0.28, 0.62), lung_r, fill=False, lw=6, color="#2563eb"))
    ax_mech.text(0.28, 0.62, "Pulmão", ha="center", va="center", fontsize=10, weight="bold")

    # Diafragma
    dia_y = 0.25 - 0.10*(v_tidal_ml / max(VT, 1e-6))
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

    res_level = min(abs(flow_Ls) / max(abs(FLOW_PEAK_INSP), 1e-6), 1.0)
    el_level  = min(v_tidal_ml / max(VT, 1e-6), 1.0)
    res_alpha = 0.25 + 0.60*res_level
    el_alpha  = 0.25 + 0.60*el_level

    ax_mech.text(
        0.05, 0.46,
        "Equação do movimento:\n"
        r"$P_{aw}(t)=R\cdot\dot V(t) + E\cdot V(t) + \mathrm{baseline}$",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, edgecolor="#e5e7eb")
    )

    ax_mech.text(0.06, 0.30, r"$R\cdot\dot V$ (resistivo)",
                 fontsize=10, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fecaca", alpha=res_alpha, edgecolor="#fecaca"))
    ax_mech.text(0.06, 0.22, r"$E\cdot V$ (elástico)",
                 fontsize=10, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#ddd6fe", alpha=el_alpha, edgecolor="#ddd6fe"))
    ax_mech.text(0.06, 0.14, "baseline (no ventilador = PEEP)",
                 fontsize=10, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#e5e7eb", alpha=0.85, edgecolor="#e5e7eb"))

    # Bars common scale
    P_res = R_AW * flow_Ls
    P_el  = E_RS * (v_tidal_ml/1000.0)
    P_res_mag = min(abs(P_res), BAR_MAX_CM)
    P_el_mag  = min(P_el,       BAR_MAX_CM)

    ax_mech.add_patch(plt.Rectangle((0.78, 0.10), 0.06, 0.30, fill=False, lw=2, color="#9ca3af"))
    ax_mech.add_patch(plt.Rectangle((0.88, 0.10), 0.06, 0.30, fill=False, lw=2, color="#9ca3af"))
    ax_mech.add_patch(plt.Rectangle((0.78, 0.10), 0.06, 0.30*(P_res_mag/BAR_MAX_CM), color="#ef4444", alpha=0.85))
    ax_mech.add_patch(plt.Rectangle((0.88, 0.10), 0.06, 0.30*(P_el_mag/BAR_MAX_CM),  color="#7c3aed", alpha=0.70))
    ax_mech.text(0.81, 0.42, "R·V’", ha="center", fontsize=9, weight="bold", color="#b91c1c")
    ax_mech.text(0.91, 0.42, "E·V",  ha="center", fontsize=9, weight="bold", color="#6d28d9")

    # ------------------------------------------------------------
    # (C) Traçados: Ppl/Palv/PL + fluxo
    # ------------------------------------------------------------
    ax_traces.set_title("Traçados: pressões e fluxo (respiração corrente)", fontsize=11, weight="bold")
    ax_traces.plot(t_hist, ppl_hist,  lw=2.2, color="#111827", label="Ppl")
    ax_traces.plot(t_hist, palv_hist, lw=2.2, color="#2563eb", label="Palv")
    ax_traces.plot(t_hist, pl_hist,   lw=2.2, color="#7c3aed", label="PL")
    ax_traces.axhline(0, color="#9ca3af", lw=1.2)
    ax_traces.set_ylim(-10, +10)
    ax_traces.set_ylabel("Pressão (cmH₂O)")
    ax_traces.grid(True, alpha=0.25)

    ax2 = ax_traces.twinx()
    ax2.plot(t_hist, flow_hist, lw=2.2, color="#dc2626", label="Fluxo (L/min)")
    ax2.set_ylim(-60, 60)
    ax2.set_ylabel("Fluxo (L/min)")

    l1, lab1 = ax_traces.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax_traces.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8, frameon=True)
    ax_traces.set_xlabel("Tempo (s)")

    # Shade phases in last cycle
    ax_traces.axvspan(t_cycle_start, t_cycle_start + INSP, color="#fecaca", alpha=0.18)
    ax_traces.axvspan(t_cycle_start + INSP, t_cycle_start + T_CYCLE, color="#bbf7d0", alpha=0.10)

    # ------------------------------------------------------------
    # (D) Integração: VT = ∫Fluxo dt
    # ------------------------------------------------------------
    ax_integr.set_title("Integração: VT ≈ ∫ Fluxo(t) dt (conceito)", fontsize=12, weight="bold")
    ax_integr.grid(True, alpha=0.25)

    t_c = np.linspace(t_cycle_start, t_cycle_start + T_CYCLE, int(T_CYCLE * FPS))
    flow_c = np.array([flow_waveform(tt) for tt in t_c]) * 60.0
    ax_integr.plot(t_c, flow_c, lw=2.4, color="#dc2626", label="Fluxo (L/min)")
    ax_integr.axhline(0, color="#9ca3af", lw=1.2)

    t_fill = t_c[t_c <= t]
    flow_fill = np.array([flow_waveform(tt) for tt in t_fill]) * 60.0
    ax_integr.fill_between(t_fill, 0, np.clip(flow_fill, 0, None), color="#fecaca", alpha=0.45, label="Área insp.")
    ax_integr.set_ylim(-60, 60)
    ax_integr.set_xlabel("Tempo (s)")
    ax_integr.set_ylabel("Fluxo (L/min)")
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
    # (E) Zoom pressões pequenas
    # ------------------------------------------------------------
    ax_zoom.set_title("Zoom: amplitudes pequenas (cmH₂O) → efeitos reais", fontsize=12, weight="bold")
    ax_zoom.plot(t_zoom, ppl_zoom,  lw=2.6, color="#111827", label="Ppl")
    ax_zoom.plot(t_zoom, palv_zoom, lw=2.6, color="#2563eb", label="Palv")
    ax_zoom.plot(t_zoom, pl_zoom,   lw=2.6, color="#7c3aed", label="PL")
    ax_zoom.axhline(0, color="#9ca3af", lw=1.2)
    ax_zoom.set_ylim(-10, +10)
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.set_xlabel("Tempo (s)")
    ax_zoom.set_ylabel("Pressão (cmH₂O)")
    ax_zoom.legend(loc="upper right", fontsize=9, frameon=True)

    ax_zoom.axhline(PPL_BASE, color="#111827", lw=1.0, alpha=0.25)
    ax_zoom.axhline(PPL_INSP, color="#111827", lw=1.0, alpha=0.25)
    ax_zoom.axhline(PALV_BASE, color="#2563eb", lw=1.0, alpha=0.25)
    ax_zoom.axhline(PALV_MIN,  color="#2563eb", lw=1.0, alpha=0.25)

    # ============================================================
    # Global title
    # ============================================================
    fig.suptitle(
        "Fisiologia Respiratória — do spirograma (volume-tempo) à mecânica (pressões/fluxo) e equação do movimento",
        fontsize=13, weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
