import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

OUT = "Spirograma_Dinamico_Livro_LOOP_60s.mp4"

FPS = 15
DURATION_S = 60

# =========================
# Volumes (mL) — didáticos
# =========================
VT  = 500
IRV = 3000
ERV = 1100
RV  = 1200

FRC = ERV + RV
TLC = RV + ERV + VT + IRV

V_TIDAL_MIN = FRC
V_TIDAL_MAX = FRC + VT
V_TIDAL_MID = FRC + 0.5 * VT

# =========================
# Loop timing
# =========================
# Um ciclo completo que se repete (loop perfeito)
T_LOOP = 20.0  # segundos

# Dentro do loop:
# 0–7s: tidal
# 7–9s: inspiração forçada até TLC
# 9–11s: expiração forçada até RV
# 11–13s: recuperação para tidal
# 13–20s: tidal

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return 0.5 - 0.5*np.cos(np.pi*x)

def tidal(t):
    """Oscilação tidal contínua e periódica."""
    # período ~3.5s só para parecer respiração normal
    return V_TIDAL_MID + 0.5*VT*np.sin(2*np.pi*t/3.5)

def volume(t):
    """Volume absoluto (mL) com manobras forçadas e loop perfeito."""
    tau = t % T_LOOP
    base = tidal(tau)

    # Inspiração forçada (rampa suave para TLC)
    if 7.0 <= tau < 9.0:
        x = (tau - 7.0) / 2.0
        return (1 - smoothstep(x)) * base + smoothstep(x) * TLC

    # Expiração forçada (rampa suave para RV)
    if 9.0 <= tau < 11.0:
        x = (tau - 9.0) / 2.0
        start = TLC
        return (1 - smoothstep(x)) * start + smoothstep(x) * RV

    # Recuperação (RV -> tidal)
    if 11.0 <= tau < 13.0:
        x = (tau - 11.0) / 2.0
        return (1 - smoothstep(x)) * RV + smoothstep(x) * tidal(tau)

    return base

# =========================
# Plot helpers
# =========================
def canvas_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

def v_arrow(ax, x, y0, y1, text, fontsize=10):
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops=dict(arrowstyle="<->", lw=2.2, color="black")
    )
    ax.text(x + 0.15, (y0+y1)/2, text, va="center", ha="left",
            fontsize=fontsize, color="black")

# =========================
# Rendering
# =========================
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)

writer = imageio.get_writer(
    OUT,
    fps=FPS,
    codec="libx264",
    macro_block_size=1,
    ffmpeg_params=["-preset", "ultrafast", "-crf", "28"]
)

# Janela temporal visível no ecrã (como gráfico de livro)
WINDOW = 12.0  # segundos visíveis

for i in range(int(DURATION_S * FPS)):
    t = i / FPS
    tau = t % T_LOOP

    # Eixo visível 0..WINDOW, mas os dados são recortados do loop
    t_axis = np.linspace(0, WINDOW, int(WINDOW * FPS))
    # Amostramos um segmento do loop que termina em "tau" (visualmente fluido)
    t_samples = (tau - WINDOW + t_axis) % T_LOOP
    v = np.array([volume(ts) for ts in t_samples])

    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    # Estilo “livro”
    ax.set_facecolor("#efe2b6")
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color("black")

    # Curva dinâmica
    ax.plot(t_axis, v, color="#d32f2f", lw=3)

    # Limites e ticks (como esquema clássico)
    ax.set_xlim(0, WINDOW)
    ax.set_ylim(0, 6000)
    ax.set_xlabel("Tempo", fontsize=13, weight="bold")
    ax.set_ylabel("Volume pulmonar (mL)", fontsize=13, weight="bold")

    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 6001, 1000))
    ax.tick_params(axis="y", labelsize=10)

    # Linhas guia (tidal) + bases
    ax.hlines([V_TIDAL_MIN, V_TIDAL_MAX], 0, WINDOW,
              colors="black", linestyles="--", linewidth=1.6, alpha=0.6)
    ax.hlines([RV, TLC], 0, WINDOW, colors="black", linewidth=2.4, alpha=0.85)

    # Labels (posições fixas)
    ax.text(0.2, TLC + 120, "Capacidade pulmonar total", fontsize=10, color="black")
    ax.text(0.2, FRC + 120, "Capacidade residual funcional", fontsize=10, color="black")
    ax.text(0.2, RV  + 120, "Volume residual", fontsize=10, color="black")

    # Setas verticais (como no livro)
    x1 = 5.0
    v_arrow(ax, x1, V_TIDAL_MAX, TLC, "Volume de reserva\ninspiratório")
    v_arrow(ax, x1 - 1.4, V_TIDAL_MIN, V_TIDAL_MAX, "Volume\ncorrente")
    v_arrow(ax, x1 - 2.8, RV, FRC, "Volume de reserva\nexpiratório")

    # Capacidades “grandes”
    x2 = 10.2
    v_arrow(ax, x2, FRC, TLC, "Capacidade\ninspiratória")
    v_arrow(ax, x2 + 0.8, RV, TLC, "Capacidade\nvital")
    v_arrow(ax, x2 + 1.6, 0, TLC, "Capacidade\npulmonar\ntotal", fontsize=9)

    # Marcador do “agora” no fim da janela
    ax.scatter([WINDOW], [volume(tau)], s=60, color="#2563eb", zorder=5)

    # Título curto
    ax.set_title("Spirograma dinâmico (tidal + manobras forçadas) — loop didáctico", fontsize=14, weight="bold")

    plt.tight_layout()
    writer.append_data(canvas_to_rgb(fig))

writer.close()
print("OK ->", OUT)
