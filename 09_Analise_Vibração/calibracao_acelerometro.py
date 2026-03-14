# =============================================================================
#  CALIBRAÇÃO DE ACELERÔMETRO — Engenharia Mecânica | Vibrações Mecânicas
# =============================================================================
#
#  Objetivo:
#    Encontrar o fator de conversão (sensitivity) do acelerômetro, ou seja,
#    quantos Volts o sensor produz para cada 1 g de aceleração.
#    Com esse fator, qualquer sinal em Volts pode ser convertido para g.
#
#  Equipamento utilizado:
#    Calibrador de acelerômetros — sinal de referência: 159,2 Hz e 1 g de pico
#
#  Taxa de amostragem: 1000 Hz
#  Canal analisado: CH2
#
#  Princípio básico:
#    sensitivity [V/g] = amplitude_pico_medida [V] / amplitude_referência [g]
#
#    Como a referência é 1 g de pico, o fator de conversão é simplesmente
#    a amplitude de pico lida no espectro FFT na frequência do calibrador.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configurações gerais de plot ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.2,
})

# =============================================================================
# PASSO 1 — Carregar os dados
# =============================================================================
print("=" * 60)
print(" PASSO 1 — Carregando os dados")
print("=" * 60)

# Lemos o arquivo CSV e ignoramos a coluna de tempo (Time)
# "C:\Users\gnple\OneDrive\Área de Trabalho\Calibrador\calib_10000Hz_CH2.csv"
# "C:\Users\gnple\OneDrive\Área de Trabalho\Calibrador\calib_1000Hz_CH2.csv"

df = pd.read_csv(
    r"C:\Users\gnple\OneDrive\Área de Trabalho\Calibrador\calib_10000Hz_CH2.csv",
    index_col=False,          # evita interpretação incorreta das colunas
    usecols=["CH1", "CH2", "CH3", "CH4"]   # ignora coluna Time
)

# Canal de interesse: CH2 (conectado ao calibrador)
canal = "CH2"
sinal_V = df[canal].dropna().values          # sinal em Volts

print(f"\n  Canal selecionado : {canal}")
print(f"  Total de amostras : {len(sinal_V)}")

# =============================================================================
# PASSO 2 — Definir parâmetros de aquisição
# =============================================================================
print("\n" + "=" * 60)
print(" PASSO 2 — Parâmetros de aquisição")
print("=" * 60)

fs   = 10000         # Frequência de amostragem [Hz]
N    = len(sinal_V)  # Número de amostras
dt   = 1 / fs        # Intervalo de tempo entre amostras [s]
T    = N * dt        # Duração total do sinal [s]
df_freq = fs / N     # Resolução em frequência [Hz]

print(f"\n  Freq. de amostragem (fs)    : {fs} Hz")
print(f"  Número de amostras (N)      : {N}")
print(f"  Duração do sinal (T)        : {T:.3f} s")
print(f"  Resolução em frequência     : {df_freq:.4f} Hz")

# Eixo de tempo para plotagem
t = np.arange(N) * dt

# =============================================================================
# PASSO 3 — Calcular a FFT
# =============================================================================
print("\n" + "=" * 60)
print(" PASSO 3 — Calculando a FFT (Transformada de Fourier)")
print("=" * 60)

# A FFT decompõe o sinal em suas componentes de frequência.
# Usamos apenas a metade positiva do espectro (teorema de Nyquist).

fft_completa  = np.fft.fft(sinal_V)              # FFT complexa
freq          = np.fft.fftfreq(N, d=dt)          # Eixo de frequências

# Selecionamos apenas frequências positivas
idx_positivo  = freq >= 0
freq_pos      = freq[idx_positivo]
fft_pos       = fft_completa[idx_positivo]

# Amplitude de pico = (2 * |FFT|) / N
# O fator 2 recupera a energia do lado negativo; dividir por N normaliza.
amplitude_V   = (2 * np.abs(fft_pos)) / N

print(f"\n  FFT calculada com {N} pontos.")
print(f"  Frequência máxima representável (Nyquist): {fs/2} Hz")

# =============================================================================
# PASSO 4 — Localizar o pico na frequência do calibrador
# =============================================================================
print("\n" + "=" * 60)
print(" PASSO 4 — Identificando o pico na frequência do calibrador")
print("=" * 60)

freq_calib      = 159.2          # Frequência do calibrador [Hz]
amp_ref_g_rms   = 1.00           # Amplitude de referência — datasheet [g RMS]

# ── Conversão RMS → Pico ──────────────────────────────────────────────────────
# Para um sinal senoidal puro a relação entre pico e RMS é:
#
#         A_pico = A_rms × √2     ←→     A_rms = A_pico / √2
#
# A FFT (com a normalização usada acima) retorna amplitudes de PICO.
# Portanto, para comparar grandezas equivalentes, precisamos da referência
# também em pico: 1,00 g_rms × √2 = 1,4142 g_pico
amp_ref_g_pico  = amp_ref_g_rms * np.sqrt(2)

print(f"\n  Referência (datasheet)           : {amp_ref_g_rms:.2f} g RMS")
print(f"  Referência convertida para pico  : {amp_ref_g_pico:.4f} g pico  (= {amp_ref_g_rms} × √2)")

# Janela de busca: ± 2 Hz em torno da frequência nominal
tolerancia   = 2.0            # [Hz]
mascara      = np.abs(freq_pos - freq_calib) <= tolerancia
idx_pico     = np.argmax(amplitude_V[mascara])

# Índice e valores do pico dentro da janela
freq_pico    = freq_pos[mascara][idx_pico]
amp_pico_V   = amplitude_V[mascara][idx_pico]
amp_rms_V    = amp_pico_V / np.sqrt(2)   # amplitude RMS do sinal medido

print(f"\n  Frequência nominal do calibrador : {freq_calib} Hz")
print(f"  Frequência encontrada no pico    : {freq_pico:.2f} Hz")
print(f"  Amplitude medida — pico (FFT)    : {amp_pico_V:.6f} V")
print(f"  Amplitude medida — RMS           : {amp_rms_V:.6f} V  (= pico / √2)")

# =============================================================================
# PASSO 5 — Calcular o fator de conversão (sensitivity)
# =============================================================================
print("\n" + "=" * 60)
print(" PASSO 5 — Calculando o fator de conversão")
print("=" * 60)

# ── Duas abordagens equivalentes para calcular a sensitivity ─────────────────
#
#  Abordagem A — comparar pico com pico (recomendada com FFT):
#    sensitivity = amp_pico_V  / amp_ref_g_pico
#
#  Abordagem B — comparar RMS com RMS (equivalente):
#    sensitivity = amp_rms_V   / amp_ref_g_rms
#
#  Ambas devem dar o mesmo resultado. Usaremos a Abordagem A.

sensitivity_V_g = amp_pico_V / amp_ref_g_pico   # [V/g]  — pico/pico
sensitivity_check = amp_rms_V / amp_ref_g_rms    # [V/g]  — rms/rms (verificação)

fator_g_V = 1.0 / sensitivity_V_g

print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  Amplitude medida (pico, FFT)   : {amp_pico_V:.6f} V pico       │
  │  Amplitude medida (RMS)         : {amp_rms_V:.6f} V rms        │
  │                                                          │
  │  Referência (datasheet)         : {amp_ref_g_rms:.4f} g rms         │
  │  Referência convertida p/ pico  : {amp_ref_g_pico:.4f} g pico        │
  │                                                          │
  │  SENSITIVITY (pico/pico)        : {sensitivity_V_g:.6f} V/g          │
  │  SENSITIVITY (rms/rms, check)   : {sensitivity_check:.6f} V/g          │
  │  FATOR INVERSO                  : {fator_g_V:.4f} g/V             │
  └──────────────────────────────────────────────────────────┘

  Interpretação:
    • Para cada 1 g de aceleração, o sensor gera {sensitivity_V_g:.6f} V.
    • Para converter um sinal de X Volts em g, basta calcular:

              aceleração [g] = sinal [V] / {sensitivity_V_g:.6f}
                             = sinal [V] × {fator_g_V:.4f}
""")

# =============================================================================
# PASSO 6 — Converter o sinal completo para g (exemplo de aplicação)
# =============================================================================
print("=" * 60)
print(" PASSO 6 — Convertendo o sinal de V para g")
print("=" * 60)

sinal_g = sinal_V / sensitivity_V_g   # conversão direta

print(f"\n  Primeiros 5 valores em Volts : {sinal_V[:5].round(6)}")
print(f"  Primeiros 5 valores em g     : {sinal_g[:5].round(6)}")

# =============================================================================
# PASSO 7 — Visualizações
# =============================================================================
print("\n" + "=" * 60)
print(" PASSO 7 — Gerando gráficos")
print("=" * 60)

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "Calibração de Acelerômetro — Engenharia de Vibrações\n"
    f"Canal: {canal}  |  Calibrador: {freq_calib} Hz / {amp_ref_g_rms} g RMS  |  fs = {fs} Hz",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

# ── Subplot 1: Série temporal em Volts ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
trecho = slice(0, min(2000, N))          # exibe até 2 s para facilitar leitura
ax1.plot(t[trecho], sinal_V[trecho], color="#2196F3", alpha=0.85)
ax1.set_xlabel("Tempo [s]")
ax1.set_ylabel("Amplitude [V]")
ax1.set_title("Série Temporal — Sinal Bruto em Volts")
ax1.set_xlim(t[trecho.start], t[trecho.stop - 1])

# ── Subplot 2: Espectro FFT completo (em V) ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(freq_pos, amplitude_V, color="#4CAF50", alpha=0.8)
ax2.axvline(freq_pico, color="red", ls="--", lw=1.4,
            label=f"Pico: {freq_pico:.1f} Hz → {amp_pico_V:.5f} V")
ax2.set_xlabel("Frequência [Hz]")
ax2.set_ylabel("Amplitude [V]")
ax2.set_title("Espectro FFT — Amplitude em Volts")
ax2.set_xlim(0, fs / 2)
ax2.legend()

# ── Subplot 3: Zoom no pico do calibrador ────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
janela_plot = 15    # Hz de cada lado para visualização
mask_zoom   = (freq_pos >= freq_calib - janela_plot) & \
              (freq_pos <= freq_calib + janela_plot)
ax3.plot(freq_pos[mask_zoom], amplitude_V[mask_zoom],
         color="#FF9800", marker="o", ms=3)
ax3.axvline(freq_pico, color="red", ls="--", lw=1.4,
            label=f"{freq_pico:.1f} Hz\n{amp_pico_V:.5f} V")
ax3.set_xlabel("Frequência [Hz]")
ax3.set_ylabel("Amplitude [V]")
ax3.set_title(f"Zoom — Região do Calibrador ({freq_calib} Hz)")
ax3.legend(fontsize=9)

# ── Subplot 4: Série temporal convertida em g ────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(t[trecho], sinal_g[trecho], color="#9C27B0", alpha=0.85)
ax4.axhline(0, color="k", lw=0.7, ls="--")
ax4.set_xlabel("Tempo [s]")
ax4.set_ylabel("Aceleração [g]")
ax4.set_title("Série Temporal — Sinal Convertido para g")
ax4.set_xlim(t[trecho.start], t[trecho.stop - 1])

# Anotação do fator de conversão
fig.text(
    0.5, 0.01,
    f"Referência: {amp_ref_g_rms:.2f} g RMS = {amp_ref_g_pico:.4f} g pico   |   "
    f"Sensitivity = {sensitivity_V_g:.6f} V/g   |   "
    f"Fator de conversão = {fator_g_V:.4f} g/V",
    ha="center", fontsize=10, color="#C62828",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", alpha=0.85)
)

plt.savefig("calibracao_acelerometro.png", bbox_inches="tight")
plt.show()
print("\n  Figura salva: calibracao_acelerometro.png")

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "=" * 60)
print(" RESUMO DA CALIBRAÇÃO")
print("=" * 60)
print(f"""
  Canal analisado              : {canal}
  Frequência do calibrador     : {freq_calib} Hz
  Pico identificado em         : {freq_pico:.2f} Hz

  ── Amplitudes ──────────────────────────────────────────
  Sinal medido (pico)          : {amp_pico_V:.6f} V
  Sinal medido (RMS)           : {amp_rms_V:.6f} V
  Referência datasheet         : {amp_ref_g_rms:.2f} g RMS
  Referência convertida        : {amp_ref_g_pico:.4f} g pico

  ── Fórmula ─────────────────────────────────────────────
  sensitivity = amp_pico_V / amp_ref_g_pico
              = {amp_pico_V:.6f} / {amp_ref_g_pico:.4f}

  Sensitivity                  : {sensitivity_V_g:.6f} V/g
  Fator inverso                : {fator_g_V:.4f} g/V

  Para converter seu sinal:
    aceleração [g] = sinal [V] × {fator_g_V:.4f}
""")
