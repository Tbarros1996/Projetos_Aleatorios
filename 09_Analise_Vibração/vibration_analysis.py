"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ANALISADOR DE VIBRAÇÕES EDUCACIONAL - DISCIPLINA DE VIBRAÇÕES       ║
║          Instituto Federal de Pernambuco - Engenharia Mecânica               ║
║                                                                              ║
║  Canais de aquisição:                                                        ║
║    CH1 → Tacômetro (referência de fase / rotação)                            ║
║    CH2 → Acelerômetro (Radial Vertical)                                      ║
║    CH3 → Acelerômetro (Radial Horizontal)                                    ║
║    CH4 → Acelerômetro (Axial)                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Módulos didáticos implementados:
  1. Carregamento e inspeção dos dados
  2. Conceitos de amostragem (Nyquist, resolução, aliasing)
  3. Análise do tacômetro e extração de RPM
  4. Condicionamento do sinal (filtros, janelas)
  5. Análise espectral (FFT com múltiplas janelas)
  6. Análise de ordem (Order Tracking)
  7. Diagnóstico de falhas (desbalanceamento, desalinhamento, rolamentos,
     ventiladores, engrenagens, polias/correias, folgas mecânicas)
  8. Órbita e análise temporal multi-canal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
import pandas as pd
from datetime import datetime
import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE CORES E ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#e0e0e0',
    'axes.labelcolor':  '#e0e0e0',
    'axes.titlecolor':  '#ffffff',
    'xtick.color':      '#c0c0c0',
    'ytick.color':      '#c0c0c0',
    'grid.color':       '#334466',
    'grid.linestyle':   '--',
    'grid.alpha':       0.4,
    'text.color':       '#e0e0e0',
    'legend.facecolor': '#0f3460',
    'legend.edgecolor': '#aaaaaa',
    'font.family':      'monospace',
})

COR_CH2 = '#00d4ff'   # azul ciano  – Acelerômetro radial vertical
COR_CH3 = '#ff6b6b'   # vermelho    – Acelerômetro radial horizontal
COR_CH4 = '#ffd93d'   # amarelo     – Acelerômetro axial
COR_TACO = '#c3f584'  # verde lima  – Tacômetro
COR_DEST = '#ff9a3c'  # laranja     – destaque / alertas
COR_OK   = '#6bcb77'  # verde       – indicadores normais

# Nomes padronizados dos canais — referência única para rótulos em todos os módulos
NOMES_CANAIS = {
    'CH2': 'CH2 — Radial Vertical',
    'CH3': 'CH3 — Radial Horizontal',
    'CH4': 'CH4 — Axial',
}
NOMES_CANAIS_CURTO = {
    'CH2': 'Radial Vert.',
    'CH3': 'Radial Horiz.',
    'CH4': 'Axial',
}

# Unidade de saída dos acelerômetros — atualizado pelo módulo de calibração
# Usado por funções de plot para rotular o eixo Y corretamente
_UNIDADE: dict = {'saida': 'V'}   # 'V' | 'g' | 'm/s²'


def get_ylabel_amp() -> str:
    """Retorna o rótulo do eixo de amplitude com a unidade atual."""
    u = _UNIDADE['saida']
    return f'Amplitude ({u} RMS)' if u != 'V' else 'Amplitude RMS (V)'


def get_ylabel_tempo() -> str:
    """Retorna o rótulo do eixo de amplitude temporal com a unidade atual."""
    u = _UNIDADE['saida']
    return f'Aceleração ({u})' if u != 'V' else 'Amplitude (V)'


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — UTILITÁRIOS GERAIS
# ═════════════════════════════════════════════════════════════════════════════

def cabecalho(titulo, largura=76, cor='='):
    """Imprime cabeçalho formatado no terminal."""
    linha = cor * largura
    print(f"\n{linha}")
    print(f"  {titulo}")
    print(linha)


def box(titulo, largura=76):
    """Caixa decorativa."""
    print("\n╔" + "═" * (largura - 2) + "╗")
    print("║  " + titulo.ljust(largura - 4) + "║")
    print("╚" + "═" * (largura - 2) + "╝")


def pausa(msg="  ▶  Pressione ENTER para continuar..."):
    input(f"\n{msg}")


def escolha_menu(opcoes: list, titulo: str = "Escolha uma opção") -> int:
    """Menu interativo numerado; retorna índice 0-based."""
    print(f"\n  {titulo}:")
    for i, op in enumerate(opcoes, 1):
        print(f"   [{i}] {op}")
    while True:
        try:
            val = int(input("  → "))
            if 1 <= val <= len(opcoes):
                return val - 1
            print(f"  Digite entre 1 e {len(opcoes)}")
        except ValueError:
            print("  Entrada inválida.")


def entrada_float(mensagem, padrao=None, minval=None, maxval=None) -> float:
    """Lê float do usuário com validação."""
    while True:
        try:
            txt = input(f"  {mensagem}" + (f" [{padrao}]: " if padrao is not None else ": "))
            if txt.strip() == "" and padrao is not None:
                return float(padrao)
            val = float(txt.replace(",", "."))
            if minval is not None and val < minval:
                print(f"  Mínimo: {minval}")
                continue
            if maxval is not None and val > maxval:
                print(f"  Máximo: {maxval}")
                continue
            return val
        except ValueError:
            print("  Digite um número válido.")


def entrada_int(mensagem, padrao=None, minval=None, maxval=None) -> int:
    """Lê inteiro do usuário com validação."""
    while True:
        try:
            txt = input(f"  {mensagem}" + (f" [{padrao}]: " if padrao is not None else ": "))
            if txt.strip() == "" and padrao is not None:
                return int(padrao)
            val = int(txt)
            if minval is not None and val < minval:
                print(f"  Mínimo: {minval}")
                continue
            if maxval is not None and val > maxval:
                print(f"  Máximo: {maxval}")
                continue
            return val
        except ValueError:
            print("  Digite um número inteiro válido.")


def salvar_figura(fig, prefixo="fig"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = f"{prefixo}_{ts}.png"
    fig.savefig(nome, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  📁 Figura salva: {nome}")
    return nome


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CARREGAMENTO DE DADOS
# ═════════════════════════════════════════════════════════════════════════════

def carregar_csv(caminho: str) -> pd.DataFrame | None:
    """
    Lê o arquivo CSV do Viking VK701H (ou formato compatível).
    Colunas esperadas: Time, CH1, CH2, CH3, CH4
    """
    cabecalho("CARREGAMENTO DOS DADOS")

    if not os.path.exists(caminho):
        print(f"  [ERRO] Arquivo não encontrado: {caminho}")
        return None

    try:
        df = pd.read_csv(caminho, index_col=False)
        # Aceitar arquivos com coluna extra vazia no final (trailing comma)
        df = df.iloc[:, :5]
        df.columns = ['Time', 'CH1', 'CH2', 'CH3', 'CH4']

        # Forçar numérico nos canais (coluna Time é ignorada para Fs)
        for ch in ['CH1', 'CH2', 'CH3', 'CH4']:
            df[ch] = pd.to_numeric(df[ch], errors='coerce')

        # Remover linhas inválidas
        n_antes = len(df)
        df.dropna(subset=['CH1', 'CH2', 'CH3', 'CH4'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        n_depois = len(df)

        print(f"  ✓ Arquivo lido: {os.path.basename(caminho)}")
        print(f"  ✓ Amostras: {n_depois}  (removidos {n_antes - n_depois} inválidos)")
        print(f"  ✓ Coluna Time não usada para Fs (informe manualmente abaixo).")

        # Fs sempre solicitado ao usuário
        fs_detectado = None

        # ── Remover componente DC dos acelerômetros (CH2, CH3, CH4) ──────────
        for ch_dc in ['CH2', 'CH3', 'CH4']:
            media_dc = df[ch_dc].mean()
            df[ch_dc] = df[ch_dc] - media_dc
        print("  ✓ Componente DC removida dos canais CH2, CH3, CH4")

        # Estatísticas rápidas
        print("\n  Resumo dos canais:")
        print(f"  {'Canal':<10} {'Mín':>12} {'Máx':>12} {'Média':>12} {'RMS':>12}")
        print("  " + "─" * 60)
        for ch in ['CH1', 'CH2', 'CH3', 'CH4']:
            v = df[ch].values
            rms = np.sqrt(np.mean(v**2))
            print(f"  {ch:<10} {v.min():>12.5f} {v.max():>12.5f} {v.mean():>12.5f} {rms:>12.5f}")

        return df, fs_detectado

    except Exception as e:
        print(f"  [ERRO] {e}")
        traceback.print_exc()
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CONCEITOS DIDÁTICOS DE AMOSTRAGEM
# ═════════════════════════════════════════════════════════════════════════════

CONSTANTE_NYQUIST = 2.56   # Padrão de coletores de vibração (SKF, CSI, Emerson…)


def aula_amostragem(fs: float, n_amostras: int):
    """
    Módulo educacional: explica os parâmetros de aquisição e seus impactos.
    Usa a constante 2,56 padrão de instrumentos de vibração.
    """
    box("MÓDULO DIDÁTICO 1 — PARÂMETROS DE AQUISIÇÃO E TEOREMA DE NYQUIST")

    duracao        = n_amostras / fs
    fmax_teorico   = fs / 2.0                         # Nyquist teórico
    fmax_pratico   = fs / CONSTANTE_NYQUIST           # limite real dos coletores
    resolucao_freq = fs / n_amostras                  # Δf = 1 / T_total
    linhas_fft     = int(n_amostras / CONSTANTE_NYQUIST)  # linhas espectrais úteis

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │                    SEUS DADOS DE AQUISIÇÃO                       │
  ├──────────────────────────────────────────────────────────────────┤
  │  Frequência de amostragem (Fs)    = {fs:>10.1f} Hz               │
  │  Número de amostras (N)           = {n_amostras:>10d} pontos      │
  │  Duração total da aquisição       = {duracao:>10.3f} s            │
  │  Intervalo entre amostras (Δt)    = {1/fs*1000:>10.4f} ms        │
  │                                                                   │
  │  ── LIMITES ESPECTRAIS ───────────────────────────────────────── │
  │  Nyquist teórico  (Fs / 2,00)     = {fmax_teorico:>10.1f} Hz     │
  │  Fmax PRÁTICO     (Fs / 2,56)  ★  = {fmax_pratico:>10.1f} Hz     │
  │  Resolução espectral (Δf = Fs/N)  = {resolucao_freq:>10.4f} Hz   │
  │  Linhas espectrais úteis          = {linhas_fft:>10d} linhas      │
  └──────────────────────────────────────────────────────────────────┘
""")

    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║         POR QUE A CONSTANTE 2,56?                               ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  Coletores de vibração (SKF Microlog, CSI 2140, Emerson AMS…)    ║
  ║  usam filtros anti-aliasing ANALÓGICOS antes do ADC.             ║
  ║  Esses filtros atenuam a partir de ~80% de Fs/2.                 ║
  ║                                                                  ║
  ║  Para garantir que o sinal esteja bem dentro da faixa plana      ║
  ║  do filtro, o padrão industrial define:                          ║
  ║                                                                  ║
  ║       Fmax = Fs / 2,56                                           ║
  ║                                                                  ║
  ║  Consequência prática:                                           ║
  ║    → Para medir até Fmax = {fmax_pratico:>6.0f} Hz               ║
  ║      é necessário Fs = {fmax_pratico*2.56:>6.0f}                 ║
  ║     Hz (= {fmax_pratico:.0f} × 2,56)                             ║
  ║                                                                  ║
  ║  ⚠ Se f > Fmax → ALIASING (frequência fantasma no espectro)     ║
  ║                                                                  ║
  ║  Relação com linhas de resolução (L):                            ║
  ║       N = L × 2,56   (ex: 400 linhas → N = 1024 amostras)        ║
  ╚══════════════════════════════════════════════════════════════════╝
""")

    print("""
  RESOLUÇÃO ESPECTRAL (Δf):
  ─────────────────────────
    Δf = Fs / N  =  1 / T_total  =  Fmax / L

    → Aumentar o TEMPO de coleta  →  melhor resolução em frequência
    → Aumentar Fs (mantendo N)    →  piora a resolução
    → Dobrar N (mantendo Fs)      →  melhora Δf em 2×

  TABELA PADRÃO DE COLETORES (Fs → Fmax → Linhas a Δf=Fs/N):
  ─────────────────────────────────────────────────────────
    Fs = 2.560 Hz  → Fmax =  1.000 Hz  → 400L @ Δf=6,40 Hz / 1600L @ Δf=1,60 Hz
    Fs = 5.120 Hz  → Fmax =  2.000 Hz  → 400L @ Δf=12,80Hz / 1600L @ Δf=3,20 Hz
    Fs =12.800 Hz  → Fmax =  5.000 Hz  → 400L                / 1600L
    Fs =25.600 Hz  → Fmax = 10.000 Hz  → 400L                / 1600L

  JANELAS (Windowing):
  ─────────────────────────
    Retangular : máxima resolução, alto VAZAMENTO espectral
    Hann / Hamming: equilíbrio — RECOMENDADA para máquinas rotativas
    Flat-top   : amplitude muito precisa, baixa resolução
    Blackman   : mínimo vazamento, pior resolução
""")

    # Verificação prática para a máquina em análise
    print("  VERIFICAÇÃO RÁPIDA PARA SUA MÁQUINA:")
    print("  ─────────────────────────────────────")
    rpm_exemplo = entrada_float("  RPM aproximado da máquina (0 = pular)", 0, 0, 20000)

    if rpm_exemplo > 0:
        freq_1x   = rpm_exemplo / 60
        freq_10x  = 10 * freq_1x
        fs_minimo = freq_10x * CONSTANTE_NYQUIST   # Fs = Fmax × 2,56

        print(f"\n    1X (rotação)             = {freq_1x:.2f} Hz")
        print(f"    Fmax desejado (até 10X)  = {freq_10x:.2f} Hz")
        print(f"    Fs mínimo (Fmax × 2,56)  = {fs_minimo:.1f} Hz")

        if fs >= fs_minimo:
            print(f"    ✓ Fs={fs:.0f} Hz cobre as 10 primeiras harmônicas com margem.")
        else:
            n_harm_ok = int(fmax_pratico / freq_1x)
            print(f"    ⚠ Fs={fs:.0f} Hz insuficiente para 10X! "
                  f"Cobre apenas até {n_harm_ok}X = {n_harm_ok*freq_1x:.1f} Hz")
            print(f"    → Recomendado: Fs ≥ {fs_minimo:.0f} Hz")

    pausa()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3b — DEMONSTRAÇÃO INTERATIVA DO ALIASING
# ═════════════════════════════════════════════════════════════════════════════

def demo_aliasing(fs_dados: float):
    """
    Demonstração visual interativa do fenômeno de aliasing.
    O aluno escolhe a frequência do sinal e a taxa de subamostragem
    e observa a 'frequência fantasma' que aparece no espectro.
    """
    box("MÓDULO DIDÁTICO 1b — DEMONSTRAÇÃO DO ALIASING")

    print("""
  O ALIASING ocorre quando um sinal é amostrado com Fs < 2 × f_sinal.
  O sinal aparece no espectro em uma FREQUÊNCIA ERRADA (fantasma).

  Fórmula da frequência alias:
      f_alias = | f_sinal - k × Fs |   (para k inteiro que minimiza o resultado)

  Este módulo gera um sinal sintético e mostra o que acontece
  quando você SUBAMOSTRA (reduz o Fs artificialmente).
""")

    # ── Parâmetros do sinal sintético ────────────────────────────────────────
    fmax_atual = fs_dados / CONSTANTE_NYQUIST
    print(f"  Fmax atual do seu equipamento: {fmax_atual:.1f} Hz  (Fs={fs_dados:.0f} Hz)")
    print()

    f_sinal  = entrada_float(
        "  Frequência do sinal de teste (Hz)",
        padrao=min(fmax_atual * 0.6, 100), minval=1, maxval=fmax_atual * 2)
    amp      = entrada_float("  Amplitude do sinal", padrao=1.0, minval=0.01)
    fs_sub_factor = entrada_float(
        "  Fator de subamostragem (ex: 2 = divide Fs por 2)",
        padrao=3.0, minval=1.1, maxval=10.0)

    fs_sub    = fs_dados / fs_sub_factor
    fmax_sub  = fs_sub / CONSTANTE_NYQUIST
    duracao   = 1.0        # 1 segundo de sinal sintético
    N_orig    = int(fs_dados * duracao)

    # Sinal contínuo de referência (alta resolução)
    t_orig    = np.linspace(0, duracao, N_orig, endpoint=False)
    sinal_orig = amp * np.sin(2 * np.pi * f_sinal * t_orig)

    # Sinal amostrado corretamente (Fs original)
    sinal_bem_amostrado = sinal_orig.copy()

    # Sinal subamosstrado: pegar 1 de cada N amostras
    passo     = max(1, int(round(fs_sub_factor)))
    t_sub     = t_orig[::passo]
    sinal_sub = sinal_orig[::passo]
    fs_efetivo = len(t_sub) / duracao   # Fs real após downsampling

    # Frequência alias esperada
    k = round(f_sinal / fs_efetivo)
    f_alias = abs(f_sinal - k * fs_efetivo)
    fmax_sub_real = fs_efetivo / 2.0

    print(f"""
  ─────────────────────────────────────────────────────────────────
  RESULTADO ESPERADO:
    Sinal original          : {f_sinal:.2f} Hz  (Fs={fs_dados:.0f} Hz → OK)
    Fs subamostrado         : {fs_efetivo:.1f} Hz
    Fmax subamostrado       : {fmax_sub_real:.1f} Hz  (Fs/2)
    Frequência ALIAS        : {f_alias:.2f} Hz  ← fantasma no espectro!
    {('⚠ ALIASING! f_sinal > Fmax_sub → fantasma aparece em ' + f'{f_alias:.1f} Hz')
      if f_sinal > fmax_sub_real else '✓ Sem aliasing: f_sinal ≤ Fmax_sub'}
  ─────────────────────────────────────────────────────────────────
""")

    # ── Gráfico ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    gs_fig = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # --- Linha 1: Sinais no tempo ---
    # Sinal bem amostrado
    ax1 = fig.add_subplot(gs_fig[0, 0])
    t_plot = t_orig[:min(int(0.05 * fs_dados), len(t_orig))]  # primeiros 50 ms
    ax1.plot(t_plot * 1000, sinal_bem_amostrado[:len(t_plot)],
             color=COR_CH2, lw=2, label=f'Sinal bem amostrado\nFs={fs_dados:.0f} Hz')
    ax1.set_title(f'✓ Bem Amostrado — Fs={fs_dados:.0f} Hz', color=COR_OK, fontsize=11, fontweight='bold')
    ax1.set_xlabel('Tempo (ms)'); ax1.set_ylabel(get_ylabel_tempo())
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Sinal subamostrado
    ax2 = fig.add_subplot(gs_fig[0, 1])
    mask_sub = t_sub <= t_plot[-1]
    # Reconstrução aparente do sinal subamostrado
    t_rec = np.linspace(0, t_plot[-1], 500)
    if np.sum(mask_sub) >= 2:
        sinal_rec = np.interp(t_rec, t_sub[mask_sub], sinal_sub[mask_sub])
        ax2.plot(t_rec * 1000, sinal_rec, color=COR_CH3, lw=2,
                 label=f'Sinal reconstruído\n(aparência enganosa!)')
    ax2.plot(t_sub[mask_sub] * 1000, sinal_sub[mask_sub],
             'o', color=COR_DEST, ms=8, label=f'Amostras (Fs≈{fs_efetivo:.0f} Hz)')
    # Sinal original em cinza para comparação
    ax2.plot(t_plot * 1000, sinal_bem_amostrado[:len(t_plot)],
             color='white', lw=1, alpha=0.3, label='Original (ref.)')
    cor_titulo = '#ff4444' if f_sinal > fmax_sub_real else COR_OK
    ax2.set_title(f'⚠ SUBAMOSTRADO — Fs≈{fs_efetivo:.0f} Hz', color=cor_titulo, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Tempo (ms)'); ax2.set_ylabel(get_ylabel_tempo())
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # --- Linha 2: Espectros ---
    # Espectro do sinal bem amostrado
    ax3 = fig.add_subplot(gs_fig[1, 0])
    freqs_b, amp_b = calcular_espectro(sinal_bem_amostrado, fs_dados, 'Hann', 'hann')
    ax3.plot(freqs_b, amp_b, color=COR_CH2, lw=1.5)
    ax3.axvline(f_sinal, color=COR_OK, ls='--', lw=3,
                label=f'f real = {f_sinal:.1f} Hz')
    ax3.axvline(fs_dados / CONSTANTE_NYQUIST, color='gray', ls=':', lw=2,
                label=f'Fmax = {fs_dados/CONSTANTE_NYQUIST:.0f} Hz')
    ax3.set_xlim([0, min(fs_dados / 2, f_sinal * 3 + 50)])
    ax3.set_title('Espectro — Bem Amostrado', color=COR_OK, fontsize=11, fontweight='bold')
    ax3.set_xlabel('Frequência (Hz)'); ax3.set_ylabel(get_ylabel_tempo())
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # Espectro do sinal subamostrado
    ax4 = fig.add_subplot(gs_fig[1, 1])
    if len(sinal_sub) > 10:
        freqs_s, amp_s = calcular_espectro(sinal_sub, fs_efetivo, 'Hann', 'hann')
        ax4.plot(freqs_s, amp_s, color=COR_CH3, lw=1.5)
        ax4.axvline(f_sinal, color=COR_OK, ls='--', lw=2, alpha=0.5,
                    label=f'f real = {f_sinal:.1f} Hz (ref.)')
        ax4.axvline(f_alias, color='#ff4444', ls='-', lw=3,
                    label=f'ALIAS = {f_alias:.1f} Hz ← FANTASMA!')
        ax4.axvline(fs_efetivo / 2, color='gray', ls=':', lw=2,
                    label=f'Fmax = {fs_efetivo/2:.0f} Hz')
        ax4.set_xlim([0, fs_efetivo / 2])
    cor_titulo2 = '#ff4444' if f_sinal > fmax_sub_real else COR_OK
    ax4.set_title('Espectro — Subamostrado  ⚠ ALIAS!', color=cor_titulo2, fontsize=11, fontweight='bold')
    ax4.set_xlabel('Frequência (Hz)'); ax4.set_ylabel(get_ylabel_tempo())
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    # --- Linha 3: Espelho de Nyquist estendido ---
    ax5 = fig.add_subplot(gs_fig[2, :])

    # Estender o eixo X para cobrir f_sinal mesmo se > Fs
    # Mostramos pelo menos 2 períodos do espelho para f_sinal > Fs
    x_max = max(fs_efetivo * 1.05,
                f_sinal * 1.15,
                fs_efetivo * (int(f_sinal / fs_efetivo) + 1) * 1.05)
    f_range = np.linspace(0, x_max, 4000)

    # Curva do espelho: dobramento periódico em Fs/2
    alias_curve = np.abs(((f_range + fs_efetivo / 2) % fs_efetivo) - fs_efetivo / 2)

    ax5.plot(f_range, alias_curve, color=COR_CH2, lw=2.5,
             label='Frequência no espectro = |f mod Fs|')

    # Linhas de Nyquist em cada múltiplo de Fs/2
    n_periods = int(x_max / (fs_efetivo / 2)) + 1
    for i in range(1, n_periods + 1):
        fx = i * fs_efetivo / 2
        if fx <= x_max:
            ax5.axvline(fx, color='gray', ls='--', lw=1.2, alpha=0.6,
                        label=f'Nyquist = {fs_efetivo/2:.0f} Hz' if i == 1 else '')

    # Zonas coloridas para cada período
    for i in range(n_periods):
        xL = i * fs_efetivo / 2
        xR = min((i + 1) * fs_efetivo / 2, x_max)
        cor_zona = 'green' if i % 2 == 0 else 'red'
        label_zona = ('Zona segura' if (i == 0) else
                      'Zona aliasing' if (i == 1) else '')
        ax5.fill_betweenx([0, fs_efetivo / 2], xL, xR,
                          alpha=0.07, color=cor_zona,
                          label=label_zona)

    # Linha vertical: frequência do sinal (verde)
    ax5.axvline(f_sinal, color=COR_OK, ls='--', lw=2.5,
                label=f'f_sinal = {f_sinal:.1f} Hz',
                zorder=4)

    # Linha vertical: frequência alias (laranja — cor diferente de f_sinal)
    COR_ALIAS = '#ff9900'
    ax5.axvline(f_alias, color=COR_ALIAS, ls='-.', lw=2.5,
                label=f'f_alias = {f_alias:.1f} Hz  ← fantasma',
                zorder=4)

    # Linha horizontal tracejada de f_alias
    ax5.axhline(f_alias, color=COR_ALIAS, ls=':', lw=1.5, alpha=0.7)

    # Ponto de interseção: onde f_sinal toca a curva
    ax5.plot(f_sinal, f_alias, 'o', color='#ff4444', ms=14,
             zorder=6, label='Ponto de mapeamento')

    # Seta guia: percurso visual do f_sinal → curva → f_alias
    ax5.annotate('',
                 xy=(f_alias + fs_efetivo * 0.02, f_alias),
                 xytext=(f_sinal, f_alias),
                 arrowprops=dict(arrowstyle='->', color=COR_ALIAS,
                                 lw=2, connectionstyle='arc3,rad=0'))
    ax5.text(max(f_alias + 5, 5), f_alias * 1.05,
             f'aparece em {f_alias:.1f} Hz',
             color=COR_ALIAS, fontsize=10, fontweight='bold', va='bottom')

    # Se f_sinal > Fs, adicionar nota explicativa
    if f_sinal > fs_efetivo:
        ax5.text(f_sinal * 0.5, fs_efetivo / 2 * 0.85,
                 f'⚠ f_sinal={f_sinal:.0f} Hz > Fs={fs_efetivo:.0f} Hz\n'
                 f'   O sinal passa por {int(f_sinal//(fs_efetivo/2))} espelhos!',
                 color='#ff6666', fontsize=9,
                 bbox=dict(facecolor='#2a0000', alpha=0.7, lw=1))

    ax5.set_xlim([0, x_max])
    ax5.set_ylim([0, fs_efetivo / 2 * 1.15])
    ax5.set_xlabel('Frequência do sinal real (Hz)', fontsize=11)
    ax5.set_ylabel('Frequência no espectro medido (Hz)', fontsize=11)
    ax5.set_title('Espelho de Nyquist — Mapeamento de Alias  '
                  '(verde=zona segura | vermelho=zona aliasing)',
                  color='white', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, ncol=4, loc='upper right')
    ax5.grid(True, alpha=0.3)

    fig.suptitle(
        f'ALIASING — f_sinal={f_sinal:.1f} Hz | Fs_sub≈{fs_efetivo:.0f} Hz | '
        f'ALIAS={f_alias:.1f} Hz',
        fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, 'demo_aliasing')
    plt.show()

    # Permitir exploração interativa com diferentes frequências
    while input("\n  Testar outra frequência? [s/N]: ").strip().lower() == 's':
        f_sinal2  = entrada_float("  Nova frequência do sinal (Hz)", minval=1, maxval=fmax_atual * 2)
        fmax_sub2 = fs_efetivo / 2.0
        k2        = round(f_sinal2 / fs_efetivo)
        f_alias2  = abs(f_sinal2 - k2 * fs_efetivo)
        status    = '⚠ ALIASING' if f_sinal2 > fmax_sub2 else '✓ sem alias'
        print(f"    f_sinal={f_sinal2:.1f} Hz  →  alias={f_alias2:.1f} Hz  [{status}]")


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — DETECÇÃO DE RPM PELO TACÔMETRO (CH1)
# ═════════════════════════════════════════════════════════════════════════════

def detectar_rpm(sinal_taco: np.ndarray, fs: float) -> tuple:
    """
    Detecção robusta de pulsos do tacômetro por periodicidade.
    Retorna: (indices_pulsos, rpm, diagnostico_dict)
    """
    cabecalho("ANÁLISE DO TACÔMETRO (CH1) — EXTRAÇÃO DE RPM")

    media = np.mean(sinal_taco)
    desvio = np.std(sinal_taco)
    if desvio == 0:
        print("  [ERRO] Sinal do tacômetro constante!")
        return None, None, None

    sn = (sinal_taco - media) / desvio  # normalizado

    dist_min = int(0.015 * fs)  # ≥ 15 ms entre pulsos (≤ 4000 RPM)

    print("  Testando detecção de pulsos positivos e negativos...")
    todos_candidatos = []

    for sinal_mod, tipo in [(sn, 'POSITIVO'), (-sn, 'NEGATIVO')]:
        for h in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            picos, props = signal.find_peaks(sinal_mod, height=h,
                                             distance=dist_min, prominence=0.2)
            if len(picos) > 5:
                per = np.diff(picos) / fs
                rpm_c = 60 / np.mean(per)
                cv = np.std(per) / np.mean(per) * 100
                todos_candidatos.append({
                    'tipo': tipo, 'threshold': h, 'picos': picos,
                    'n': len(picos), 'rpm': rpm_c, 'cv': cv,
                    'periodos': per,
                })

    validos = [c for c in todos_candidatos
               if 200 <= c['rpm'] <= 10000 and c['cv'] < 15 and c['n'] >= 8]

    if not validos:
        print("  [ERRO] Nenhum padrão periódico encontrado no tacômetro!")
        print("         Verifique o sinal do CH1.")
        return None, None, None

    melhor = min(validos, key=lambda x: x['cv'])
    tipo_pulso = melhor['tipo']
    indices_pulsos = melhor['picos']

    # Filtrar espúrios
    if len(indices_pulsos) > 2:
        per_med = np.median(np.diff(indices_pulsos))
        filt = [indices_pulsos[0]]
        for idx in indices_pulsos[1:]:
            if idx - filt[-1] > 0.4 * per_med:
                filt.append(idx)
        indices_pulsos = np.array(filt)

    periodos = np.diff(indices_pulsos) / fs
    rpm_final = 60 / np.mean(periodos)
    cv_final  = np.std(periodos) / np.mean(periodos) * 100

    print(f"\n  ✓ Tipo de pulso      : {tipo_pulso}")
    print(f"  ✓ Pulsos detectados  : {len(indices_pulsos)}")
    print(f"  ✓ RPM médio          : {rpm_final:.2f} RPM")
    print(f"  ✓ Frequência         : {rpm_final/60:.3f} Hz")
    print(f"  ✓ Período médio      : {np.mean(periodos)*1000:.2f} ms")
    print(f"  ✓ Estabilidade (CV)  : ±{cv_final:.2f}%")

    if cv_final < 1:
        print("  ★ QUALIDADE: EXCELENTE")
    elif cv_final < 2:
        print("  ★ QUALIDADE: MUITO BOA")
    elif cv_final < 5:
        print("  ★ QUALIDADE: BOA")
    else:
        print("  ⚠ QUALIDADE: ACEITÁVEL — rotação instável?")

    diagnostico = {
        'tipo_pulso': tipo_pulso,
        'periodos': periodos,
        'cv': cv_final,
        'sn': sn,
        'melhor': melhor,
    }

    # ── Gráfico de diagnóstico do tacômetro ──────────────────────────────
    _plot_tacometro(sinal_taco, sn, indices_pulsos, periodos, rpm_final, fs, tipo_pulso)

    return indices_pulsos, rpm_final, diagnostico


def _plot_tacometro(sinal_taco, sn, indices, periodos, rpm, fs, tipo_pulso):
    t = np.arange(len(sinal_taco)) / fs
    fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                             facecolor='#1a1a2e')
    fig.suptitle(f'TACÔMETRO (CH1) — RPM: {rpm:.2f} | {len(indices)} pulsos',
                 fontsize=14, fontweight='bold', color='white')

    # Sinal completo
    ax = axes[0, 0]
    ax.plot(t, sinal_taco, color=COR_TACO, lw=0.6, alpha=0.8)
    for idx in indices:
        ax.axvline(idx / fs, color='red', lw=1.2, alpha=0.5)
    ax.set_title('Sinal do Tacômetro com pulsos marcados', color='white')
    ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Tensão (V)')
    ax.grid(True, alpha=0.3)

    # Zoom em 5 pulsos
    ax = axes[0, 1]
    if len(indices) >= 5:
        i0 = max(0, indices[0] - int(0.05 * fs))
        i1 = min(len(t) - 1, indices[4] + int(0.05 * fs))
        ax.plot(t[i0:i1], sinal_taco[i0:i1], color=COR_TACO, lw=1.5)
        for k, idx in enumerate(indices[:5]):
            ax.axvline(idx / fs, color='red', lw=2.5, ls='--')
            ax.text(idx / fs, sinal_taco[i0:i1].max() * 0.92,
                    f'P{k+1}', ha='center', color='red', fontsize=9, fontweight='bold')
    ax.set_title('Zoom: 5 primeiros pulsos', color='white')
    ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Tensão (V)')
    ax.grid(True, alpha=0.3)

    # Variação de período
    ax = axes[1, 0]
    per_ms = periodos * 1000
    ax.plot(per_ms, 'o-', color=COR_OK, lw=2, ms=5)
    ax.axhline(np.mean(per_ms), color='red', ls='--', lw=2.5,
               label=f'Média: {np.mean(per_ms):.2f} ms')
    ax.axhline(np.mean(per_ms) * 1.05, color=COR_DEST, ls=':', lw=1.5, label='±5%')
    ax.axhline(np.mean(per_ms) * 0.95, color=COR_DEST, ls=':', lw=1.5)
    ax.set_title('Estabilidade da Rotação (período entre pulsos)', color='white')
    ax.set_xlabel('Intervalo #'); ax.set_ylabel('Período (ms)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Histograma de períodos
    ax = axes[1, 1]
    ax.hist(per_ms, bins=min(30, len(per_ms) // 2 + 1),
            color=COR_CH2, edgecolor='black', alpha=0.8)
    ax.axvline(np.mean(per_ms), color='red', ls='--', lw=2.5,
               label=f'Média: {np.mean(per_ms):.2f} ms')
    ax.axvline(np.median(per_ms), color='yellow', ls='--', lw=2,
               label=f'Mediana: {np.median(per_ms):.2f} ms')
    ax.set_title('Distribuição dos Períodos', color='white')
    ax.set_xlabel('Período (ms)'); ax.set_ylabel('Contagem')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    salvar_figura(fig, 'tacometro')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — REFERÊNCIA DE FASE ANGULAR
# ═════════════════════════════════════════════════════════════════════════════

def criar_fase_referencia(n_amostras: int, indices_pulsos: np.ndarray, fs: float) -> np.ndarray:
    """Interpola ângulo 0–360° continuamente entre os pulsos do tacômetro."""
    angulos = np.arange(len(indices_pulsos)) * 360.0
    tempos  = indices_pulsos / fs
    t_total = np.arange(n_amostras) / fs
    f_interp = interp1d(tempos, angulos, kind='linear', fill_value='extrapolate')
    return f_interp(t_total) % 360.0


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — ANÁLISE ESPECTRAL (FFT) COM JANELAS
# ═════════════════════════════════════════════════════════════════════════════

JANELAS_DISPONIVEIS = {
    '1': ('Retangular',  'boxcar',   'Máx. resolução, alto vazamento. Bom para sinais síncronos (rotacionais).'),
    '2': ('Hann',        'hann',     'Equilíbrio excelente. RECOMENDADA para análise de vibrações em geral.'),
    '3': ('Hamming',     'hamming',  'Similar ao Hann, lóbulos laterais um pouco maiores.'),
    '4': ('Flat-Top',    'flattop',  'Amplitude muito precisa, baixa resolução. Ideal para calibração.'),
    '5': ('Blackman',    'blackman', 'Mínimo vazamento, menor resolução. Bom para sinais de larga banda.'),
}


def selecionar_janela() -> tuple:
    """Menu interativo de escolha de janela."""
    print("\n  JANELAS DE PONDERAÇÃO DISPONÍVEIS:")
    print("  " + "─" * 70)
    for k, (nome, scipy_name, descricao) in JANELAS_DISPONIVEIS.items():
        print(f"  [{k}] {nome:<12}  — {descricao}")
    print("  " + "─" * 70)
    while True:
        op = input("  Escolha [1-5] (padrão=2): ").strip()
        if op == "":
            op = "2"
        if op in JANELAS_DISPONIVEIS:
            nome, scipy_name, _ = JANELAS_DISPONIVEIS[op]
            print(f"  ✓ Janela selecionada: {nome}")
            return nome, scipy_name
        print("  Opção inválida.")


def selecionar_escala() -> str:
    """Menu interativo de escolha de escala do eixo Y do espectro."""
    print("""
  ESCALA DO EIXO DE AMPLITUDE:
  ─────────────────────────────────────────────────────────────────────
  [1] Linear   — Mostra amplitudes absolutas. Facilita comparar picos
                 próximos em magnitude. Picos pequenos ficam escondidos.
  [2] Log (dB) — Escala logarítmica. Revela picos de baixa amplitude
                 e componentes de rolamento / ruído de fundo.
                 RECOMENDADA para diagnóstico de falhas.
  ─────────────────────────────────────────────────────────────────────""")
    while True:
        op = input("  Escolha [1/2] (padrão=1 linear): ").strip()
        if op == "" or op == "1":
            print("  ✓ Escala: Linear")
            return 'linear'
        if op == "2":
            print("  ✓ Escala: Logarítmica")
            return 'log'
        print("  Digite 1 ou 2.")


def selecionar_eixo_x() -> str:
    """Menu de escolha de eixo X do espectro: Hz ou Ordem."""
    print("""
  EIXO X DO ESPECTRO:
  ─────────────────────────────────────────────────────────────────────
  [1] Hz (frequência absoluta) — padrão industrial. Permite identificar
      diretamente BPF, frequências de rolamento e outros componentes
      não-síncronos com a rotação.
  [2] Ordem (× RPM)           — normaliza pela rotação. Facilita
      comparar espectros medidos em diferentes rotações e identificar
      harmônicas de desbalanceamento/desalinhamento (1X, 2X, 3X...).
      Útil quando o RPM varia.
  ─────────────────────────────────────────────────────────────────────""")
    while True:
        op = input("  Escolha [1/2] (padrão=1 Hz): ").strip()
        if op == "" or op == "1":
            print("  ✓ Eixo X: Hz")
            return 'hz'
        if op == "2":
            print("  ✓ Eixo X: Ordem")
            return 'ordem'
        print("  Digite 1 ou 2.")


def _aplicar_escala(ax, escala: str, amp: np.ndarray):
    """Aplica escala linear ou log ao eixo Y já plotado."""
    if escala == 'log':
        ax.set_yscale('log')
        ax.set_ylabel(get_ylabel_amp() + ' (log)', fontsize=10)
    else:
        ax.set_yscale('linear')
        ax.set_ylabel(get_ylabel_amp(), fontsize=10)


def calcular_espectro(sinal: np.ndarray, fs: float, nome_janela: str, scipy_janela: str):
    """
    Calcula o espectro de amplitude (rms) com a janela escolhida.
    Retorna: (freqs, amplitude_rms, fator_amplitude)
    """
    N = len(sinal)
    win = signal.get_window(scipy_janela, N)
    # Fator de correção de amplitude (COLA)
    fator = N / np.sum(win)

    yf = rfft(sinal * win)
    freqs = rfftfreq(N, d=1.0 / fs)

    # Espectro de amplitude RMS
    amp = (2.0 / N) * np.abs(yf) * fator
    amp[0] /= 2      # componente DC não dobra
    amp[-1] /= 2     # Nyquist não dobra
    amp_rms = amp / np.sqrt(2)

    return freqs, amp_rms


def analise_espectral_interativa(sinais_acel: dict, fs: float, rpm: float, indices_pulsos: np.ndarray):
    """
    Módulo interativo completo de análise espectral.
    sinais_acel = {'CH2': array, 'CH3': array, 'CH4': array}
    """
    box("MÓDULO DIDÁTICO 2 — ANÁLISE ESPECTRAL (FFT)")

    print("""
  A FFT decompõe o sinal no domínio da FREQUÊNCIA.
  Cada "pico" no espectro representa uma frequência de vibração presente.

  Componentes típicas em máquinas rotativas:
    • 1X   → 1 vez a rotação (desbalanceamento, excentricidade)
    • 2X   → 2 vezes (desalinhamento, folga)
    • 3X+  → harmônicas superiores (diversas causas)
    • Sub-harmônicas (0,5X): folga severa, instabilidade de mancal
""")

    # Escolha de canal
    idx_canal = escolha_menu(
        ['CH2 — Radial Vertical',
         'CH3 — Radial Horizontal',
         'CH4 — Axial',
         'Todos os canais sobrepostos'],
        "Canal para análise espectral"
    )
    nomes_canais = ['CH2', 'CH3', 'CH4', 'TODOS']
    canal_selecionado = nomes_canais[idx_canal]

    # Escolha de janela
    nome_jan, scipy_jan = selecionar_janela()

    # Escolha de escala
    escala = selecionar_escala()

    # Eixo X: Hz ou Ordem
    eixo_x = selecionar_eixo_x()

    # Faixa: em Hz ou em ordens dependendo da escolha
    freq_rot = rpm / 60.0
    fmax_hz_nyq = fs / 2
    n_ordens_max = fmax_hz_nyq / freq_rot   # ordens máximas possíveis

    if eixo_x == 'hz':
        fmax_plot = entrada_float(
            f"  Freq. máxima do gráfico (Hz) [padrão={min(fs/2,500):.0f}]",
            padrao=min(fs / 2, 500), minval=10, maxval=fs / 2)
        xmax_ordens = None   # não usado no modo Hz
    else:
        xmax_ordens = entrada_float(
            f"  Ordem máxima [padrão=20]",
            padrao=min(20.0, n_ordens_max * 0.8), minval=1, maxval=n_ordens_max)
        fmax_plot = xmax_ordens * freq_rot

    # Quantas harmônicas marcar
    n_harm = entrada_int("  Número de harmônicas a marcar (1X, 2X, ...)", padrao=5,
                         minval=1, maxval=20)

    # ── Plotagem ──────────────────────────────────────────────────────────
    offset = 1e-12 if escala == 'log' else 0.0

    def _eixo_x_vals(freqs):
        """Converte vetor de frequências para o eixo escolhido."""
        return freqs / freq_rot if eixo_x == 'ordem' else freqs

    xlabel_str = 'Ordem (× RPM)' if eixo_x == 'ordem' else 'Frequência (Hz)'
    xmax_plot  = xmax_ordens if eixo_x == 'ordem' else fmax_plot

    if canal_selecionado == 'TODOS':
        canais_plot = ['CH2', 'CH3', 'CH4']
        cores_plot  = [COR_CH2, COR_CH3, COR_CH4]
        fig, ax = plt.subplots(figsize=(18, 7), facecolor='#1a1a2e')
        for ch, cor in zip(canais_plot, cores_plot):
            if ch in sinais_acel:
                freqs, amp = calcular_espectro(sinais_acel[ch], fs, nome_jan, scipy_jan)
                ax.plot(_eixo_x_vals(freqs), amp + offset,
                        color=cor, lw=1, alpha=0.85, label=ch)
        _aplicar_escala(ax, escala, amp)
        axes_list = [ax]
    else:
        fig, ax = plt.subplots(figsize=(18, 7), facecolor='#1a1a2e')
        cor = {'CH2': COR_CH2, 'CH3': COR_CH3, 'CH4': COR_CH4}[canal_selecionado]
        freqs, amp = calcular_espectro(sinais_acel[canal_selecionado], fs, nome_jan, scipy_jan)
        ax.plot(_eixo_x_vals(freqs), amp + offset, color=cor, lw=1, label=canal_selecionado)
        _aplicar_escala(ax, escala, amp)
        axes_list = [ax]

    # Marcar harmônicas
    cores_harm = plt.cm.plasma(np.linspace(0.2, 0.9, n_harm))
    for k in range(1, n_harm + 1):
        fx_hz  = k * freq_rot
        fx_plot = k if eixo_x == 'ordem' else fx_hz
        if fx_hz <= fmax_plot:
            for ax in axes_list:
                ax.axvline(fx_plot, color=cores_harm[k - 1], ls='--', lw=1.5, alpha=0.9)
                ylims = ax.get_ylim()
                ytop = ylims[1] * 0.85 if escala == 'linear' else ylims[1] ** 0.9
                lbl = f'{k}X' if eixo_x == 'ordem' else f'{k}X\n{fx_hz:.1f}Hz'
                ax.text(fx_plot, ytop, lbl, ha='center', va='top',
                        color=cores_harm[k - 1], fontsize=8, fontweight='bold')

    escala_label = 'Log' if escala == 'log' else 'Linear'
    eixo_label   = 'Ordem' if eixo_x == 'ordem' else 'Hz'
    for ax in axes_list:
        ax.set_xlim([0, xmax_plot])
        ax.set_xlabel(xlabel_str, fontsize=11)
        ax.set_title(
            f'Espectro de Vibração — Janela: {nome_jan} | '
            f'Escala: {escala_label} | Eixo X: {eixo_label} | '
            f'RPM: {rpm:.1f} | '
            f'Δf = {fs/len(sinais_acel.get("CH2", sinais_acel.get("CH3"))):.4f} Hz',
            fontsize=11, fontweight='bold', color='white')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        # Adicionar ticks de ordem se modo ordem
        if eixo_x == 'ordem':
            ticks = np.arange(0, int(xmax_plot) + 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{t}X' for t in ticks], fontsize=8)

    plt.tight_layout()
    salvar_figura(fig, f'espectro_{canal_selecionado}')
    plt.show()

    # Comparação de janelas (opcional)
    if input("\n  Deseja comparar o efeito de DIFERENTES JANELAS? [s/N]: ").strip().lower() == 's':
        _comparar_janelas(sinais_acel, canal_selecionado if canal_selecionado != 'TODOS' else 'CH2',
                          fs, rpm, fmax_plot)


def _comparar_janelas(sinais_acel, canal, fs, rpm, fmax_plot):
    """Plot educacional comparando todas as janelas disponíveis."""
    box("COMPARAÇÃO DE JANELAS DE PONDERAÇÃO")

    print("""
  Este gráfico mostra como a escolha da janela afeta:
    • O VAZAMENTO ESPECTRAL (alargamento dos picos)
    • A AMPLITUDE APARENTE dos picos
    • A RESOLUÇÃO ESPECTRAL (separação de frequências próximas)
""")

    escala = selecionar_escala()
    offset = 1e-12 if escala == 'log' else 0.0

    sinal = sinais_acel.get(canal, list(sinais_acel.values())[0])
    nomes = [v[0] for v in JANELAS_DISPONIVEIS.values()]
    scipy_nomes = [v[1] for v in JANELAS_DISPONIVEIS.values()]
    cores = [COR_CH2, COR_CH3, COR_CH4, COR_DEST, COR_TACO]

    fig, axes = plt.subplots(len(nomes), 1, figsize=(16, 3.5 * len(nomes)),
                             facecolor='#1a1a2e', sharex=True)

    freq_rot = rpm / 60.0

    for ax, nome, scipy_n, cor in zip(axes, nomes, scipy_nomes, cores):
        freqs, amp = calcular_espectro(sinal, fs, nome, scipy_n)
        ax.plot(freqs, amp + offset, color=cor, lw=1.2)
        _aplicar_escala(ax, escala, amp)

        for k in range(1, 6):
            fx = k * freq_rot
            if fx <= fmax_plot:
                ax.axvline(fx, color='white', ls=':', lw=1, alpha=0.5)

        ax.set_xlim([0, fmax_plot])
        ax.set_title(f'Janela: {nome}', color=cor, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frequência (Hz)', fontsize=11)
    fig.suptitle(f'Efeito da Janela no Espectro — {canal}', fontsize=14,
                 fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, 'comparacao_janelas')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — ANÁLISE DE ORDEM (ORDER TRACKING)
# ═════════════════════════════════════════════════════════════════════════════

def analise_ordem(sinais_acel: dict, fase_ref: np.ndarray, indices_pulsos: np.ndarray,
                   rpm: float, fs: float):
    """
    Módulo educacional de análise de ordem.
    Reamostrar sinais no domínio angular para remover variação de RPM.
    """
    box("MÓDULO DIDÁTICO 3 — ANÁLISE DE ORDEM (ORDER TRACKING)")

    print("""
  O Order Tracking REAMOSTRA o sinal no domínio ANGULAR (0°–360°)
  em vez do domínio temporal.

  Vantagens:
    • Elimina o efeito de variação de rotação (run-up/coast-down)
    • O eixo X vira "Ordem" (múltiplos da rotação), não frequência
    • Picos sempre aparecem em posições EXATAS (1X, 2X, ...)

  Uso didático:
    • Comparar revoluções sobrepostas → verificar repetibilidade
    • Identificar fenômenos síncronos com a rotação
""")

    canal = 'CH2'
    ch_idx = escolha_menu(
        ['CH2 — Radial Vertical', 'CH3 — Radial Horizontal', 'CH4 — Axial'],
        "Canal para Order Tracking"
    )
    canais = ['CH2', 'CH3', 'CH4']
    canal = canais[ch_idx]
    sinal = sinais_acel.get(canal, list(sinais_acel.values())[0])
    cor = [COR_CH2, COR_CH3, COR_CH4][ch_idx]

    # Identificar limites de cada revolução
    indices_rev = [0]
    for i in range(1, len(fase_ref)):
        if fase_ref[i] < fase_ref[i - 1]:
            indices_rev.append(i)
    indices_rev.append(len(fase_ref))

    n_revs = len(indices_rev) - 1
    print(f"\n  Revoluções detectadas: {n_revs}")

    # Descartar primeiras e últimas 2 (transitório)
    rev_validas = list(range(2, n_revs - 2)) if n_revs > 5 else list(range(n_revs))
    print(f"  Revoluções usadas (descartando primeiras/últimas 2): {len(rev_validas)}")

    angulos_grid = np.linspace(0, 360, 360, endpoint=False)
    revs_interpoladas = []

    for i in rev_validas:
        i0 = indices_rev[i]
        i1 = indices_rev[i + 1]
        fase_r = fase_ref[i0:i1]
        sinal_r = sinal[i0:i1]
        if len(fase_r) > 10:
            interp = np.interp(angulos_grid, fase_r, sinal_r)
            revs_interpoladas.append(interp)

    if not revs_interpoladas:
        print("  [ERRO] Nenhuma revolução interpolada.")
        return

    mat = np.array(revs_interpoladas)
    media = np.mean(mat, axis=0)
    desvio_pad = np.std(mat, axis=0)

    # ── Escolha do modo do espectro de ordem ──────────────────────────────
    print("""
  MODO DO ESPECTRO DE ORDEM:
  ─────────────────────────────────────────────────────────────────
  [1] Espectro CONTÍNUO — mostra todos os componentes espectrais,
      inclusive não-síncronos (rolamentos, BPF, etc.)
  [2] Harmônicas DISCRETAS — barras somente nos múltiplos inteiros
      da rotação (1X, 2X, ...). Ideal para desbalanceamento e
      desalinhamento. Elimina visualmente o ruído de fundo.
  ─────────────────────────────────────────────────────────────────""")
    modo_espectro = '1'
    while modo_espectro not in ('1', '2'):
        modo_espectro = input("  Escolha [1/2] (padrão=1): ").strip() or '1'

    n_ordens_max = entrada_int("  Número máximo de ordens a exibir", padrao=20,
                                minval=5, maxval=100)

    # ── Calcular espectro de ordem ─────────────────────────────────────────
    N_ord   = len(media)
    win_ord = signal.get_window('hann', N_ord)
    yf_ord  = rfft(media * win_ord)
    ordens  = rfftfreq(N_ord, d=1.0 / N_ord)   # eixo em "ordens"
    amp_ord = (2.0 / N_ord) * np.abs(yf_ord) * (N_ord / np.sum(win_ord))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='#1a1a2e')

    # Revoluções sobrepostas
    ax = axes[0]
    n_plot = min(10, len(revs_interpoladas))
    cores_rev = plt.cm.cool(np.linspace(0, 1, n_plot))
    for k in range(n_plot):
        ax.plot(angulos_grid, revs_interpoladas[k], color=cores_rev[k],
                lw=0.8, alpha=0.6, label=f'Rev {rev_validas[k]+1}' if k < 5 else '')
    ax.axvline(0, color='red', ls='--', lw=2, label='0° (pulso taco)')
    for ang in [90, 180, 270]:
        ax.axvline(ang, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_title(f'{n_plot} Revoluções Sobrepostas — {canal}', color='white', fontsize=11)
    ax.set_xlabel('Ângulo (°)'); ax.set_ylabel(get_ylabel_tempo())
    ax.set_xlim([0, 360]); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Média ± desvio
    ax = axes[1]
    ax.plot(angulos_grid, media, color=cor, lw=2.5, label='Média')
    ax.fill_between(angulos_grid, media - desvio_pad, media + desvio_pad,
                    alpha=0.3, color=cor, label='±1σ')
    ax.axvline(0, color='red', ls='--', lw=2)
    for ang in [90, 180, 270]:
        ax.axvline(ang, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_title(f'Média de {len(rev_validas)} Rev — {canal}', color='white', fontsize=11)
    ax.set_xlabel('Ângulo (°)'); ax.set_ylabel(get_ylabel_tempo())
    ax.set_xlim([0, 360]); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Espectro de ordem — modo escolhido
    ax = axes[2]

    if modo_espectro == '1':
        # ── ESPECTRO CONTÍNUO ──────────────────────────────────────────
        mask_ord = ordens <= n_ordens_max
        ax.plot(ordens[mask_ord], amp_ord[mask_ord], color=cor, lw=1.5)
        # Marcar harmônicas inteiras
        for k_h in range(1, min(n_ordens_max + 1, 21)):
            idx_h = np.argmin(np.abs(ordens - k_h))
            ax.axvline(k_h, color='white', ls=':', lw=0.8, alpha=0.4)
            ax.text(k_h, amp_ord[mask_ord].max() * 0.92,
                    f'{k_h}X', ha='center', fontsize=7, color='white', alpha=0.7)
        ax.set_title('Espectro de Ordem — CONTÍNUO', color='white', fontsize=11)
        ax.set_xlabel('Ordem (1 = 1×RPM)'); ax.set_ylabel(get_ylabel_tempo())
        ax.set_xlim([0, n_ordens_max])
        # Ticks nos inteiros
        ax.set_xticks(range(0, n_ordens_max + 1, max(1, n_ordens_max // 10)))
        ax.grid(True, alpha=0.3)
    else:
        # ── HARMÔNICAS DISCRETAS ────────────────────────────────────────
        ords_int  = np.arange(1, n_ordens_max + 1)
        amps_int  = []
        for k_h in ords_int:
            idx_h = np.argmin(np.abs(ordens - k_h))
            # Tomar o máximo na vizinhança ±0,5 ordens
            vizinhos = np.abs(ordens - k_h) < 0.5
            amps_int.append(np.max(amp_ord[vizinhos]) if np.any(vizinhos) else 0.0)
        amps_int = np.array(amps_int)

        bars = ax.bar(ords_int, amps_int, color=cor, alpha=0.85,
                      edgecolor='black', linewidth=1.5, width=0.6)
        # Anotar amplitudes
        for b, a_v in zip(bars, amps_int):
            if a_v > amps_int.max() * 0.05:
                ax.text(b.get_x() + b.get_width()/2, a_v,
                        f'{a_v:.4f}', ha='center', va='bottom',
                        fontsize=7, color='white', fontweight='bold')
        ax.set_title('Espectro de Ordem — HARMÔNICAS DISCRETAS', color='white', fontsize=11)
        ax.set_xlabel('Ordem (1X, 2X, ...)'); ax.set_ylabel(get_ylabel_tempo())
        ax.set_xlim([0.3, n_ordens_max + 0.7])
        ax.set_xticks(ords_int)
        ax.set_xticklabels([f'{k}X' for k in ords_int], fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    modo_label = 'Contínuo' if modo_espectro == '1' else 'Harmônicas Discretas'
    fig.suptitle(f'ORDER TRACKING — RPM: {rpm:.1f} — {canal} — Modo: {modo_label}',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, f'order_tracking_{canal}')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — ÓRBITA E ANÁLISE TEMPORAL MULTI-CANAL
# ═════════════════════════════════════════════════════════════════════════════

def analise_orbita(sinais_acel: dict, fase_ref: np.ndarray, rpm: float, fs: float):
    """
    Plot de órbita CH2×CH3 (horizontal × vertical) e análise temporal.
    """
    box("MÓDULO DIDÁTICO 4 — ÓRBITA E ANÁLISE TEMPORAL MULTI-CANAL")

    print("""
  A ÓRBITA representa a trajetória do eixo no plano XY:
    → Eixo X: acelerômetro horizontal (CH2)
    → Eixo Y: acelerômetro vertical   (CH3)

  Formatos típicos de órbita:
    ○ Círculo/elipse regular   → desbalanceamento
    ∞ Formato "banana"         → desalinhamento angular
    ✦ Formas caóticas          → folga mecânica
    ◯ Loop interno             → contato rotor-estator
""")

    ch2 = sinais_acel.get('CH2')
    ch3 = sinais_acel.get('CH3')

    if ch2 is None or ch3 is None:
        print("  [AVISO] CH2 ou CH3 não disponível. Pulando órbita.")
        return

    # Filtrar passa-banda em torno de 1X–3X para a órbita
    freq_rot = rpm / 60.0
    lowcut  = max(0.4 * freq_rot, 0.1)
    highcut = min(5 * freq_rot, fs / 2 - 1)
    sos = signal.butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
    ch2_f = signal.sosfilt(sos, ch2)
    ch3_f = signal.sosfilt(sos, ch3)

    t = np.arange(len(ch2)) / fs
    t_max_plot = min(3.0, t[-1])
    mask = t <= t_max_plot

    fig = plt.figure(figsize=(20, 12), facecolor='#1a1a2e')
    gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Temporal CH2
    ax1 = fig.add_subplot(gs_fig[0, 0])
    ax1.plot(t[mask], ch2[mask], color=COR_CH2, lw=0.8, alpha=0.7, label='Bruto')
    ax1.plot(t[mask], ch2_f[mask], color='white', lw=1.2, label='Filtrado')
    ax1.set_title('CH2 — Radial Vertical', color='white'); ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel(get_ylabel_tempo()); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Temporal CH3
    ax2 = fig.add_subplot(gs_fig[0, 1])
    ax2.plot(t[mask], ch3[mask], color=COR_CH3, lw=0.8, alpha=0.7, label='Bruto')
    ax2.plot(t[mask], ch3_f[mask], color='white', lw=1.2, label='Filtrado')
    ax2.set_title('CH3 — Radial Horizontal', color='white'); ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel(get_ylabel_tempo()); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Temporal CH4 (axial)
    ax3 = fig.add_subplot(gs_fig[0, 2])
    ch4 = sinais_acel.get('CH4')
    if ch4 is not None:
        ch4_f = signal.sosfilt(sos, ch4)
        ax3.plot(t[mask], ch4[mask], color=COR_CH4, lw=0.8, alpha=0.7, label='Bruto')
        ax3.plot(t[mask], ch4_f[mask], color='white', lw=1.2, label='Filtrado')
        ax3.set_title('CH4 — Axial', color='white')
        ax3.set_xlabel('Tempo (s)'); ax3.set_ylabel(get_ylabel_tempo())
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Órbita filtrada
    ax4 = fig.add_subplot(gs_fig[1, 0])
    sc = ax4.scatter(ch2_f[mask], ch3_f[mask],
                     c=t[mask], cmap='plasma', s=1, alpha=0.7)
    plt.colorbar(sc, ax=ax4, label='Tempo (s)')
    ax4.axhline(0, color='gray', lw=0.5); ax4.axvline(0, color='gray', lw=0.5)
    ax4.set_aspect('equal', 'box')
    ax4.set_title('Órbita CH2×CH3 (RadVert × RadHoriz) — Filtrado', color='white', fontsize=11)
    ax4.set_xlabel('Radial Vert. (CH2)'); ax4.set_ylabel('Radial Horiz. (CH3)')
    ax4.grid(True, alpha=0.3)

    # Órbita de 1 revolução representativa
    ax5 = fig.add_subplot(gs_fig[1, 1])
    # Encontrar início de revolução no meio do sinal
    mid = len(fase_ref) // 2
    i_start = mid
    for i in range(mid, len(fase_ref) - 1):
        if fase_ref[i] > 350 and fase_ref[i + 1] < 10:
            i_start = i + 1
            break
    # Encontrar fim da próxima revolução
    i_end = i_start + 1
    for i in range(i_start + 1, min(i_start + int(2 * fs / (rpm / 60)), len(fase_ref))):
        if fase_ref[i] > 350:
            i_end = i
            break

    ax5.plot(ch2_f[i_start:i_end], ch3_f[i_start:i_end],
             color=COR_DEST, lw=2.5)
    ax5.plot(ch2_f[i_start], ch3_f[i_start], 'go', ms=12, label='0° (início)')
    ax5.axhline(0, color='gray', lw=0.5); ax5.axvline(0, color='gray', lw=0.5)
    ax5.set_aspect('equal', 'box')
    ax5.set_title('Órbita de 1 Revolução (central)', color='white', fontsize=11)
    ax5.set_xlabel('Radial Vert. (CH2)'); ax5.set_ylabel('Radial Horiz. (CH3)')
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    # Estatísticas de vibração
    ax6 = fig.add_subplot(gs_fig[1, 2])
    ax6.axis('off')
    canais_stat = {'CH2': ch2, 'CH3': ch3}
    if ch4 is not None:
        canais_stat['CH4'] = ch4

    linhas = ["ESTATÍSTICAS DE VIBRAÇÃO\n",
              f"{'Canal':<6} {'RMS':>10} {'Pico':>10} {'Crista':>8} {'Kurt':>8}"]
    linhas.append("─" * 46)
    for ch, v in canais_stat.items():
        rms_v = np.sqrt(np.mean(v**2))
        pico  = np.max(np.abs(v))
        crest = pico / (rms_v + 1e-12)
        kurt  = float(kurtosis(v))
        linhas.append(f"{ch:<6} {rms_v:>10.5f} {pico:>10.5f} {crest:>8.2f} {kurt:>8.2f}")

    linhas.append("\nINDICADORES:")
    linhas.append("  Fator de Crista > 3 → impactos / folga")
    linhas.append("  Curtose > 3          → impactos / falha incipiente")

    txt = "\n".join(linhas)
    ax6.text(0.02, 0.95, txt, transform=ax6.transAxes, fontsize=9,
             va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.95,
                       edgecolor='#aaaaaa', lw=1.5))

    fig.suptitle(f'ANÁLISE TEMPORAL E ÓRBITA — RPM: {rpm:.1f}',
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, 'orbita_temporal')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — CALCULADORA DE FREQUÊNCIAS DE FALHA
# ═════════════════════════════════════════════════════════════════════════════

def calcular_frequencias_falha(rpm: float) -> dict:
    """
    Menu interativo para calcular e explicar frequências características
    de diferentes tipos de falha em máquinas rotativas.
    """
    box("MÓDULO DIDÁTICO 5 — FREQUÊNCIAS CARACTERÍSTICAS DE FALHAS")

    freq_rot = rpm / 60.0

    print(f"""
  FREQUÊNCIA DE ROTAÇÃO:
    {rpm:.2f} RPM  →  {freq_rot:.4f} Hz  →  1X = {freq_rot:.4f} Hz
""")

    frequencias = {'rpm': rpm, 'freq_rot': freq_rot, 'falhas': {}}

    while True:
        # Mostrar componentes já calculados no cabeçalho do menu
        calculados = list(frequencias['falhas'].keys())
        if calculados:
            print(f"\n  Componentes calculados: {', '.join(calculados)}")

        idx_tipo = escolha_menu([
            'Desbalanceamento',
            'Desalinhamento',
            'Rolamento (calcular BPFI, BPFO, BSF, FTF)',
            'Ventilador (Blade Pass Frequency)',
            'Engrenagem (Gear Mesh Frequency)',
            'Polia / Correia',
            'Folga Mecânica',
            'Mostrar TODAS as frequências calculadas',
            'Remover um componente calculado',
            'Voltar ao menu principal',
        ], "Selecione o componente")

        if idx_tipo == 0:
            _explicar_desbalanceamento(freq_rot, frequencias)
        elif idx_tipo == 1:
            _explicar_desalinhamento(freq_rot, frequencias)
        elif idx_tipo == 2:
            _calcular_rolamento(freq_rot, frequencias)
        elif idx_tipo == 3:
            _calcular_ventilador(freq_rot, frequencias)
        elif idx_tipo == 4:
            _calcular_engrenagem(freq_rot, frequencias)
        elif idx_tipo == 5:
            _calcular_polia(freq_rot, frequencias)
        elif idx_tipo == 6:
            _explicar_folga(freq_rot, frequencias)
        elif idx_tipo == 7:
            _mostrar_todas_frequencias(frequencias)
        elif idx_tipo == 8:
            _remover_componente(frequencias)
        elif idx_tipo == 9:
            break

    return frequencias


def _explicar_desbalanceamento(freq_rot, frequencias):
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  DESBALANCEAMENTO (Unbalance)                            ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Causa: massa desigualmente distribuída no rotor         ║
  ║                                                          ║
  ║  Assinatura espectral:                                   ║
  ║    • Pico dominante em 1X (sempre)                       ║
  ║    • Direção RADIAL (horizontal e vertical)              ║
  ║    • Órbita em forma de elipse/círculo regular           ║
  ║    • Amplitude proporcional ao quadrado da rotação       ║
  ║                                                          ║
  ║  Norma ISO 21940 define graus de balanceamento (G0.4     ║
  ║  até G4000)                                              ║
  ╚══════════════════════════════════════════════════════════╝
""")
    print(f"  Frequência de desbalanceamento: {freq_rot:.4f} Hz  (= 1X)")
    print(f"  2X pode aparecer se há deflexão ou excentricidade combinada.")
    frequencias['falhas']['desbalanceamento'] = {
        '1X': freq_rot,
        '2X': 2 * freq_rot,
    }


def _explicar_desalinhamento(freq_rot, frequencias):
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  DESALINHAMENTO (Misalignment)                           ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Causa: eixos não colineares no acoplamento              ║
  ║                                                          ║
  ║  Tipos:                                                  ║
  ║    → Angular: eixos com ângulo entre si                  ║
  ║    → Paralelo: eixos paralelos mas deslocados            ║
  ║    → Combinado: ambos simultaneamente                    ║
  ║                                                          ║
  ║  Assinatura espectral:                                   ║
  ║    • 1X presente (às vezes dominante)                    ║
  ║    • 2X alto é CARACTERÍSTICO                            ║
  ║    • 3X significativo em desalinhamento severo           ║
  ║    • Forte componente AXIAL (1X axial >> radial)         ║
  ║    • Órbita em formato "8" ou banana                     ║
  ╚══════════════════════════════════════════════════════════╝
""")
    print(f"  1X: {freq_rot:.4f} Hz")
    print(f"  2X: {2*freq_rot:.4f} Hz  ← INDICADOR PRIMÁRIO")
    print(f"  3X: {3*freq_rot:.4f} Hz")
    frequencias['falhas']['desalinhamento'] = {
        '1X': freq_rot, '2X': 2 * freq_rot, '3X': 3 * freq_rot
    }


def _calcular_rolamento(freq_rot, frequencias):
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  ROLAMENTO — FREQUÊNCIAS DE DEFEITO                      ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Frequências de falha:                                   ║
  ║    BPFO → Outer Race (pista externa) — mais comum!       ║
  ║    BPFI → Inner Race (pista interna)                     ║
  ║    BSF  → Ball/Roller Spin (giro do elemento)            ║
  ║    FTF  → Fundamental Train Freq. (gaiola / cage)        ║
  ╚══════════════════════════════════════════════════════════╝
""")

    modo = escolha_menu(
        ['Calcular a partir dos parâmetros GEOMÉTRICOS do rolamento '
         '(n, Dp, Db, α)',
         'Inserir diretamente as FREQUÊNCIAS de defeito '
         '(do catálogo ou medição anterior)'],
        "Como deseja informar o rolamento?"
    )

    if modo == 0:
        # ── Via parâmetros geométricos ───────────────────────────────────
        print("""
  PARÂMETROS GEOMÉTRICOS:
    n  = número de elementos rolantes (esferas ou rolos)
    Dp = diâmetro da circunferência primitiva (pitch diameter) em mm
    Db = diâmetro do elemento rolante (ball/roller diameter) em mm
    α  = ângulo de contato em graus (0° esferas radiais, ~15° angulares)

  Dica: esses dados constam no catálogo do fabricante (SKF, FAG, NSK…)
  ou podem ser medidos diretamente no rolamento desmontado.
""")
        n_elem    = entrada_int("  Número de elementos (n)", minval=3, maxval=60)
        Dp        = entrada_float("  Diâmetro primitivo Dp (mm)", minval=1, maxval=1000)
        Db        = entrada_float("  Diâmetro do elemento Db (mm)", minval=0.5, maxval=200)
        alpha_deg = entrada_float("  Ângulo de contato α (°)", padrao=0, minval=0, maxval=40)
        alpha_rad = np.deg2rad(alpha_deg)

        ratio = Db / Dp * np.cos(alpha_rad)

        BPFO = (n_elem / 2) * freq_rot * (1 - ratio)
        BPFI = (n_elem / 2) * freq_rot * (1 + ratio)
        BSF  = (Dp / (2 * Db)) * freq_rot * (1 - ratio**2)
        FTF  = (freq_rot / 2) * (1 - ratio)

        print(f"""
  FÓRMULAS APLICADAS:
    BPFO = (n/2) × fr × (1 - Db/Dp × cos α) = {BPFO:.4f} Hz
    BPFI = (n/2) × fr × (1 + Db/Dp × cos α) = {BPFI:.4f} Hz
    BSF  = (Dp/2Db) × fr × [1-(Db/Dp×cosα)²] = {BSF:.4f} Hz
    FTF  = (fr/2)  × (1 - Db/Dp × cos α)    = {FTF:.4f} Hz
""")
        dados_extra = {'n': n_elem, 'Dp': Dp, 'Db': Db, 'alpha': alpha_deg,
                       'modo': 'geometrico'}

    else:
        # ── Via frequências diretas do catálogo ──────────────────────────
        print("""
  FREQUÊNCIAS DO CATÁLOGO:
  Os fabricantes fornecem os fatores de frequência adimensionais
  (ex: BPFO_fator = 3.572), e a frequência real é:

      f_defeito = fator × (RPM / 60)

  Você pode inserir o fator adimensional OU a frequência em Hz:
""")
        modo_cat = escolha_menu(
            ['Inserir FATORES adimensionais (do catálogo)',
             'Inserir FREQUÊNCIAS em Hz diretamente'],
            "Formato de entrada"
        )

        if modo_cat == 0:
            print("  (Exemplos de fatores típicos: BPFO≈3.6, BPFI≈5.4, BSF≈2.3, FTF≈0.4)")
            fator_BPFO = entrada_float("  Fator BPFO", minval=0.1, maxval=50)
            fator_BPFI = entrada_float("  Fator BPFI", minval=0.1, maxval=50)
            fator_BSF  = entrada_float("  Fator BSF ", minval=0.1, maxval=30)
            fator_FTF  = entrada_float("  Fator FTF ", minval=0.1, maxval=5)
            BPFO = fator_BPFO * freq_rot
            BPFI = fator_BPFI * freq_rot
            BSF  = fator_BSF  * freq_rot
            FTF  = fator_FTF  * freq_rot
            print(f"\n  Frequências calculadas (fator × {freq_rot:.4f} Hz):")
            dados_extra = {'fator_BPFO': fator_BPFO, 'fator_BPFI': fator_BPFI,
                           'fator_BSF': fator_BSF, 'fator_FTF': fator_FTF,
                           'modo': 'catalogo_fatores'}
        else:
            print("  (Informe as frequências de defeito já em Hz)")
            BPFO = entrada_float("  BPFO (Hz)", minval=0.1)
            BPFI = entrada_float("  BPFI (Hz)", minval=0.1)
            BSF  = entrada_float("  BSF  (Hz)", minval=0.1)
            FTF  = entrada_float("  FTF  (Hz)", minval=0.1)
            dados_extra = {'modo': 'frequencias_diretas'}

    # ── Exibir resultado final (comum às duas vias) ──────────────────────
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  FREQUÊNCIAS DE FALHA DO ROLAMENTO                      │
  ├────────────┬───────────────┬───────────────────────────┤
  │  Tipo      │  Frequência   │  Significado               │
  ├────────────┼───────────────┼───────────────────────────┤
  │  BPFO      │  {BPFO:>8.3f} Hz │  Defeito pista EXTERNA    │
  │  BPFI      │  {BPFI:>8.3f} Hz │  Defeito pista INTERNA    │
  │  BSF       │  {BSF:>8.3f} Hz │  Defeito no ELEMENTO      │
  │  FTF       │  {FTF:>8.3f} Hz │  Defeito na GAIOLA        │
  └────────────┴───────────────┴───────────────────────────┘

  Bandas laterais (pista interna - tipicamente mais complexa):
    BPFI - 1X = {BPFI - freq_rot:.4f} Hz
    BPFI + 1X = {BPFI + freq_rot:.4f} Hz

  ATENÇÃO: Rolamentos defeituosos geram também:
    • Ruído de banda larga + excitação de frequências naturais
    • Indicadores estatísticos elevados: Kurtose > 3, Crista > 3
""")

    label_rol = f'rolamento_{len([k for k in frequencias["falhas"] if "rolamento" in k])+1}'
    dados_rol = {'BPFO': BPFO, 'BPFI': BPFI, 'BSF': BSF, 'FTF': FTF,
                 'BPFI-1X': BPFI - freq_rot, 'BPFI+1X': BPFI + freq_rot}
    dados_rol.update(dados_extra)
    frequencias['falhas'][label_rol] = dados_rol


def _calcular_ventilador(freq_rot, frequencias):
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  VENTILADOR — BLADE PASS FREQUENCY (BPF)                 ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Causa de vibração: interação entre pás e obstáculos     ║
  ║  (difusor, voluta, suportes)                             ║
  ║                                                          ║
  ║  Assinatura:                                             ║
  ║    • Pico em BPF = N_pás × fr                           ║
  ║    • Harmônicas: 2×BPF, 3×BPF                           ║
  ║    • 1X presente (desbalanceamento típico)               ║
  ║    • Modulação: BPF ± 1X (bandas laterais)              ║
  ╚══════════════════════════════════════════════════════════╝
""")
    n_pas = entrada_int("  Número de pás do ventilador", minval=2, maxval=100)
    BPF = n_pas * freq_rot
    print(f"\n  BPF = {n_pas} × {freq_rot:.4f} = {BPF:.4f} Hz")
    print(f"  2×BPF = {2*BPF:.4f} Hz")
    print(f"  3×BPF = {3*BPF:.4f} Hz")
    print(f"  Bandas laterais: {BPF - freq_rot:.4f} Hz e {BPF + freq_rot:.4f} Hz")

    frequencias['falhas']['ventilador'] = {
        'BPF': BPF, '2BPF': 2 * BPF, '3BPF': 3 * BPF,
        'BPF-1X': BPF - freq_rot, 'BPF+1X': BPF + freq_rot, 'n_pas': n_pas
    }


def _calcular_engrenagem(freq_rot, frequencias):
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  ENGRENAGEM — GEAR MESH FREQUENCY (GMF)                  ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Assinatura:                                             ║
  ║    • GMF = N_dentes × fr (pinhão ou coroa?)              ║
  ║    • Harmônicas: 2×GMF, 3×GMF                           ║
  ║    • Bandas laterais: GMF ± fr_pinhão, GMF ± fr_coroa   ║
  ║    • Ghost frequencies: freqs não-inteiras de engr.     ║
  ╚══════════════════════════════════════════════════════════╝
""")
    print("  Engrenagem do EIXO MONITORADO (onde o taco está acoplado):")
    n_dentes = entrada_int("  Número de dentes", minval=2, maxval=500)
    GMF = n_dentes * freq_rot
    print(f"\n  GMF = {n_dentes} × {freq_rot:.4f} = {GMF:.4f} Hz")
    print(f"  2×GMF = {2*GMF:.4f} Hz")
    print(f"  3×GMF = {3*GMF:.4f} Hz")

    tem_par = input("  Deseja calcular para o eixo PAR também? [s/N]: ").strip().lower() == 's'
    dados_gmf = {'GMF': GMF, '2GMF': 2 * GMF, '3GMF': 3 * GMF, 'n_dentes': n_dentes}

    if tem_par:
        rpm_par = entrada_float("  RPM do eixo par", minval=1)
        n_dentes_par = entrada_int("  Dentes do eixo par", minval=2, maxval=500)
        freq_par = rpm_par / 60.0
        GMF_par = n_dentes_par * freq_par
        print(f"\n  GMF eixo par = {GMF_par:.4f} Hz")
        print(f"  Bandas laterais: GMF ± fr_pinhão: {GMF:.3f}±{freq_rot:.3f}")
        dados_gmf['GMF_par'] = GMF_par
        dados_gmf['freq_par'] = freq_par

    frequencias['falhas']['engrenagem'] = dados_gmf


def _calcular_polia(freq_rot, frequencias):
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  POLIA / CORREIA (Belt/Pulley)                           ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Frequências importantes:                                ║
  ║    fr_corr = velocidade da correia / comprimento         ║
  ║    fr_pola2 = fr_polia1 × D1 / D2                       ║
  ║                                                          ║
  ║  Assinatura de correia defeituosa:                       ║
  ║    • 2× fr_correia (correia bate 2× por volta)          ║
  ║    • Sub-harmônicas (0,5X)                               ║
  ║    • Modulação em amplitude a fr_correia                 ║
  ╚══════════════════════════════════════════════════════════╝
""")
    D1 = entrada_float("  Diâmetro da polia motriz D1 (mm)", minval=10)
    D2 = entrada_float("  Diâmetro da polia movida D2 (mm)", minval=10)
    L  = entrada_float("  Comprimento da correia (mm)", minval=100)

    freq_polia2 = freq_rot * D1 / D2
    rpm_polia2  = freq_polia2 * 60

    # Velocidade linear da correia: v = π × D1 × fr (m/s, com D em metros)
    v_correia = np.pi * (D1 / 1000) * freq_rot    # m/s
    freq_correia = v_correia / (L / 1000)          # Hz

    relacao = D1 / D2

    print(f"""
  Relação de transmissão     : {relacao:.3f}
  Rotação polia movida       : {rpm_polia2:.2f} RPM  ({freq_polia2:.4f} Hz)
  Velocidade da correia      : {v_correia:.3f} m/s
  Frequência da correia (fr_b): {freq_correia:.4f} Hz
  2 × fr_correia             : {2*freq_correia:.4f} Hz
""")

    frequencias['falhas']['polia_correia'] = {
        'freq_polia1': freq_rot, 'freq_polia2': freq_polia2,
        'freq_correia': freq_correia, '2fr_correia': 2 * freq_correia,
        'D1': D1, 'D2': D2, 'L': L
    }


def _explicar_folga(freq_rot, frequencias):
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║  FOLGA MECÂNICA (Mechanical Looseness)                   ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Tipos:                                                  ║
  ║    Tipo A: parafusos soltos na base/estrutura            ║
  ║    Tipo B: folga no mancal / bucha                       ║
  ║    Tipo C: folga no rotor (anel interno solto no eixo)   ║
  ║                                                          ║
  ║  Assinatura espectral:                                   ║
  ║    • Múltiplas harmônicas 1X, 2X, 3X, ..., nX           ║
  ║    • Sub-harmônicas: 0,5X (folga tipo B/C)               ║
  ║    • Truncamento de forma de onda (clipping)             ║
  ║    • Alto fator de crista (Crest Factor)                 ║
  ║    • Kurtose elevada                                     ║
  ╚══════════════════════════════════════════════════════════╝
""")
    print("  Harmônicas a verificar:")
    for k in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]:
        print(f"    {k}X = {k * freq_rot:.4f} Hz")

    frequencias['falhas']['folga'] = {
        f'{k}X': k * freq_rot for k in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
    }


def _mostrar_todas_frequencias(frequencias):
    """Lista todas as frequências calculadas até o momento."""
    print("\n" + "─" * 65)
    print(f"  {'FREQ. DE FALHA CALCULADAS':^63}")
    print("─" * 65)
    print(f"  RPM: {frequencias['rpm']:.2f}  |  1X = {frequencias['freq_rot']:.4f} Hz")
    print("─" * 65)

    for componente, dados in frequencias['falhas'].items():
        print(f"\n  [{componente.upper()}]")
        for chave, valor in dados.items():
            if isinstance(valor, float):
                print(f"    {chave:<12}: {valor:>10.4f} Hz")
            else:
                print(f"    {chave:<12}: {valor}")

    print("─" * 65)


def _remover_componente(frequencias):
    """Remove um componente do dicionário de frequências calculadas."""
    falhas = frequencias.get('falhas', {})

    if not falhas:
        print("  Nenhum componente calculado ainda.")
        return

    print("\n  COMPONENTES DISPONÍVEIS PARA REMOVER:")
    print("  " + "─" * 50)
    chaves = list(falhas.keys())
    for i, ch in enumerate(chaves, 1):
        dados = falhas[ch]
        freqs_str = ', '.join(
            f"{k}={v:.2f}Hz" for k, v in dados.items()
            if isinstance(v, float)
        )
        print(f"  [{i}] {ch.upper():<20}  {freqs_str[:55]}")
    print(f"  [{len(chaves)+1}] Cancelar (não remover nada)")
    print("  " + "─" * 50)

    idx = entrada_int(f"  Escolha [1-{len(chaves)+1}]",
                      minval=1, maxval=len(chaves) + 1)

    if idx == len(chaves) + 1:
        print("  Operação cancelada.")
        return

    removido = chaves[idx - 1]
    del frequencias['falhas'][removido]
    print(f"  ✓ Componente '{removido}' removido.")
    restantes = list(frequencias['falhas'].keys())
    print(f"  Restantes: {restantes if restantes else 'nenhum'}")


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 10 — SOBREPOSIÇÃO DE FREQUÊNCIAS DE FALHA NO ESPECTRO
# ═════════════════════════════════════════════════════════════════════════════

def plot_espectro_com_falhas(sinais_acel: dict, fs: float, rpm: float,
                              frequencias: dict):
    """
    Plota o espectro de TODOS os canais com as frequências de falha calculadas
    marcadas no gráfico — ferramenta diagnóstica principal.
    """
    box("DIAGNÓSTICO — ESPECTRO COM FREQUÊNCIAS DE FALHA MARCADAS")

    nome_jan, scipy_jan = selecionar_janela()
    escala = selecionar_escala()
    offset = 1e-12 if escala == 'log' else 0.0
    fmax_plot = entrada_float("  Frequência máxima (Hz)", padrao=min(fs / 2, 500),
                              minval=10, maxval=fs / 2)

    canais_cores = [('CH2', COR_CH2, 'Radial Vert.'),
                    ('CH3', COR_CH3, 'Radial Horiz.'),
                    ('CH4', COR_CH4, 'Axial')]

    n_ax = sum(1 for ch, _, _ in canais_cores if ch in sinais_acel)
    if n_ax == 0:
        print("  Nenhum canal de acelerômetro disponível.")
        return

    fig, axes = plt.subplots(n_ax, 1, figsize=(20, 5 * n_ax),
                             facecolor='#1a1a2e', sharex=True)
    if n_ax == 1:
        axes = [axes]

    freq_rot = rpm / 60.0

    # Paleta de cores para os componentes de falha
    cores_componentes = {
        'desbalanceamento': '#ff4444',
        'desalinhamento':   '#ff9900',
        'rolamento':        '#aa44ff',
        'ventilador':       '#00ccff',
        'engrenagem':       '#44ff88',
        'polia_correia':    '#ff66cc',
        'folga':            '#ffdd00',
    }

    ax_idx = 0
    for ch, cor, descricao in canais_cores:
        if ch not in sinais_acel:
            continue
        ax = axes[ax_idx]
        ax_idx += 1

        freqs, amp = calcular_espectro(sinais_acel[ch], fs, nome_jan, scipy_jan)
        ax.plot(freqs, amp + offset, color=cor, lw=0.9, alpha=0.85,
                    label=f'{ch} — {descricao}', zorder=2)
        _aplicar_escala(ax, escala, amp)

        # Harmônicas de rotação
        for k in range(1, 11):
            fx = k * freq_rot
            if fx <= fmax_plot:
                ax.axvline(fx, color='white', ls=':', lw=0.8, alpha=0.4, zorder=1)
                if k <= 5:
                    ax.text(fx, amp.max() * 1.5, f'{k}X', color='white',
                            fontsize=7, ha='center', alpha=0.7)

        # Frequências de falha calculadas
        for comp_nome, comp_dados in frequencias.get('falhas', {}).items():
            # Encontrar cor da componente
            cor_comp = '#ffffff'
            for kc, vc in cores_componentes.items():
                if kc in comp_nome.lower():
                    cor_comp = vc
                    break

            for freq_nome, freq_val in comp_dados.items():
                if isinstance(freq_val, float) and 0 < freq_val <= fmax_plot:
                    ax.axvline(freq_val, color=cor_comp, ls='--', lw=1.8,
                               alpha=0.8, zorder=3)
                    ylims = ax.get_ylim()
                    ytxt = (ylims[0] + ylims[1]) / 2 if escala == 'linear' \
                           else (ylims[0] * ylims[1]) ** 0.5
                    ax.text(freq_val, ytxt,
                            f'{comp_nome[:4]}\n{freq_nome}',
                            color=cor_comp, fontsize=6, ha='center',
                            bbox=dict(facecolor='black', alpha=0.5, lw=0))

        ax.set_xlim([0, fmax_plot])
        ax.set_ylabel(get_ylabel_amp(), fontsize=10)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frequência (Hz)', fontsize=11)

    # Legenda de componentes de falha
    legend_elements = []
    for comp_nome, cor_comp in cores_componentes.items():
        if any(comp_nome in k for k in frequencias.get('falhas', {}).keys()):
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], color=cor_comp, ls='--', lw=2, label=comp_nome.capitalize())
            )
    if legend_elements:
        axes[0].legend(handles=legend_elements, loc='upper left', fontsize=8)

    escala_label = 'Log' if escala == 'log' else 'Linear'
    fig.suptitle(
        f'DIAGNÓSTICO DE FALHAS — RPM: {rpm:.1f} | Janela: {nome_jan} | '
        f'Escala: {escala_label} | '
        f'Δf = {fs / len(sinais_acel[list(sinais_acel.keys())[0]]):.4f} Hz',
        fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, 'diagnostico_falhas')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 11 — ANÁLISE ESTATÍSTICA NO DOMÍNIO TEMPORAL
# ═════════════════════════════════════════════════════════════════════════════

def analise_estatistica(sinais_acel: dict, fs: float, rpm: float):
    """
    Indicadores estatísticos: RMS, Pico, Fator de Crista, Kurtose, Skewness.
    O usuário escolhe qual indicador destacar na distribuição de amplitude.
    """
    box("MÓDULO DIDÁTICO 6 — INDICADORES ESTATÍSTICOS NO TEMPO")

    print("""
  Indicadores estatísticos são calculados diretamente no sinal temporal.
  São muito sensíveis a IMPACTOS (falhas incipientes de rolamento).

  ┌──────────────────────────────────────────────────────────────┐
  │  Indicador      │ Fórmula            │ Alerta típico         │
  ├──────────────────────────────────────────────────────────────┤
  │  RMS            │ √(∑x²/N)          │ Depende da máquina    │
  │  Pico           │ max|x|             │ > 3×RMS               │
  │  Fator de Crista│ Pico / RMS         │ > 3,0 = impactos      │
  │  Kurtose        │ E[(x-μ)⁴]/σ⁴      │ > 3 = impacto incip.  │
  │  Skewness       │ E[(x-μ)³]/σ³      │ ≠ 0 = assimetria      │
  └──────────────────────────────────────────────────────────────┘
""")

    # ── Seleção do indicador a destacar no histograma ─────────────────────
    print("""
  INDICADOR A DESTACAR NA DISTRIBUIÇÃO DE AMPLITUDE:
  ─────────────────────────────────────────────────────────────────
  [1] Fator de Crista (CF)  — marca o pico e o RMS no histograma
  [2] Kurtose               — mostra diferença para a curva normal
  [3] Skewness              — mostra assimetria da distribuição
  [4] RMS                   — marca ±1σ e ±2σ no histograma
  [5] Todos os marcadores   — exibe tudo simultaneamente
  ─────────────────────────────────────────────────────────────────""")
    indicador_escolha = '1'
    while indicador_escolha not in ('1','2','3','4','5'):
        indicador_escolha = input("  Escolha [1-5] (padrão=1): ").strip() or '1'

    nomes_indicadores = {
        '1': 'Fator de Crista', '2': 'Kurtose',
        '3': 'Skewness',        '4': 'RMS / ±σ', '5': 'Todos'
    }
    print(f"  ✓ Destacando: {nomes_indicadores[indicador_escolha]}")

    canais_cores = [('CH2', COR_CH2), ('CH3', COR_CH3), ('CH4', COR_CH4)]
    canais_disponiveis = [(ch, cor) for ch, cor in canais_cores if ch in sinais_acel]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#1a1a2e')

    for idx, (ch, cor) in enumerate(canais_disponiveis):
        v = sinais_acel[ch]
        t = np.arange(len(v)) / fs

        rms_v  = np.sqrt(np.mean(v**2))
        pico   = np.max(np.abs(v))
        cf     = pico / (rms_v + 1e-12)
        kurt_v = float(kurtosis(v))
        skew_v = float(skew(v))
        sigma  = np.std(v)
        media  = np.mean(v)

        # ── Alertas ───────────────────────────────────────────────────────
        alerta_cf   = '⚠ ALTO' if cf > 3 else '✓'
        alerta_kurt = '⚠ ALTO' if kurt_v > 3 else '✓'

        print(f"  {ch}:")
        print(f"    RMS            : {rms_v:.6f}")
        print(f"    Pico           : {pico:.6f}")
        print(f"    Fator de Crista: {cf:.3f}  {alerta_cf}")
        print(f"    Kurtose        : {kurt_v:.4f}  {alerta_kurt}")
        print(f"    Skewness       : {skew_v:.4f}")
        print()

        # ── Sinal temporal ────────────────────────────────────────────────
        ax = axes[0, idx]
        n_plot = min(int(2 * fs), len(t))
        ax.plot(t[:n_plot], v[:n_plot], color=cor, lw=0.7, alpha=0.9)
        ax.axhline( rms_v, color='lime',   ls='--', lw=1.5, label=f'RMS={rms_v:.5f}')
        ax.axhline(-rms_v, color='lime',   ls='--', lw=1.5)
        ax.axhline( pico,  color='red',    ls=':',  lw=1.5, label=f'Pico={pico:.5f}')
        ax.axhline(-pico,  color='red',    ls=':',  lw=1.5)
        titulo_cf = f'CF={cf:.2f} {"⚠" if cf>3 else "✓"}  |  K={kurt_v:.2f} {"⚠" if kurt_v>3 else "✓"}'
        ax.set_title(f'{ch} — {titulo_cf}', color='white', fontsize=10)
        ax.set_xlabel('Tempo (s)'); ax.set_ylabel(get_ylabel_tempo())
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # ── Histograma com indicador escolhido ────────────────────────────
        ax2 = axes[1, idx]
        n_bins = min(150, len(v) // 100)
        ax2.hist(v, bins=n_bins, color=cor, alpha=0.65,
                 edgecolor='none', density=True, label='Distribuição')

        # Curva normal de referência (sempre mostrada)
        x_norm = np.linspace(v.min(), v.max(), 400)
        from scipy.stats import norm as sp_norm
        ax2.plot(x_norm, sp_norm.pdf(x_norm, media, sigma),
                 'white', lw=2.5, ls='--', label=f'Normal (K=3, Sk=0)')

        # ── Marcadores do indicador escolhido ─────────────────────────────
        def _marca_cf():
            ax2.axvline( pico, color='red',   ls='-',  lw=2.5,
                         label=f'Pico={pico:.4f}')
            ax2.axvline(-pico, color='red',   ls='-',  lw=2.5)
            ax2.axvline( rms_v, color='lime', ls='--', lw=2,
                         label=f'RMS={rms_v:.4f}')
            ax2.axvline(-rms_v, color='lime', ls='--', lw=2)
            ymax = ax2.get_ylim()[1] * 0.6
            ax2.annotate(f'CF={cf:.2f}', xy=(pico, ymax),
                         color='red', fontsize=10, fontweight='bold',
                         ha='right',
                         bbox=dict(facecolor='#330000', alpha=0.7, lw=0))

        def _marca_kurtose():
            # Mostrar a diferença entre a distribuição real e a normal
            # gerando curva com a kurtose real (aproximação Gram–Charlier)
            ax2.fill_between(x_norm,
                             sp_norm.pdf(x_norm, media, sigma),
                             alpha=0.25, color='#ff9900',
                             label=f'Diferença (K={kurt_v:.2f})')
            ymax = ax2.get_ylim()[1] * 0.8
            cor_k = '#ff4444' if kurt_v > 3 else COR_OK
            ax2.text(media, ymax,
                     f'Kurtose = {kurt_v:.3f}\n'
                     f'{"⚠ Impulsivo!" if kurt_v>3 else "✓ Normal"}',
                     ha='center', color=cor_k, fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, lw=0))

        def _marca_skewness():
            # Setas indicando o lado de assimetria
            ax2.axvline(media,  color='yellow',  ls='-',  lw=2,
                        label=f'Média={media:.4f}')
            ax2.axvline(np.median(v), color='#aaddff', ls='--', lw=2,
                        label=f'Mediana={np.median(v):.4f}')
            ymax = ax2.get_ylim()[1] * 0.7
            cor_sk = '#ff9900' if abs(skew_v) > 0.5 else COR_OK
            ax2.annotate('', xy=(media + skew_v * sigma * 0.8, ymax * 0.8),
                         xytext=(media, ymax * 0.8),
                         arrowprops=dict(arrowstyle='->', color=cor_sk, lw=2.5))
            ax2.text(media, ymax,
                     f'Skewness = {skew_v:.3f}\n'
                     f'{"⚠ Assimétrico" if abs(skew_v)>0.5 else "✓ Simétrico"}',
                     ha='center', color=cor_sk, fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, lw=0))

        def _marca_rms():
            for n_sig, cor_s, ls_s in [(1, 'lime', '-'), (2, 'yellow', '--'),
                                        (3, 'red', ':')]:
                ax2.axvline( media + n_sig * sigma, color=cor_s, ls=ls_s, lw=2,
                             label=f'+{n_sig}σ')
                ax2.axvline( media - n_sig * sigma, color=cor_s, ls=ls_s, lw=2)
            ymax = ax2.get_ylim()[1] * 0.75
            ax2.text(media, ymax,
                     f'RMS={rms_v:.4f}\nσ={sigma:.4f}',
                     ha='center', color='lime', fontsize=10, fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.6, lw=0))

        if   indicador_escolha == '1': _marca_cf()
        elif indicador_escolha == '2': _marca_kurtose()
        elif indicador_escolha == '3': _marca_skewness()
        elif indicador_escolha == '4': _marca_rms()
        elif indicador_escolha == '5':
            _marca_cf(); _marca_kurtose(); _marca_rms()

        ax2.set_title(
            f'{ch} — Distribuição  |  '
            f'Destacando: {nomes_indicadores[indicador_escolha]}',
            color='white', fontsize=9)
        ax2.set_xlabel('Amplitude'); ax2.set_ylabel('Densidade')
        ax2.legend(fontsize=7, ncol=2); ax2.grid(True, alpha=0.3, axis='y')

    # Esconder axes não usados
    for j in range(len(canais_disponiveis), 3):
        axes[0, j].axis('off')
        axes[1, j].axis('off')

    fig.suptitle(
        f'INDICADORES ESTATÍSTICOS — RPM: {rpm:.1f} | '
        f'Destaque: {nomes_indicadores[indicador_escolha]}',
        fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, 'estatisticas_temporais')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 12b — SÉRIES TEMPORAIS
# ═════════════════════════════════════════════════════════════════════════════

def plot_series_temporais(sinais_acel: dict, sinal_taco: np.ndarray,
                          fs: float, rpm: float):
    """
    Plots interativos das séries temporais com opções de zoom,
    filtro e seleção de janela temporal.
    """
    box("MÓDULO — SÉRIES TEMPORAIS")

    freq_rot = rpm / 60.0 if rpm else None
    t_total  = len(list(sinais_acel.values())[0]) / fs

    rpm_info = f"RPM={rpm:.1f}" if rpm else "RPM não detectado"
    print(f"  Duração total: {t_total:.3f} s  |  Fs={fs:.0f} Hz  |  {rpm_info}")

    while True:
        idx_op = escolha_menu([
            'Todos os canais sobrepostos (CH2, CH3, CH4)',
            'Canal individual com zoom interativo',
            'Janela temporal selecionável (início + duração)',
            'Sinal filtrado (passa-banda selecionável)',
            'Tacômetro (CH1)',
            'Voltar',
        ], "Tipo de plot de série temporal")

        if idx_op == 5:
            break

        elif idx_op == 0:
            # Todos sobrepostos
            t_plot = entrada_float("  Duração a plotar (s)", padrao=min(2.0, t_total),
                                   minval=0.01, maxval=t_total)
            n_pts = int(t_plot * fs)
            t_arr = np.arange(n_pts) / fs

            fig, axes = plt.subplots(3, 1, figsize=(18, 10), facecolor='#1a1a2e',
                                     sharex=True)
            for (ch, cor), ax in zip(
                    [('CH2', COR_CH2), ('CH3', COR_CH3), ('CH4', COR_CH4)], axes):
                if ch in sinais_acel:
                    v = sinais_acel[ch][:n_pts]
                    rms_v = np.sqrt(np.mean(v**2))
                    ax.plot(t_arr, v, color=cor, lw=0.7)
                    ax.axhline( rms_v, color='white', ls='--', lw=1, alpha=0.5,
                                label=f'RMS={rms_v:.5f}')
                    ax.axhline(-rms_v, color='white', ls='--', lw=1, alpha=0.5)
                    ax.set_ylabel(ch, fontsize=11)
                    ax.legend(fontsize=8, loc='upper right')
                    ax.grid(True, alpha=0.3)

                    # Marcar pulsos do tacômetro se disponíveis
                    if freq_rot:
                        T_rot = 1.0 / freq_rot
                        for tp in np.arange(0, t_plot, T_rot):
                            ax.axvline(tp, color='gray', lw=0.5, alpha=0.3)

            axes[-1].set_xlabel('Tempo (s)', fontsize=11)
            rpm_str = f'{rpm:.1f}' if rpm else 'N/A'
            fig.suptitle(f'SÉRIES TEMPORAIS — RPM: {rpm_str}  '
                         f'| Fs={fs:.0f} Hz',
                         fontsize=13, fontweight='bold', color='white')
            plt.tight_layout()
            salvar_figura(fig, 'series_temporais_todas')
            plt.show()

        elif idx_op == 1:
            # Canal individual
            ch_idx = escolha_menu(
                ['CH2', 'CH3', 'CH4'], "Canal")
            ch = ['CH2', 'CH3', 'CH4'][ch_idx]
            cor = [COR_CH2, COR_CH3, COR_CH4][ch_idx]
            v = sinais_acel.get(ch)
            if v is None:
                print(f"  Canal {ch} indisponível.")
                continue

            t_plot = entrada_float("  Duração a plotar (s)",
                                   padrao=min(2.0, t_total), minval=0.01, maxval=t_total)
            n_pts = int(t_plot * fs)
            t_arr = np.arange(n_pts) / fs

            rms_v  = np.sqrt(np.mean(v[:n_pts]**2))
            pico_v = np.max(np.abs(v[:n_pts]))

            fig, ax = plt.subplots(figsize=(18, 6), facecolor='#1a1a2e')
            ax.plot(t_arr, v[:n_pts], color=cor, lw=0.7, label=ch)
            ax.axhline( rms_v,  color='lime', ls='--', lw=1.5,
                        label=f'RMS={rms_v:.5f}')
            ax.axhline(-rms_v,  color='lime', ls='--', lw=1.5)
            ax.axhline( pico_v, color='red',  ls=':',  lw=1.5,
                        label=f'Pico={pico_v:.5f}')
            ax.axhline(-pico_v, color='red',  ls=':',  lw=1.5)
            ax.set_xlabel('Tempo (s)', fontsize=11)
            ax.set_ylabel(get_ylabel_tempo(), fontsize=11)
            ax.set_title(f'{ch} — CF={pico_v/rms_v:.2f} | K={float(kurtosis(v[:n_pts])):.2f}',
                         color='white', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            salvar_figura(fig, f'serie_{ch}')
            plt.show()

        elif idx_op == 2:
            # Janela temporal selecionável
            ch_idx = escolha_menu(['CH2', 'CH3', 'CH4'], "Canal")
            ch  = ['CH2', 'CH3', 'CH4'][ch_idx]
            cor = [COR_CH2, COR_CH3, COR_CH4][ch_idx]
            v   = sinais_acel.get(ch)
            if v is None:
                print(f"  Canal {ch} indisponível."); continue

            t_ini  = entrada_float(f"  Início da janela (s) [0 a {t_total:.2f}]",
                                   padrao=0.0, minval=0.0, maxval=t_total - 0.01)
            dur    = entrada_float(f"  Duração da janela (s)",
                                   padrao=min(0.5, t_total - t_ini),
                                   minval=0.001, maxval=t_total - t_ini)
            i0 = int(t_ini * fs)
            i1 = min(i0 + int(dur * fs), len(v))
            t_arr = np.arange(i1 - i0) / fs + t_ini
            seg   = v[i0:i1]

            rms_s = np.sqrt(np.mean(seg**2))
            pico_s = np.max(np.abs(seg))

            fig, ax = plt.subplots(figsize=(18, 6), facecolor='#1a1a2e')
            ax.plot(t_arr, seg, color=cor, lw=0.9, label=ch)
            ax.axhline( rms_s, color='lime', ls='--', lw=1.5,
                        label=f'RMS={rms_s:.5f}')
            ax.axhline(-rms_s, color='lime', ls='--', lw=1.5)
            ax.axhline( pico_s, color='red', ls=':', lw=1.5,
                        label=f'Pico={pico_s:.5f}')
            ax.axhline(-pico_s, color='red', ls=':', lw=1.5)

            # Marcar revoluções
            if freq_rot:
                T_rot = 1.0 / freq_rot
                for tp in np.arange(t_ini, t_ini + dur, T_rot):
                    ax.axvline(tp, color='gray', lw=1, alpha=0.4, ls=':')

            ax.set_xlabel('Tempo (s)', fontsize=11)
            ax.set_ylabel(get_ylabel_tempo(), fontsize=11)
            ax.set_title(
                f'{ch} — Janela [{t_ini:.3f}–{t_ini+dur:.3f} s] | '
                f'RMS={rms_s:.5f} | CF={pico_s/rms_s:.2f} | '
                f'K={float(kurtosis(seg)):.2f}',
                color='white', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            ax.set_xlim([t_ini, t_ini + dur])
            plt.tight_layout()
            salvar_figura(fig, f'serie_janela_{ch}')
            plt.show()

        elif idx_op == 3:
            # Sinal filtrado
            ch_idx = escolha_menu(['CH2', 'CH3', 'CH4'], "Canal")
            ch  = ['CH2', 'CH3', 'CH4'][ch_idx]
            cor = [COR_CH2, COR_CH3, COR_CH4][ch_idx]
            v   = sinais_acel.get(ch)
            if v is None:
                print(f"  Canal {ch} indisponível."); continue

            fmax_disp = fs / CONSTANTE_NYQUIST
            print(f"  Fmax={fmax_disp:.0f} Hz")
            if freq_rot:
                print(f"  Sugestões: 1X={freq_rot:.1f} Hz  "
                      f"2X={2*freq_rot:.1f}  5X={5*freq_rot:.1f}  10X={10*freq_rot:.1f}")
            f_low  = entrada_float("  Frequência inferior do filtro (Hz)",
                                   padrao=freq_rot * 0.5 if freq_rot else 1.0,
                                   minval=0.5, maxval=fs/2 - 100)
            f_high = entrada_float("  Frequência superior do filtro (Hz)",
                                   padrao=min(500.0, fmax_disp * 0.9),
                                   minval=f_low + 1, maxval=fs/2 - 1)
            t_plot = entrada_float("  Duração a plotar (s)",
                                   padrao=min(2.0, t_total), minval=0.01, maxval=t_total)

            sos  = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
            v_f  = signal.sosfilt(sos, v)
            n_pts = int(t_plot * fs)
            t_arr = np.arange(n_pts) / fs

            fig, axes = plt.subplots(2, 1, figsize=(18, 9), facecolor='#1a1a2e',
                                     sharex=True)
            axes[0].plot(t_arr, v[:n_pts], color=cor, lw=0.7, alpha=0.8,
                         label='Original')
            axes[0].set_ylabel(get_ylabel_tempo()); axes[0].legend(fontsize=9)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title(f'{ch} — Original', color='white')

            axes[1].plot(t_arr, v_f[:n_pts], color='#aaddff', lw=1.0,
                         label=f'Filtrado [{f_low:.0f}–{f_high:.0f} Hz]')
            rms_f = np.sqrt(np.mean(v_f[:n_pts]**2))
            axes[1].axhline( rms_f, color='lime', ls='--', lw=1.5,
                             label=f'RMS={rms_f:.5f}')
            axes[1].axhline(-rms_f, color='lime', ls='--', lw=1.5)
            axes[1].set_xlabel('Tempo (s)'); axes[1].set_ylabel(get_ylabel_tempo())
            axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
            axes[1].set_title(f'Filtrado [{f_low:.0f}–{f_high:.0f} Hz]',
                              color='white')

            fig.suptitle(f'SÉRIE TEMPORAL FILTRADA — {ch}  '
                         f'CF={np.max(np.abs(v_f[:n_pts]))/(rms_f+1e-12):.2f}',
                         fontsize=13, fontweight='bold', color='white')
            plt.tight_layout()
            salvar_figura(fig, f'serie_filtrada_{ch}')
            plt.show()

        elif idx_op == 4:
            # Tacômetro
            n_pts = int(min(1.0, t_total) * fs)
            t_arr = np.arange(n_pts) / fs
            fig, ax = plt.subplots(figsize=(18, 5), facecolor='#1a1a2e')
            ax.plot(t_arr, sinal_taco[:n_pts], color=COR_TACO, lw=0.8)
            ax.set_title('CH1 — Tacômetro (1 s)', color='white',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Tensão (V)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            salvar_figura(fig, 'serie_tacometro')
            plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 12 — MENU PRINCIPAL INTERATIVO
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 11b — ANÁLISE DE ENVELOPE (DEMODULAÇÃO AM)
# ═════════════════════════════════════════════════════════════════════════════

def analise_envelope(sinais_acel: dict, fs: float, rpm: float,
                     freq_rot: float = None):
    """
    Análise de envelope para detecção de falhas em rolamentos.

    Passos:
      1. Filtro passa-banda ao redor de uma frequência de ressonância
      2. Transformada de Hilbert → sinal analítico → envelope
      3. FFT do envelope → espectro de amplitude modulada
      4. Marcação das frequências características dos rolamentos
    """
    box("MÓDULO DIDÁTICO — ANÁLISE DE ENVELOPE (Demodulação AM)")

    if freq_rot is None:
        freq_rot = rpm / 60.0

    print(f"""
  PRINCÍPIO DA ANÁLISE DE ENVELOPE:
  ──────────────────────────────────────────────────────────────────
  Defeitos de rolamento geram IMPACTOS de curta duração que excitam
  frequências naturais da estrutura (tipicamente 1–20 kHz).

  Esses impactos se repetem na frequência BPFO, BPFI ou BSF,
  gerando uma MODULAÇÃO EM AMPLITUDE (AM) em alta frequência.

  O envelope captura essa modulação:
    ① Filtro passa-banda  → isola a banda de ressonância
    ② Transformada de Hilbert → extrai o envelope instantâneo
    ③ FFT do envelope  → revela as frequências de impacto

  Fmax útil para rolamentos: geralmente 5–10× BPFI
  Com Fs={fs:.0f} Hz → Fmax={fs/CONSTANTE_NYQUIST:.0f} Hz
  ──────────────────────────────────────────────────────────────────
""")

    # ── Canal ────────────────────────────────────────────────────────────────
    ch_idx = escolha_menu(
        ['CH2 — Radial Vertical', 'CH3 — Radial Horizontal', 'CH4 — Axial'],
        "Canal para análise de envelope"
    )
    canais_list = ['CH2', 'CH3', 'CH4']
    canal = canais_list[ch_idx]
    sinal = sinais_acel.get(canal)
    if sinal is None:
        print(f"  [ERRO] Canal {canal} não disponível.")
        return
    cor = [COR_CH2, COR_CH3, COR_CH4][ch_idx]

    # ── Frequências dos rolamentos ────────────────────────────────────────────
    print("""
  FREQUÊNCIAS DOS ROLAMENTOS:
  ──────────────────────────────────────────────────────────────────
  Os fatores são adimensionais (por revolução).
  A frequência real = fator × fr   onde fr = RPM/60.

  [1] Usar fatores do catálogo (BPFO, BPFI, BSF, FTF por revolução)
  [2] Inserir frequências em Hz diretamente
""")
    modo_freq = '0'
    while modo_freq not in ('1', '2'):
        modo_freq = input("  Escolha [1/2]: ").strip()

    if modo_freq == '1':
        print(f"  (RPM={rpm:.1f} → fr={freq_rot:.4f} Hz)")
        print("  Exemplo para este rolamento: BPFO=3,584 BPFI=5,416 BSF=4,71 FTF=0,398")
        fat_BPFO = entrada_float("  Fator BPFO (0=omitir)", padrao=0, minval=0)
        fat_BPFI = entrada_float("  Fator BPFI (0=omitir)", padrao=0, minval=0)
        fat_BSF  = entrada_float("  Fator BSF  (0=omitir)", padrao=0, minval=0)
        fat_BRF  = entrada_float("  Fator BRF  (0=omitir)", padrao=0, minval=0)
        fat_FTF  = entrada_float("  Fator FTF  (0=omitir)", padrao=0, minval=0)
        freqs_rol = {}
        if fat_BPFO > 0: freqs_rol['BPFO'] = fat_BPFO * freq_rot
        if fat_BPFI > 0: freqs_rol['BPFI'] = fat_BPFI * freq_rot
        if fat_BSF  > 0: freqs_rol['BSF']  = fat_BSF  * freq_rot
        if fat_BRF  > 0: freqs_rol['BRF']  = fat_BRF  * freq_rot
        if fat_FTF  > 0: freqs_rol['FTF']  = fat_FTF  * freq_rot
    else:
        freqs_rol = {}
        for nome in ['BPFO', 'BPFI', 'BSF', 'BRF', 'FTF']:
            v = entrada_float(f"  {nome} em Hz (0=omitir)", padrao=0, minval=0)
            if v > 0:
                freqs_rol[nome] = v

    print(f"\n  Frequências configuradas:")
    for k, v in freqs_rol.items():
        print(f"    {k:<6} = {v:.4f} Hz")

    # ── Filtro de banda para envelope ─────────────────────────────────────────
    fmax_disp = fs / CONSTANTE_NYQUIST
    print(f"""
  FILTRO PASSA-BANDA (isola a banda de ressonância):
  ──────────────────────────────────────────────────────────────────
  Escolha uma faixa que contenha a ressonância da estrutura onde
  o acelerômetro está montado.

  Estratégia recomendada:
    → Olhe o espectro de alta frequência
    → Identifique regiões com amplitude elevada acima de ~500 Hz
    → Use essa faixa como banda de demodulação

  Para Fs={fs:.0f} Hz, sugestões comuns:
    • Banda larga : {fmax_disp*0.3:.0f} – {fmax_disp*0.9:.0f} Hz
    • Banda média : {fmax_disp*0.1:.0f} – {fmax_disp*0.5:.0f} Hz
    • Zona rolam. : 500 – 3000 Hz  (típico para rolamentos de médio porte)
  ──────────────────────────────────────────────────────────────────""")

    f_low  = entrada_float("  Frequência inferior do filtro (Hz)",
                           padrao=500, minval=10, maxval=fs / 2 - 100)
    f_high = entrada_float("  Frequência superior do filtro (Hz)",
                           padrao=min(3000, fmax_disp * 0.9),
                           minval=f_low + 10, maxval=fs / 2 - 1)

    # Fmax do espectro de envelope
    fmax_env = entrada_float(
        "  Freq. máxima do espectro de envelope (Hz)",
        padrao=min(max(freqs_rol.values()) * 5 if freqs_rol else 500, fmax_disp),
        minval=10, maxval=fmax_disp)

    escala_env = selecionar_escala()

    # ── Processamento ─────────────────────────────────────────────────────────
    print("\n  Processando...")

    # Filtro passa-banda (Butterworth 4ª ordem)
    sos_bp = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
    sinal_bp = signal.sosfilt(sos_bp, sinal)

    # Envelope via transformada de Hilbert
    analitico  = signal.hilbert(sinal_bp)
    envelope   = np.abs(analitico)

    # Remover DC do envelope
    envelope -= np.mean(envelope)

    # Espectro do envelope
    freqs_env, amp_env = calcular_espectro(envelope, fs, 'Hann', 'hann')

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16), facecolor='#1a1a2e')
    gs_fig = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    t = np.arange(len(sinal)) / fs
    t_max_plot = min(2.0, t[-1])
    mask_t = t <= t_max_plot

    # ── Linha 1: Sinal original e filtrado ────────────────────────────────────
    ax1 = fig.add_subplot(gs_fig[0, 0])
    ax1.plot(t[mask_t], sinal[mask_t], color=cor, lw=0.7, alpha=0.8, label='Original')
    ax1.set_title(f'{canal} — Sinal Original (DC removido)', color='white', fontsize=10)
    ax1.set_xlabel('Tempo (s)'); ax1.set_ylabel(get_ylabel_tempo())
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs_fig[0, 1])
    ax2.plot(t[mask_t], sinal_bp[mask_t], color='#aaddff', lw=0.8, label='Filtrado')
    ax2.set_title(f'Passa-Banda [{f_low:.0f}–{f_high:.0f} Hz]', color='white', fontsize=10)
    ax2.set_xlabel('Tempo (s)'); ax2.set_ylabel(get_ylabel_tempo())
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    # ── Linha 2: Envelope no tempo ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs_fig[1, :])
    ax3.plot(t[mask_t], sinal_bp[mask_t], color='#aaddff', lw=0.6, alpha=0.5,
             label='Sinal filtrado')
    ax3.plot(t[mask_t], envelope[mask_t], color=COR_DEST, lw=2,
             label='Envelope (|Hilbert|)')
    ax3.plot(t[mask_t], -envelope[mask_t], color=COR_DEST, lw=2, alpha=0.6)

    # Marcar os períodos esperados de BPFO e BPFI
    cores_rol = {'BPFO': '#ff4444', 'BPFI': '#aa44ff',
                 'BSF': '#44ff88', 'BRF': '#00d4ff', 'FTF': '#ffdd00'}
    for nome, fval in freqs_rol.items():
        if fval > 0:
            periodo = 1.0 / fval
            cor_r = cores_rol.get(nome, 'white')
            k = 0
            while k * periodo < t_max_plot:
                ax3.axvline(k * periodo, color=cor_r, ls=':', lw=1, alpha=0.5)
                k += 1
            # Só rotular uma vez
            ax3.axvline(0, color=cor_r, ls=':', lw=2, alpha=0,
                        label=f'{nome}={fval:.2f}Hz (T={periodo*1000:.1f}ms)')

    ax3.set_title('Envelope Temporal — Marcas de Período dos Rolamentos',
                  color='white', fontsize=11)
    ax3.set_xlabel('Tempo (s)'); ax3.set_ylabel(get_ylabel_tempo())
    ax3.legend(fontsize=8, ncol=4); ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, t_max_plot])

    # ── Linha 3: Espectro do envelope ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs_fig[2, :])
    offset_env = 1e-12 if escala_env == 'log' else 0.0
    ax4.plot(freqs_env, amp_env + offset_env, color=COR_DEST, lw=1.2,
             label='Espectro do Envelope')
    _aplicar_escala(ax4, escala_env, amp_env)

    # Harmônicas de rotação (linhas brancas finas)
    for k_r in range(1, 11):
        fk = k_r * freq_rot
        if fk <= fmax_env:
            ax4.axvline(fk, color='white', ls=':', lw=0.8, alpha=0.35)

    # Frequências dos rolamentos e suas harmônicas
    amp_max_env = amp_env[freqs_env <= fmax_env].max() if len(amp_env) > 0 else 1.0
    for nome, fval in freqs_rol.items():
        cor_r = cores_rol.get(nome, 'white')
        for k_h in range(1, 6):
            fk = k_h * fval
            if fk <= fmax_env:
                ax4.axvline(fk, color=cor_r, ls='--', lw=2.0, alpha=0.85,
                            label=f'{nome}' if k_h == 1 else '')
                ypos = (amp_max_env * (0.85 - 0.12 * (k_h - 1))
                        if escala_env == 'linear' else amp_max_env ** 0.7)
                ax4.text(fk, ypos,
                         f'{k_h}×{nome}\n{fk:.1f}Hz',
                         color=cor_r, fontsize=7, ha='center', fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.4, lw=0, pad=1))

    ax4.set_xlim([0, fmax_env])
    ax4.set_xlabel('Frequência (Hz)', fontsize=11)
    ax4.set_title(
        f'Espectro de Envelope — Canal: {canal} | '
        f'Banda: [{f_low:.0f}–{f_high:.0f} Hz] | '
        f'Escala: {"Log" if escala_env=="log" else "Linear"}',
        color='white', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, ncol=len(freqs_rol) + 1)
    ax4.grid(True, alpha=0.3)

    # ── Linha 4: Tabela de amplitudes nas frequências de rolamento ────────────
    ax5 = fig.add_subplot(gs_fig[3, :])
    ax5.axis('off')

    linhas_tab = ["AMPLITUDES DO ESPECTRO DE ENVELOPE NAS FREQUÊNCIAS DOS ROLAMENTOS\n"]
    linhas_tab.append(f"  {'Frequência':<8} {'Hz':<10} {'1×Amp':>12} {'2×Amp':>12} "
                      f"{'3×Amp':>12} {'SNR (dB)':>12}")
    linhas_tab.append("  " + "─" * 66)

    noise_floor = np.median(amp_env)  # nível de ruído estimado

    for nome, fval in freqs_rol.items():
        amps_harm = []
        for k_h in range(1, 4):
            fk = k_h * fval
            if fk <= freqs_env[-1]:
                idx_f = np.argmin(np.abs(freqs_env - fk))
                amps_harm.append(amp_env[idx_f])
            else:
                amps_harm.append(0.0)
        snr_db = 20 * np.log10(amps_harm[0] / (noise_floor + 1e-15)) if amps_harm[0] > 0 else -99
        linhas_tab.append(
            f"  {nome:<8} {fval:<10.3f} {amps_harm[0]:>12.6f} {amps_harm[1]:>12.6f} "
            f"{amps_harm[2]:>12.6f} {snr_db:>10.1f} dB"
            + ("  ← ⚠ ALTO" if snr_db > 10 else ""))

    linhas_tab.append(f"\n  Nível de ruído estimado (mediana): {noise_floor:.6f}")
    linhas_tab.append("  SNR > 10 dB sugere componente significativa!")

    ax5.text(0.01, 0.95, "\n".join(linhas_tab),
             transform=ax5.transAxes, fontsize=9,
             va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#0f3460',
                       edgecolor='#aaaaaa', alpha=0.95, lw=1.5))

    fig.suptitle(
        f'ANÁLISE DE ENVELOPE — {canal} | RPM: {rpm:.1f} | '
        f'fr={freq_rot:.2f} Hz | Fs={fs:.0f} Hz',
        fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, f'envelope_{canal}')
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 11c — FAST KURTOGRAM (identificação da banda ótima de demodulação)
# ═════════════════════════════════════════════════════════════════════════════

def fast_kurtogram(sinais_acel: dict, fs: float, rpm: float):
    """
    Kurtograma rápido: divide o espectro em bandas e calcula a kurtose
    de cada banda. A banda com maior kurtose é a mais impulsiva
    (melhor candidata para a análise de envelope).

    Implementação simplificada por banco de filtros de 1/3 de oitava.
    """
    box("MÓDULO — FAST KURTOGRAM (Banda Ótima para Demodulação)")

    print("""
  O KURTOGRAMA mostra a kurtose espectral em função da frequência
  central e da largura de banda do filtro.

  Alta kurtose → sinal impulsivo → provável defeito de rolamento!

  ╔═══════════════════════════════════════════════════════════════╗
  ║  Kurtose > 3 : impulsividade acima do esperado (Gaussiano)   ║
  ║  Kurtose > 10: sinal altamente impulsivo → defeito claro     ║
  ║  A banda com MAIOR kurtose é a ideal para o envelope         ║
  ╚═══════════════════════════════════════════════════════════════╝
""")

    ch_idx = escolha_menu(
        ['CH2 — Radial Vertical', 'CH3 — Radial Horizontal', 'CH4 — Axial'],
        "Canal para Kurtograma"
    )
    canais_list = ['CH2', 'CH3', 'CH4']
    canal = canais_list[ch_idx]
    sinal = sinais_acel.get(canal)
    if sinal is None:
        print(f"  [ERRO] Canal {canal} não disponível.")
        return
    cor = [COR_CH2, COR_CH3, COR_CH4][ch_idx]

    fmax_disp  = fs / CONSTANTE_NYQUIST
    freq_rot   = rpm / 60.0

    # Começa acima das harmônicas de rotação para não capturar 1X/2X/3X
    fc_min = max(100.0, 10 * freq_rot)
    fc_max = fmax_disp * 0.9

    print(f"\n  Calculando kurtograma para [{fc_min:.0f}–{fc_max:.0f} Hz]...")
    print(f"  (início em {fc_min:.0f} Hz = 10×fr para evitar harmônicas de rotação)\n")

    # ── Banco de filtros em 1/3 de oitava ────────────────────────────────────
    # Frequências centrais em 1/3 oitava de 10 Hz até Fmax
    n_bands  = int(np.log2(fc_max / fc_min) * 3) + 1
    fcs      = fc_min * 2 ** (np.arange(n_bands) / 3.0)
    fcs      = fcs[fcs <= fc_max]

    # Largura de banda de cada filtro (1/3 oitava)
    fator_bw = 2 ** (1 / 6)   # metade da largura de 1/3 oitava
    resultados = []

    for fc in fcs:
        fl = fc / fator_bw
        fh = min(fc * fator_bw, fs / 2 - 1)
        if fl < 1.0 or fh <= fl:
            continue
        try:
            sos_k = signal.butter(4, [fl, fh], btype='band', fs=fs, output='sos')
            s_filt = signal.sosfilt(sos_k, sinal)
            env_k  = np.abs(signal.hilbert(s_filt))
            env_k -= np.mean(env_k)
            kurt_k = float(kurtosis(env_k))
            resultados.append({'fc': fc, 'fl': fl, 'fh': fh,
                                'bw': fh - fl, 'kurtose': kurt_k})
        except Exception:
            continue

    if not resultados:
        print("  [ERRO] Nenhuma banda processada.")
        return

    fcs_arr   = np.array([r['fc']      for r in resultados])
    bws_arr   = np.array([r['bw']      for r in resultados])
    kurts_arr = np.array([r['kurtose'] for r in resultados])

    idx_best  = np.argmax(kurts_arr)
    fc_best   = fcs_arr[idx_best]
    fl_best   = resultados[idx_best]['fl']
    fh_best   = resultados[idx_best]['fh']
    kurt_best = kurts_arr[idx_best]

    print(f"\n  ✓ Banda com MAIOR kurtose:")
    print(f"    Fc = {fc_best:.1f} Hz  |  [{fl_best:.1f} – {fh_best:.1f} Hz]")
    print(f"    Kurtose = {kurt_best:.2f}")

    if kurt_best < 3:
        print("""
  ⚠  KURTOSE < 3 EM TODAS AS BANDAS
  ─────────────────────────────────────────────────────────
  A kurtose do envelope está abaixo de 3 (valor Gaussiano).
  Isso indica que NÃO há impactos impulsivos claros neste sinal.

  Interpretações possíveis:
    • Rolamento em estado SAUDÁVEL ou com defeito muito incipiente
    • Defeito mascarado por ruído de fundo elevado (desbalanceamento,
      desalinhamento, BPF de ventilador)
    • Ressonâncias do sistema fortemente amortecidas
    • Acelerômetro mal posicionado (longe do rolamento)

  Neste sinal, as análises indicam:
    → 3X axial (CH4) dominante → DESALINHAMENTO como causa principal
    → Frequências de rolamento (BPFO/BPFI) abaixo do ruído de fundo
    → Recomenda-se investigar alinhamento antes de suspeitar de rolamento
  ─────────────────────────────────────────────────────────""")
    elif kurt_best < 10:
        print(f"\n  → Kurtose moderada — possível defeito INCIPIENTE de rolamento.")
        print(f"  → Monitorar evolução ao longo do tempo.")
    else:
        print(f"\n  → Kurtose alta — defeito de rolamento SIGNIFICATIVO!")
        print(f"  → Recomenda-se análise de envelope nesta banda.")

    print(f"\n  → Use a banda [{fl_best:.0f}–{fh_best:.0f} Hz] no módulo de Análise de Envelope (opção A)!")

    # ── Gráfico ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor='#1a1a2e')

    # Kurtose vs frequência central (barras coloridas por kurtose)
    ax = axes[0]
    norm_kurt = (kurts_arr - kurts_arr.min()) / (kurts_arr.max() - kurts_arr.min() + 1e-9)
    cores_k = plt.cm.hot(norm_kurt)
    ax.bar(fcs_arr, kurts_arr, width=bws_arr * 0.7,
           color=cores_k, edgecolor='none', alpha=0.9)
    ax.axhline(3.0, color='yellow', ls='--', lw=2, label='Gaussiano (K=3)')
    ax.axhline(10.0, color='red', ls='--', lw=2, label='Altamente impulsivo (K=10)')
    ax.axvline(fc_best, color='#ff9900', ls='-', lw=3,
               label=f'Banda ótima: {fc_best:.0f} Hz (K={kurt_best:.1f})')
    ax.set_xlabel('Frequência central da banda (Hz)', fontsize=11)
    ax.set_ylabel('Kurtose do envelope', fontsize=11)
    ax.set_title(f'Kurtose Espectral por Banda (1/3 Oitava) — {canal}',
                 color='white', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Espectro do sinal completo com banda ótima destacada
    ax2 = axes[1]
    freqs_s, amp_s = calcular_espectro(sinal, fs, 'Hann', 'hann')
    ax2.semilogy(freqs_s, amp_s + 1e-12, color=cor, lw=0.8, alpha=0.8, label=canal)
    ax2.axvspan(fl_best, fh_best, alpha=0.25, color='#ff9900',
                label=f'Banda ótima [{fl_best:.0f}–{fh_best:.0f} Hz]')
    ax2.axvline(fl_best, color='#ff9900', ls='--', lw=2)
    ax2.axvline(fh_best, color='#ff9900', ls='--', lw=2)
    # Harmônicas de rotação
    freq_rot_loc = rpm / 60.0
    for k_r in range(1, 11):
        fk = k_r * freq_rot_loc
        if fk <= fmax_disp:
            ax2.axvline(fk, color='white', ls=':', lw=0.7, alpha=0.3)
    ax2.set_xlim([0, fmax_disp])
    ax2.set_xlabel('Frequência (Hz)', fontsize=11)
    ax2.set_ylabel('Amplitude (log)', fontsize=11)
    ax2.set_title('Espectro do Sinal com Banda Ótima de Demodulação Destacada',
                  color='white', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle(f'FAST KURTOGRAM — {canal} | RPM: {rpm:.1f}  '
                 f'→  Banda ótima: [{fl_best:.0f}–{fh_best:.0f} Hz]  K={kurt_best:.2f}',
                 fontsize=13, fontweight='bold', color='white')
    plt.tight_layout()
    salvar_figura(fig, f'kurtogram_{canal}')
    plt.show()

    # Oferecer envelope automático na banda ótima
    if input(f"\n  Rodar Análise de Envelope na banda ótima [{fl_best:.0f}–{fh_best:.0f} Hz]? [s/N]: "
             ).strip().lower() == 's':
        return fl_best, fh_best
    return None, None




# ═════════════════════════════════════════════════════════════════════════════
# SEÇÃO 13 — CALIBRAÇÃO DE ACELERÔMETROS
# ═════════════════════════════════════════════════════════════════════════════

def menu_calibracao(sinais_acel_orig: dict, estado: dict, fs: float):
    """
    Aplica (ou reverte) a calibração dos acelerômetros.

    O fator de calibração converte o sinal de Volts para unidades de
    aceleração (g ou m/s²), usando a sensibilidade medida no calibrador.

    Fluxo:
      1. Usuário informa a sensibilidade de cada canal (V/g ou fator g/V)
      2. sinais_acel é multiplicado pelo fator inverso → sinal em g (ou m/s²)
      3. O estado guarda os fatores e a unidade ativa para exibir no menu
      4. Qualquer análise posterior usa o sinal calibrado
      5. O usuário pode reverter à unidade original (V) a qualquer momento
    """
    box("MÓDULO — CALIBRAÇÃO DE ACELERÔMETROS")

    cal = estado.get('calibracao', {})   # {ch: {'fator_gV': float, 'unidade': str}}
    unidade_atual = estado.get('unidade_saida', 'V')

    print(f"""
  SITUAÇÃO ATUAL:
  ─────────────────────────────────────────────────────────────────
  Unidade de saída dos sinais    : {unidade_atual}
  Canais com calibração aplicada : {[ch for ch in cal] if cal else 'nenhum'}

  O calibrador fornece a sensibilidade do acelerômetro (V/g).
  O programa multiplica o sinal por (1 / sensibilidade) para obter g.

  Exemplo do resultado de calibração:
    Sinal medido (pico)    = 0.134611 V
    Referência             = 1.4142 g pico
    Sensitivity            = 0.095184 V/g   ← informar este valor
    Fator inverso          = 10.5059 g/V    ← ou este
  ─────────────────────────────────────────────────────────────────""")

    while True:
        idx_op = escolha_menu([
            'Inserir/atualizar calibração de um canal',
            'Importar calibração no formato do resumo (texto colado)',
            'Aplicar calibração acumulada a todos os canais configurados',
            'Reverter sinais à unidade original (Volts)',
            'Ver resumo dos fatores configurados',
            'Voltar ao menu principal',
        ], "Calibração")

        # ── 0: inserir manualmente ─────────────────────────────────────────
        if idx_op == 0:
            ch_idx = escolha_menu(['CH2', 'CH3', 'CH4'], "Canal a calibrar")
            ch = ['CH2', 'CH3', 'CH4'][ch_idx]

            print(f"""
  MODO DE ENTRADA PARA {ch}:
  [1] Sensibilidade  (V/g)   → o programa calcula 1/sens para converter
  [2] Fator inverso  (g/V)   → aplicado diretamente ao sinal
  [3] Sensitivity (mV/g)     → divide por 1000 automaticamente""")
            modo = '0'
            while modo not in ('1', '2', '3'):
                modo = input("  Escolha [1/2/3]: ").strip()

            if modo == '1':
                sens = entrada_float(f"  Sensibilidade de {ch} (V/g)", minval=1e-6)
                fator = 1.0 / sens
                print(f"  → Fator aplicado: {fator:.6f} g/V")
            elif modo == '2':
                fator = entrada_float(f"  Fator inverso de {ch} (g/V)", minval=1e-6)
            else:
                sens_mv = entrada_float(f"  Sensibilidade de {ch} (mV/g)", minval=0.001)
                fator = 1000.0 / sens_mv
                print(f"  → Sensibilidade em V/g: {sens_mv/1000:.6f}")
                print(f"  → Fator aplicado     : {fator:.6f} g/V")

            unidade_ch = '0'
            while unidade_ch not in ('1', '2'):
                print("  Unidade de saída:  [1] g   [2] m/s²  (1 g = 9.80665 m/s²)")
                unidade_ch = input("  Escolha [1/2]: ").strip() or '1'
            unidade_str = 'g' if unidade_ch == '1' else 'm/s²'
            if unidade_ch == '2':
                fator *= 9.80665

            cal[ch] = {'fator_gV': fator, 'unidade': unidade_str}
            print(f"  ✓ Calibração de {ch} configurada: × {fator:.6f} → {unidade_str}")

        # ── 1: importar do resumo colado ───────────────────────────────────
        elif idx_op == 1:
            print("""
  Cole o bloco de texto do RESUMO DA CALIBRAÇÃO gerado pelo calibrador.
  Finalize com uma linha em branco e pressione ENTER duas vezes.
  (Copie e cole o texto completo incluindo "Sensitivity" e "Canal analisado")
""")
            linhas = []
            while True:
                linha = input()
                if linha == '' and linhas and linhas[-1] == '':
                    break
                linhas.append(linha)

            texto = '\n'.join(linhas)
            import re

            # Extrair canal
            m_ch = re.search(r'Canal analisado\s*:\s*(CH\d)', texto)
            # Extrair sensitivity V/g
            m_sens = re.search(r'Sensitivity\s*:\s*([\d.]+)\s*V/g', texto)
            # Extrair fator inverso g/V
            m_fat = re.search(r'Fator inverso\s*:\s*([\d.]+)\s*g/V', texto)

            if not m_ch:
                print("  ⚠ Canal não identificado. Verifique se o texto contém 'Canal analisado : CHx'.")
                continue

            ch_imp = m_ch.group(1)

            if m_fat:
                fator_imp = float(m_fat.group(1))
                print(f"  ✓ Canal: {ch_imp}  |  Fator extraído (g/V): {fator_imp:.6f}")
            elif m_sens:
                fator_imp = 1.0 / float(m_sens.group(1))
                print(f"  ✓ Canal: {ch_imp}  |  Sensitivity: {m_sens.group(1)} V/g → Fator: {fator_imp:.6f} g/V")
            else:
                print("  ⚠ Sensitivity ou Fator inverso não encontrados no texto.")
                fator_imp = entrada_float(f"  Informe manualmente o fator g/V para {ch_imp}", minval=1e-6)

            unidade_imp = '0'
            while unidade_imp not in ('1', '2'):
                print("  Unidade de saída:  [1] g   [2] m/s²")
                unidade_imp = input("  Escolha [1/2]: ").strip() or '1'
            unidade_str_imp = 'g' if unidade_imp == '1' else 'm/s²'
            if unidade_imp == '2':
                fator_imp *= 9.80665

            cal[ch_imp] = {'fator_gV': fator_imp, 'unidade': unidade_str_imp}
            print(f"  ✓ Calibração importada para {ch_imp}: × {fator_imp:.6f} → {unidade_str_imp}")

        # ── 2: aplicar calibração ──────────────────────────────────────────
        elif idx_op == 2:
            if not cal:
                print("  ⚠ Nenhuma calibração configurada. Use as opções 1 ou 2 primeiro.")
                continue

            print("\n  APLICANDO CALIBRAÇÃO:")
            print("  " + "─" * 60)
            sinais_acel = estado['sinais_acel']
            sinais_orig = sinais_acel_orig

            for ch, cfg in cal.items():
                if ch not in sinais_orig:
                    print(f"  ⚠ Canal {ch} não disponível nos dados.")
                    continue
                fator = cfg['fator_gV']
                unid  = cfg['unidade']
                sinal_cal = sinais_orig[ch] * fator
                sinais_acel[ch] = sinal_cal
                rms_antes = np.sqrt(np.mean(sinais_orig[ch]**2))
                rms_depois = np.sqrt(np.mean(sinal_cal**2))
                pico_depois = np.max(np.abs(sinal_cal))
                print(f"  {ch}:  × {fator:.5f} → {unid}")
                print(f"       RMS: {rms_antes:.6f} V  →  {rms_depois:.6f} {unid}")
                print(f"       Pico:                   {pico_depois:.6f} {unid}")

            estado['calibracao']   = cal
            unid_nova = list(cal.values())[-1]['unidade']
            estado['unidade_saida'] = unid_nova
            _UNIDADE['saida'] = unid_nova

            # ── Plot comparativo ──────────────────────────────────────────
            chs_cal = [ch for ch in cal if ch in sinais_acel_orig]
            if not chs_cal:
                continue

            n_ch  = len(chs_cal)
            fig, axes = plt.subplots(n_ch, 2, figsize=(18, 4 * n_ch), facecolor='#1a1a2e')
            if n_ch == 1:
                axes = [axes]

            cores_ch = {'CH2': COR_CH2, 'CH3': COR_CH3, 'CH4': COR_CH4}
            fs_local = fs

            for i, ch in enumerate(chs_cal):
                cor = cores_ch.get(ch, COR_DEST)
                cfg = cal[ch]
                unid = cfg['unidade']
                v_orig = sinais_orig[ch]
                v_cal  = estado['sinais_acel'][ch]
                t = np.arange(min(int(2 * fs_local), len(v_orig))) / fs_local

                # Sinal temporal
                ax_t = axes[i][0]
                ax_t.plot(t, v_orig[:len(t)], color='gray', lw=0.7, alpha=0.7,
                          label='Original (V)')
                ax_t.plot(t, v_cal[:len(t)], color=cor, lw=0.8, alpha=0.9,
                          label=f'Calibrado ({unid})')
                ax_t.set_title(f'{ch} — Sinal Temporal', color='white', fontsize=10)
                ax_t.set_xlabel('Tempo (s)'); ax_t.set_ylabel(f'Amplitude')
                ax_t.legend(fontsize=8); ax_t.grid(True, alpha=0.3)

                # Espectro antes e depois
                ax_f = axes[i][1]
                freqs_o, amp_o = calcular_espectro(v_orig, fs_local, 'Hann', 'hann')
                freqs_c, amp_c = calcular_espectro(v_cal,  fs_local, 'Hann', 'hann')
                fmax_p = min(fs_local / CONSTANTE_NYQUIST, 1000)
                mask = freqs_o <= fmax_p
                ax_f.plot(freqs_o[mask], amp_o[mask], color='gray', lw=0.8, alpha=0.7,
                          label='Original (V RMS)')
                ax_f.plot(freqs_c[mask], amp_c[mask], color=cor, lw=1.2,
                          label=f'Calibrado ({unid} RMS)')
                ax_f.set_title(f'{ch} — Espectro (0–{fmax_p:.0f} Hz)', color='white', fontsize=10)
                ax_f.set_xlabel('Frequência (Hz)'); ax_f.set_ylabel(get_ylabel_amp())
                ax_f.legend(fontsize=8); ax_f.grid(True, alpha=0.3)

            fig.suptitle(
                'CALIBRAÇÃO APLICADA — Comparação Antes/Depois',
                fontsize=13, fontweight='bold', color='white')
            plt.tight_layout()
            salvar_figura(fig, 'calibracao_comparacao')
            plt.show()
            print("\n  ✓ Calibração aplicada. Todas as análises usarão os sinais calibrados.")

        # ── 3: reverter ────────────────────────────────────────────────────
        elif idx_op == 3:
            for ch in sinais_acel_orig:
                estado['sinais_acel'][ch] = sinais_acel_orig[ch].copy()
            estado['calibracao']   = {}
            estado['unidade_saida'] = 'V'
            _UNIDADE['saida'] = 'V'
            cal = {}
            print("  ✓ Sinais revertidos para a unidade original (Volts).")

        # ── 4: resumo ──────────────────────────────────────────────────────
        elif idx_op == 4:
            print("\n  FATORES DE CALIBRAÇÃO CONFIGURADOS:")
            print("  " + "─" * 55)
            print(f"  {'Canal':<8} {'Fator (g/V ou m/s²/V)':<26} {'Unidade'}")
            print("  " + "─" * 55)
            if cal:
                for ch, cfg in cal.items():
                    print(f"  {ch:<8} {cfg['fator_gV']:<26.6f} {cfg['unidade']}")
            else:
                print("  Nenhum fator configurado.")
            print(f"\n  Unidade de saída atual : {estado.get('unidade_saida', 'V')}")
            aplicados = [ch for ch in cal if ch in estado.get('sinais_acel', {})]
            print(f"  Calibração aplicada em : {aplicados if aplicados else 'nenhum canal'}")

        # ── 5: voltar ──────────────────────────────────────────────────────
        elif idx_op == 5:
            break

    estado['calibracao'] = cal


def menu_principal():
    """Controlador principal do fluxo educacional."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ANALISADOR DE VIBRAÇÕES EDUCACIONAL                                 ║
║          Disciplina: Vibrações Mecânicas                                     ║
║          Instituto Federal de Pernambuco — Engenharia Mecânica               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CH1 → Tacômetro     CH2 → Rad.Vert.   CH3 → Rad.Horiz.  CH4 → Axial        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # ── Localizar arquivo CSV ────────────────────────────────────────────────
    caminho = input("  Digite o caminho do arquivo CSV (ou ENTER para buscar na pasta atual): ").strip()
    if caminho == "":
        csvs = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
        if not csvs:
            print("  Nenhum CSV encontrado. Por favor informe o caminho completo.")
            caminho = input("  Caminho: ").strip()
        elif len(csvs) == 1:
            caminho = csvs[0]
            print(f"  Encontrado: {caminho}")
        else:
            idx = escolha_menu(csvs, "Arquivos CSV encontrados")
            caminho = csvs[idx]

    resultado = carregar_csv(caminho)
    if resultado is None or resultado[0] is None:
        print("  Falha ao carregar arquivo.")
        return
    df, fs_auto = resultado

    # ── Frequência de amostragem ─────────────────────────────────────────────
    if fs_auto:
        resp = input(f"  Fs detectado: {fs_auto:.1f} Hz. Confirmar? [S/n]: ").strip().lower()
        fs = fs_auto if resp != 'n' else entrada_float("  Fs (Hz)", minval=100, maxval=200000)
    else:
        fs = entrada_float("  Informe a frequência de amostragem Fs (Hz)", minval=100, maxval=200000)

    # ── Extrair arrays ───────────────────────────────────────────────────────
    sinal_taco = df['CH1'].values
    sinais_acel = {
        ch: df[ch].values for ch in ['CH2', 'CH3', 'CH4']
        if ch in df.columns and not df[ch].isnull().all()
    }
    n_amostras = len(sinal_taco)

    # Estado compartilhado entre módulos
    estado = {
        'fs': fs,
        'sinal_taco': sinal_taco,
        'sinais_acel': sinais_acel,
        'n_amostras': n_amostras,
        'indices_pulsos': None,
        'rpm': None,
        'fase_ref': None,
        'frequencias': {},
        'calibracao': {},           # {ch: {'fator_gV': float, 'unidade': str}}
        'unidade_saida': 'V',       # 'V' | 'g' | 'm/s²'
    }

    # Cópia imutável dos sinais originais (para reverter calibração)
    sinais_acel_orig = {ch: v.copy() for ch, v in sinais_acel.items()}

    # ── Loop do menu ─────────────────────────────────────────────────────────
    while True:
        print("\n" + "═" * 70)
        print("  MENU PRINCIPAL")
        print("═" * 70)
        status_rpm = f"RPM={estado['rpm']:.1f}" if estado['rpm'] else "RPM não detectado"
        unid = estado.get('unidade_saida', 'V')
        chs_cal = list(estado.get('calibracao', {}).keys())
        status_cal = f"✓ CAL:{','.join(chs_cal)}→{unid}" if chs_cal else "⚠ SEM CALIBRAÇÃO"
        print(f"  [{status_rpm}]  [Fs={fs:.0f} Hz]  [N={n_amostras}]  "
              f"[T={n_amostras/fs:.2f} s]  [{status_cal}]")
        print()

        opcoes = [
            "1. Calibração de Acelerômetros  ★ PRIMEIRO PASSO",
            "2. Séries Temporais (zoom, filtro, janela selecionável)",
            "3. Detectar RPM pelo Tacômetro (CH1)",
            "4. Análise Espectral (FFT) — Hz ou Ordem",
            "5. Order Tracking (domínio angular)",
            "6. Órbita e Análise Temporal Multi-Canal",
            "7. Calculadora de Frequências de Falha",
            "8. Diagnóstico — Espectro com Frequências de Falha",
            "9. Indicadores Estatísticos (Kurtose, Fator de Crista)",
            "A. Análise de Envelope (Demodulação AM — Rolamentos)",
            "B. Fast Kurtogram (Banda Ótima de Demodulação)",
            "C. Conceitos de Amostragem (Nyquist, 2,56, Janelas)",
            "D. Demonstração de ALIASING (sinal sintético interativo)",
            "0. Sair",
        ]

        for op in opcoes:
            print(f"  {op}")
        print()

        escolha = input("  Digite a opção [0-9 / A-D]: ").strip().upper()

        if escolha == '1':
            # ── Calibração: PRIMEIRO PASSO recomendado ──────────────────────
            menu_calibracao(sinais_acel_orig, estado, fs)
            sinais_acel = estado['sinais_acel']
            _UNIDADE['saida'] = estado.get('unidade_saida', 'V')

        elif escolha == '2':
            plot_series_temporais(sinais_acel, sinal_taco, fs,
                                  estado['rpm'] if estado['rpm'] else 0)

        elif escolha == '3':
            idx_p, rpm, diag = detectar_rpm(sinal_taco, fs)
            if rpm is not None:
                estado['indices_pulsos'] = idx_p
                estado['rpm']           = rpm
                estado['diag_taco']     = diag
                estado['fase_ref']      = criar_fase_referencia(n_amostras, idx_p, fs)
                estado['frequencias']['rpm'] = rpm
                estado['frequencias']['freq_rot'] = rpm / 60.0
                estado['frequencias']['falhas'] = {}

        elif escolha == '4':
            if estado['rpm'] is None:
                print("  ⚠ Execute primeiro a detecção de RPM (opção 3).")
                continue
            analise_espectral_interativa(sinais_acel, fs, estado['rpm'],
                                         estado['indices_pulsos'])

        elif escolha == '5':
            if estado['rpm'] is None:
                print("  ⚠ Execute primeiro a detecção de RPM (opção 3).")
                continue
            analise_ordem(sinais_acel, estado['fase_ref'],
                          estado['indices_pulsos'], estado['rpm'], fs)

        elif escolha == '6':
            if estado['rpm'] is None:
                print("  ⚠ Execute primeiro a detecção de RPM (opção 3).")
                continue
            analise_orbita(sinais_acel, estado['fase_ref'], estado['rpm'], fs)

        elif escolha == '7':
            if estado['rpm'] is None:
                print("  ⚠ Execute primeiro a detecção de RPM (opção 3).")
                continue
            freqs_falha = calcular_frequencias_falha(estado['rpm'])
            estado['frequencias'].update(freqs_falha)

        elif escolha == '8':
            if estado['rpm'] is None:
                print("  ⚠ Execute primeiro a detecção de RPM (opção 3).")
                continue
            if not estado['frequencias'].get('falhas'):
                print("  ⚠ Calcule frequências de falha primeiro (opção 7).")
            plot_espectro_com_falhas(sinais_acel, fs, estado['rpm'],
                                     estado['frequencias'])

        elif escolha == '9':
            if estado['rpm'] is None:
                rpm_stat = entrada_float("  RPM aproximado (para referência)", minval=1)
                estado['rpm'] = rpm_stat
            analise_estatistica(sinais_acel, fs, estado['rpm'])

        elif escolha == 'A':
            if estado['rpm'] is None:
                print("  ⚠ Execute primeiro a detecção de RPM (opção 3).")
                continue
            analise_envelope(sinais_acel, fs, estado['rpm'],
                             freq_rot=estado['rpm'] / 60.0)

        elif escolha == 'B':
            if estado['rpm'] is None:
                print("  ⚠ Execute primeiro a detecção de RPM (opção 3).")
                continue
            fl_opt, fh_opt = fast_kurtogram(sinais_acel, fs, estado['rpm'])
            if fl_opt is not None:
                analise_envelope(sinais_acel, fs, estado['rpm'],
                                 freq_rot=estado['rpm'] / 60.0)

        elif escolha == 'C':
            aula_amostragem(fs, n_amostras)

        elif escolha == 'D':
            demo_aliasing(fs)

        elif escolha == '0':
            print("\n  Encerrando o programa. Bons estudos!\n")
            break

        else:
            print("  Opção inválida.")


# ═════════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    menu_principal()