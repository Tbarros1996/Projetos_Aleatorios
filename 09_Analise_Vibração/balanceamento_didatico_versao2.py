"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     BALANCEAMENTO DE ROTORES (PLANO ÚNICO) — MÉTODO DA MASSA DE TESTE        ║
║     Versão Didática (Vibrações Mecânicas) — ISO (G) + Sensitividade + 2 Pás  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Inclui:
  ✅ Sensitividade acelerômetros CH2/CH3/CH4 (mV/g → V/g): converte V → g
  ✅ Processamento multi-canal e opção de fusão (média vetorial de Wc)
  ✅ Estratégias em duas pás:
       (A) mesmo raio r, massas diferentes
       (B) mesma massa m, raios diferentes
  ✅ Convenção angular de pás (LAB):
       - Pá 1 = 0°
       - Ângulo cresce no sentido HORÁRIO
       - Gráficos polares: 0° no topo, sentido horário (theta_direction = -1)
  ✅ Checagem ISO 1940 / 21940:
       - Entrada: massa do rotor (kg) e grau G (mm/s)
       - Calcula e_per (g·mm/kg) e compara com e = U/M

Requisitos:
  numpy, scipy, pandas, matplotlib
"""

import os
from datetime import datetime
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

# ==============================================================================
# DIRETÓRIO DE SAÍDA
# ==============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "resultados_balanceamento")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CONFIGURAÇÕES DE PLOT
# ==============================================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
})

CORES = {
    'primaria':   '#1f4e79',
    'secundaria': '#2e75b6',
    'destaque':   '#c00000',
    'alerta':     '#ed7d31',
    'ok':         '#548235',
    'neutro':     '#595959',
}

# ==============================================================================
# UTILITÁRIOS DE ENTRADA
# ==============================================================================
def _perguntar_float(mensagem: str, default: float,
                     minimo: float = None, maximo: float = None) -> float:
    while True:
        try:
            entrada = input(f"  {mensagem} [{default}]: ").strip()
            valor = float(entrada) if entrada else default
            if minimo is not None and valor < minimo:
                print(f"  → Valor mínimo: {minimo}")
                continue
            if maximo is not None and valor > maximo:
                print(f"  → Valor máximo: {maximo}")
                continue
            return valor
        except ValueError:
            print("  → Entre um número válido.")


def _perguntar_opcao(mensagem: str, opcoes: list, default: str) -> str:
    while True:
        entrada = input(f"  {mensagem} {opcoes} [{default}]: ").strip().lower()
        if not entrada:
            return default
        if entrada in [o.lower() for o in opcoes]:
            return entrada
        print(f"  → Opções válidas: {opcoes}")

# ==============================================================================
# ISO 1940 / 21940 - permissível específico (g·mm/kg)
# ==============================================================================
def iso_permissible_specific_unbalance_gmm_per_kg(G_mm_s: float, rpm: float) -> float:
    """
    ISO:
      G = e_per(mm) * omega(rad/s)
      e_per(g·mm/kg) = 1000 * e_per(mm)
    """
    omega = 2*np.pi * rpm / 60.0
    e_per_mm = G_mm_s / omega
    return 1000.0 * e_per_mm  # g·mm/kg

def iso_check(U_gmm: float, rpm: float, rotor_mass_kg: float, G_mm_s: float) -> dict:
    """
    Compara e = U/M com e_per (ISO) para o grau G.
    """
    e = U_gmm / rotor_mass_kg  # g·mm/kg
    e_per = iso_permissible_specific_unbalance_gmm_per_kg(G_mm_s, rpm)
    ok = e <= e_per
    M_min = U_gmm / e_per if e_per > 0 else float('inf')
    return {
        "U_gmm": U_gmm,
        "rpm": rpm,
        "rotor_mass_kg": rotor_mass_kg,
        "G_mm_s": G_mm_s,
        "e_gmm_per_kg": e,
        "e_per_gmm_per_kg": e_per,
        "ok": ok,
        "M_min_kg": M_min,
    }

# ==============================================================================
# CONVERSÃO DE UNIDADES (Sensitividade)
# ==============================================================================
def converter_V_para_g(sinal_V: np.ndarray, sens_V_g: float) -> np.ndarray:
    if sens_V_g <= 0:
        raise ValueError("sens_V_g deve ser > 0.")
    return sinal_V / sens_V_g

# ==============================================================================
# LEITURA DE ARQUIVOS (CSV ou XLSX)
# ==============================================================================
def ler_dados(caminho: str, verbose: bool = True):
    """
    Aceita:
      - CSV com colunas: Time, CH1, CH2, CH3, CH4 (Time pode existir ou não)
      - XLSX com colunas CH1..CH4 ou colunas nas primeiras posições.

    Retorna:
      taco (CH1), canais_acel (dict com CH2/CH3/CH4 se existirem)
    """
    if not os.path.exists(caminho):
        print(f"  [ERRO] Arquivo não encontrado: {caminho}")
        return None, None

    ext = os.path.splitext(caminho)[1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(caminho)
    else:
        df = pd.read_csv(caminho, index_col=False)

    df.columns = [str(c).strip() for c in df.columns]

    # Se não vier com CHx, tenta mapear por posição (col0=CH1 etc.)
    cols = df.columns.tolist()
    if 'CH1' not in cols:
        if len(cols) >= 2:
            ren = {cols[0]: 'CH1', cols[1]: 'CH2'}
            if len(cols) >= 3: ren[cols[2]] = 'CH3'
            if len(cols) >= 4: ren[cols[3]] = 'CH4'
            df = df.rename(columns=ren)

    def getcol(name):
        return pd.to_numeric(df[name], errors='coerce').to_numpy() if name in df.columns else None

    taco = getcol('CH1')
    ch2  = getcol('CH2')
    ch3  = getcol('CH3')
    ch4  = getcol('CH4')

    if taco is None or ch2 is None:
        print("  [ERRO] Necessário ao menos CH1 (taco) e CH2 (acel).")
        return None, None

    mask = ~np.isnan(taco) & ~np.isnan(ch2)
    if ch3 is not None:
        mask &= ~np.isnan(ch3)
    if ch4 is not None:
        mask &= ~np.isnan(ch4)

    taco = taco[mask]
    canais = {'CH2': ch2[mask]}
    if ch3 is not None: canais['CH3'] = ch3[mask]
    if ch4 is not None: canais['CH4'] = ch4[mask]

    if verbose:
        print(f"  Amostras válidas: {len(taco)}")
        print(f"  CH1 (Taco) — min: {taco.min():.4f}  max: {taco.max():.4f}")
        for k, v in canais.items():
            print(f"  {k} (Acel, V) — RMS: {np.sqrt(np.mean(v**2)):.6f}")

    return taco, canais

# ==============================================================================
# TACÔMETRO → RPM e pulsos
# ==============================================================================
def detectar_rpm_taquimetro(sinal_taco: np.ndarray, fs: float,
                            verbose: bool = True) -> tuple:
    if verbose:
        print("\n  ── Detecção de Pulsos do Tacômetro ──")

    sinal_norm = (sinal_taco - np.mean(sinal_taco)) / (np.std(sinal_taco) + 1e-12)
    dist_min = int(0.008 * fs)  # 8 ms

    resultados = []
    for tipo, sinal_proc in [('POSITIVO', sinal_norm), ('NEGATIVO', -sinal_norm)]:
        for thresh in [0.5, 1.0, 1.5, 2.0]:
            picos, _ = signal.find_peaks(
                sinal_proc, height=thresh, distance=dist_min, prominence=0.3
            )
            if len(picos) < 5:
                continue

            periodos = np.diff(picos) / fs
            rpm_calc = 60.0 / np.mean(periodos)
            cv = np.std(periodos) / np.mean(periodos) * 100

            if 200 <= rpm_calc <= 6000 and cv < 15:
                resultados.append({
                    'tipo': tipo, 'thresh': thresh, 'picos': picos,
                    'rpm': rpm_calc, 'cv': cv, 'periodos': periodos
                })

    if len(resultados) == 0:
        print("  [ERRO] Nenhum pulso periódico detectado!")
        return None, None, None

    melhor = min(resultados, key=lambda x: x['cv'])
    indices_pulsos = melhor['picos']
    rpm_final = melhor['rpm']
    periodos  = melhor['periodos']
    var_pct   = melhor['cv']

    if verbose:
        print(f"    Pulsos detectados  : {len(indices_pulsos)}")
        print(f"    RPM calculado      : {rpm_final:.2f}")
        print(f"    Período médio      : {np.mean(periodos)*1000:.2f} ms")
        print(f"    Variação (CV)      : ±{var_pct:.2f}%")

    diag = {'periodos': periodos, 'var_pct': var_pct, 'tipo': melhor['tipo']}
    return indices_pulsos, rpm_final, diag

# ==============================================================================
# FASE por amostra (0–360°)
# ==============================================================================
def construir_referencia_fase(indices_pulsos: np.ndarray, N: int, fs: float) -> np.ndarray:
    t = np.arange(N) / fs
    t_pulsos = indices_pulsos / fs
    angulos_pulsos = np.arange(len(indices_pulsos)) * 360.0
    interp = interp1d(t_pulsos, angulos_pulsos, kind='linear', fill_value='extrapolate')
    return interp(t) % 360.0

# ==============================================================================
# FILTRO
# ==============================================================================
def filtrar_sinal_acelerometro(sinal: np.ndarray, fs: float,
                               rpm: float, n_harmonicas: int = 5) -> np.ndarray:
    f_rot  = rpm / 60.0
    lowcut = max(0.5 * f_rot, 0.5)
    highcut = min(n_harmonicas * f_rot * 1.2, fs/2.0 - 1.0)
    sos = signal.butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, sinal)

# ==============================================================================
# LOCK-IN 1X
# ==============================================================================
def extrair_componente_1x(sinal_acel: np.ndarray,
                          fase_ref: np.ndarray,
                          janela: str = 'hanning',
                          verbose: bool = False) -> tuple:
    N = len(sinal_acel)

    if janela == 'hanning':
        w = np.hanning(N); fator_norm = np.mean(w)
    elif janela == 'hamming':
        w = np.hamming(N); fator_norm = np.mean(w)
    elif janela == 'flattop':
        w = signal.windows.flattop(N); fator_norm = np.mean(w)
    else:
        w = np.ones(N); fator_norm = 1.0

    s = sinal_acel * w
    fase_rad = np.deg2rad(fase_ref)

    X = np.mean(s * np.cos(fase_rad)) * 2 / fator_norm
    Y = np.mean(s * np.sin(fase_rad)) * 2 / fator_norm

    amplitude = np.sqrt(X**2 + Y**2)
    fase      = np.rad2deg(np.arctan2(Y, X)) % 360.0

    if verbose:
        print(f"\n  ── 1X (Lock-in) | Janela: {janela} ──")
        print(f"    X (em fase)       : {X:.6f}")
        print(f"    Y (quadratura)    : {Y:.6f}")
        print(f"    Amplitude 1X (g)  : {amplitude:.6f}")
        print(f"    Fase 1X (graus)   : {fase:.1f}°")

    return amplitude, fase, X, Y

# ==============================================================================
# BALANCEAMENTO (coeficiente de influência)
# ==============================================================================
def calcular_correcao_balanceamento(A0: float, phi0: float,
                                   At: float, phit: float,
                                   Wt: float, theta_t: float,
                                   raio_teste: float, raio_correcao: float,
                                   verbose: bool = True) -> dict:
    if verbose:
        print("\n" + "═"*60)
        print("  MÉTODO DOS COEFICIENTES DE INFLUÊNCIA (PLANO ÚNICO)")
        print("═"*60)
        print(f"\n  U0 : {A0:.6f} g ∠ {phi0:.1f}°")
        print(f"  Ut : {At:.6f} g ∠ {phit:.1f}°")
        print(f"  Wt : {Wt:.2f} g ∠ {theta_t:.1f}°")
        print(f"  Raio teste    : {raio_teste*1000:.1f} mm")
        print(f"  Raio correção : {raio_correcao*1000:.1f} mm")

    fator_raio = raio_teste / raio_correcao
    Wt_efetiva = Wt * fator_raio

    U0 = A0 * np.exp(1j * np.deg2rad(phi0))
    Ut = At * np.exp(1j * np.deg2rad(phit))
    Wt_vec = Wt_efetiva * np.exp(1j * np.deg2rad(theta_t))

    delta_U = Ut - U0
    alpha   = delta_U / Wt_vec
    Wc      = -U0 / alpha

    massa_corr  = np.abs(Wc)
    angulo_corr = np.angle(Wc, deg=True) % 360.0

    Uf = U0 + alpha * Wc
    amp_residual = np.abs(Uf)
    reducao_pct  = (1.0 - amp_residual / A0) * 100.0 if A0 > 0 else 0.0

    if verbose:
        print(f"\n  |α|={np.abs(alpha):.6f}  ∠α={np.angle(alpha, deg=True)%360:.1f}°")
        print(f"  Wc = {massa_corr:.2f} g ∠ {angulo_corr:.1f}°")
        print(f"  Redução prevista ≈ {reducao_pct:.1f}%")

    return {
        'massa_correcao': massa_corr,
        'angulo_correcao': angulo_corr,
        'coef_influencia': alpha,
        'reducao_pct': reducao_pct,
        'amp_residual': amp_residual,
        'U0': U0, 'Ut': Ut, 'Wt_vec': Wt_vec, 'Wc': Wc,
        'fator_raio': fator_raio,
    }

# ==============================================================================
# FUSÃO DE SENSORES (Wc)
# ==============================================================================
def fundir_Wc_por_canais(correcoes_por_canal: dict, pesos_por_canal: dict = None) -> complex:
    if pesos_por_canal is None:
        pesos_por_canal = {k: 1.0 for k in correcoes_por_canal.keys()}
    num = 0+0j
    den = 0.0
    for ch, wc in correcoes_por_canal.items():
        w = float(pesos_por_canal.get(ch, 1.0))
        num += w * wc
        den += w
    return num/den if den > 0 else list(correcoes_por_canal.values())[0]

# ==============================================================================
# ESTRATÉGIAS 2 PÁS
# ==============================================================================
def estrategia_duas_pas_mesmo_raio_massas_diferentes(
        angulo_ideal: float,
        massa_ideal: float,
        n_pas: int = 3,
        massas_disponiveis: list = None,
        usar_pares_adjacentes: bool = True,
        top_k: int = 5) -> list:
    if massas_disponiveis is None:
        massas_disponiveis = [0.5, 1, 2, 3, 5, 7.55, 10]

    angulos = np.linspace(0, 360, n_pas, endpoint=False)
    ideal = massa_ideal * np.exp(1j*np.deg2rad(angulo_ideal))
    norm = np.abs(ideal) if np.abs(ideal) > 1e-12 else 1.0

    solucoes = []
    pares = []
    if usar_pares_adjacentes:
        for i in range(n_pas):
            pares.append((i, (i+1) % n_pas))
    else:
        for i in range(n_pas):
            for j in range(i+1, n_pas):
                pares.append((i, j))

    for i, j in pares:
        th1, th2 = angulos[i], angulos[j]
        for m1 in massas_disponiveis:
            for m2 in massas_disponiveis:
                v = (m1*np.exp(1j*np.deg2rad(th1)) +
                     m2*np.exp(1j*np.deg2rad(th2)))
                erro = np.abs(ideal - v) / norm * 100
                solucoes.append({
                    'tipo': '2pas_mesmo_raio_massas_dif',
                    'pas': (i+1, j+1),
                    'angulos': (th1, th2),
                    'massas_g': (m1, m2),
                    'erro_pct': erro,
                })

    solucoes.sort(key=lambda x: x['erro_pct'])
    return solucoes[:top_k]


def estrategia_duas_pas_mesma_massa_raios_diferentes(
        angulo_ideal: float,
        Uc_equiv_gmm: float,
        massa_fix_g: float,
        n_pas: int = 3,
        raio_min_mm: float = 10.0,
        raio_max_mm: float = 200.0,
        usar_pares_adjacentes: bool = True,
        top_k: int = 5) -> list:
    angulos = np.linspace(0, 360, n_pas, endpoint=False)
    target = Uc_equiv_gmm * np.exp(1j*np.deg2rad(angulo_ideal))

    pares = []
    if usar_pares_adjacentes:
        for i in range(n_pas):
            pares.append((i, (i+1) % n_pas))
    else:
        for i in range(n_pas):
            for j in range(i+1, n_pas):
                pares.append((i, j))

    solucoes = []
    for i, j in pares:
        th1 = angulos[i]
        th2 = angulos[j]

        rhs = target / massa_fix_g  # (g·mm)/(g) = mm

        c1, s1 = np.cos(np.deg2rad(th1)), np.sin(np.deg2rad(th1))
        c2, s2 = np.cos(np.deg2rad(th2)), np.sin(np.deg2rad(th2))
        A = np.array([[c1, c2],
                      [s1, s2]], dtype=float)
        b = np.array([rhs.real, rhs.imag], dtype=float)

        det = np.linalg.det(A)
        if abs(det) < 1e-9:
            continue

        r1, r2 = np.linalg.solve(A, b)

        ok = (r1 >= 0) and (r2 >= 0) and (raio_min_mm <= r1 <= raio_max_mm) and (raio_min_mm <= r2 <= raio_max_mm)

        v = massa_fix_g*(r1*np.exp(1j*np.deg2rad(th1)) + r2*np.exp(1j*np.deg2rad(th2)))
        norm = np.abs(target) if np.abs(target) > 1e-9 else 1.0
        erro_pct = np.abs(target - v)/norm*100

        solucoes.append({
            'tipo': '2pas_mesma_massa_raios_dif',
            'pas': (i+1, j+1),
            'angulos': (th1, th2),
            'massa_fix_g': massa_fix_g,
            'raios_mm': (r1, r2),
            'ok': ok,
            'erro_pct': erro_pct,
        })

    solucoes.sort(key=lambda x: (0 if x['ok'] else 1, x['erro_pct']))
    return solucoes[:top_k]

# ==============================================================================
# PLOTS
# ==============================================================================
def plotar_analise_etapa_multicanal(dados: dict, titulo_etapa: str,
                                   canal_plot: str = 'CH2',
                                   salvar: bool = True) -> None:
    taco     = dados['taco']
    fs       = dados['fs']
    rpm      = dados['rpm']
    fase_ref = dados['fase_ref']
    pulsos   = dados['pulsos']
    t        = dados['t']
    janela   = dados['janela']

    canal = dados['canais'][canal_plot]
    sig_g    = canal['sig_g']
    sig_f    = canal['sig_f']
    amp_1x   = canal['amp_1x']
    fase_1x  = canal['fase_1x']

    N = len(taco)
    f_rot = rpm/60.0

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"{titulo_etapa} | Canal: {canal_plot} | RPM={rpm:.1f} | 1X: {amp_1x:.4f} g ∠ {fase_1x:.1f}°",
        fontsize=13, fontweight='bold', color=CORES['primaria']
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    # [1] Tacômetro
    ax1 = fig.add_subplot(gs[0, :2])
    n_zoom = min(N, int(5/f_rot*fs)+1)
    ax1.plot(t[:n_zoom], taco[:n_zoom], color=CORES['neutro'], lw=0.8)
    for idx in pulsos:
        if idx < n_zoom:
            ax1.axvline(t[idx], color=CORES['destaque'], lw=1.0, alpha=0.7)
    ax1.set_title('Tacômetro — primeiras 5 revoluções')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('V (taco)')

    # [2] Acelerômetro bruto vs filtrado (g)
    ax2 = fig.add_subplot(gs[1, :2])
    n_show = min(N, int(10/f_rot*fs)+1)
    ax2.plot(t[:n_show], sig_g[:n_show], color='lightgray', lw=0.7, label='Bruto (g)')
    ax2.plot(t[:n_show], sig_f[:n_show], color=CORES['secundaria'], lw=1.2, label='Filtrado (g)')
    ax2.set_title(f'{canal_plot} — 10 revoluções (em g)')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Aceleração (g)')
    ax2.legend(fontsize=8)

    # [3] Espectro
    ax3 = fig.add_subplot(gs[2, :2])
    w = np.hanning(N) if janela == 'hanning' else np.ones(N)
    fator = np.mean(w)
    espectro = np.abs(fft(sig_f*w))[:N//2] * 2/(N*fator)
    freqs = fftfreq(N, 1/fs)[:N//2]
    ax3.semilogy(freqs, espectro + 1e-9, color=CORES['secundaria'], lw=0.7)
    for n in range(1, 6):
        ax3.axvline(n*f_rot, color=CORES['destaque'], ls='--', lw=1.0, alpha=0.8)
    ax3.set_xlim([0, f_rot*10])
    ax3.set_title('Espectro (g)')
    ax3.set_xlabel('Frequência (Hz)')
    ax3.set_ylabel('Amplitude (g)')

    # [4] Estabilidade RPM
    ax4 = fig.add_subplot(gs[0, 2])
    periodos_ms = dados['periodos']*1000
    ax4.plot(periodos_ms, 'o-', color=CORES['secundaria'], ms=4)
    ax4.axhline(np.mean(periodos_ms), color=CORES['destaque'], ls='--', lw=1.0)
    ax4.set_title('Estabilidade da rotação')
    ax4.set_xlabel('Intervalo #')
    ax4.set_ylabel('Período (ms)')

    # [5] Polar 1X (0° topo, sentido horário)
    ax5 = fig.add_subplot(gs[1, 2], projection='polar')
    ax5.set_theta_zero_location('N')
    ax5.set_theta_direction(-1)
    theta = np.deg2rad(fase_1x)
    ax5.annotate('', xy=(theta, amp_1x), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color=CORES['destaque'], lw=2.5))
    ax5.set_title('1X (canal plotado)', pad=15)

    # [6] Resumo multi-canal
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    linhas = [f"Resumo 1X (sens={dados['sens_V_g']*1000:.0f} mV/g)\n" + "─"*30]
    for ch, info in dados['canais'].items():
        linhas.append(f"{ch}: {info['amp_1x']:.5f} g ∠ {info['fase_1x']:.1f}°")
    linhas.append("─"*30)
    linhas.append(f"Canal principal: {dados['canal_principal']}")
    linhas.append(f"Fusão sensores : {dados['fusao_sensores']}")
    ax6.text(0.05, 0.97, "\n".join(linhas), transform=ax6.transAxes,
             fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4ff',
                       edgecolor=CORES['secundaria'], linewidth=1.5))

    if salvar:
        ts = datetime.now().strftime("%H%M%S")
        nome_arquivo = "".join(c if c.isalnum() or c in ('_', '-') else '_'
                               for c in titulo_etapa)
        fname = os.path.join(OUTPUT_DIR, f'analise_{nome_arquivo}_{canal_plot}_{ts}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  [Figura salva: {os.path.basename(fname)}]")

    plt.show()


def plotar_resultado_vetorial(U0: complex, Ut: complex, Wt: complex, Wc: complex,
                              titulo: str, salvar: bool = True) -> None:
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(titulo, fontsize=14, fontweight='bold', color=CORES['primaria'])

    gs = gridspec.GridSpec(1, 2, wspace=0.25, left=0.06, right=0.97, top=0.88, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    for ax in (ax1, ax2):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    # Vibração
    ax1.set_title("Vibração 1X (g)")
    for label, v, lw in [('U0', U0, 3), ('Ut', Ut, 2.5)]:
        ang = np.angle(v)
        mag = np.abs(v)
        ax1.annotate('', xy=(ang, mag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', lw=lw))
        ax1.plot([ang], [mag], 'o')
        ax1.text(ang, mag*1.08, f"{label}\n{mag:.3f} g\n{(np.angle(v,deg=True)%360):.1f}°",
                 ha='center', fontsize=9)
    ax1.set_rlim(0, max(np.abs(U0), np.abs(Ut))*1.35)

    # Massas
    ax2.set_title("Massas (g) — plano de correção")
    for label, v, lw in [('Wt', Wt, 2.0), ('Wc', Wc, 3.0)]:
        ang = np.angle(v)
        mag = np.abs(v)
        ax2.annotate('', xy=(ang, mag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', lw=lw))
        ax2.plot([ang], [mag], 'o')
        ax2.text(ang, mag*1.08, f"{label}\n{mag:.2f} g\n{(np.angle(v,deg=True)%360):.1f}°",
                 ha='center', fontsize=9)
    ax2.set_rlim(0, max(np.abs(Wt), np.abs(Wc))*1.35)

    if salvar:
        ts = datetime.now().strftime("%H%M%S")
        fname = os.path.join(OUTPUT_DIR, f'vetores_{ts}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  [Figura salva: {os.path.basename(fname)}]")

    plt.show()

# ==============================================================================
# CONFIGURAÇÃO (interativo) — inclui MASSA DO ROTOR e G (ISO)
# ==============================================================================
def configurar_parametros() -> dict:
    print("\n" + "╔" + "═"*70 + "╗")
    print("║" + "  CONFIGURAÇÃO DOS PARÂMETROS (inclui ISO)".center(70) + "║")
    print("╚" + "═"*70 + "╝")

    rpm_aprox = _perguntar_float("RPM aproximado", 1000.0, 100.0, 6000.0)
    f_rot = rpm_aprox / 60.0

    fs_nyquist_min = 10 * f_rot
    fs_rec = round(50 * f_rot / 100) * 100
    fs = _perguntar_float("fs (Hz)", float(fs_rec), fs_nyquist_min)

    T_min = 5 / f_rot
    T_rec = max(round(20 / f_rot, 1), 2.0)
    duracao = _perguntar_float("Duração (s)", T_rec, T_min)

    janela = _perguntar_opcao("Janela", ['retangular','hanning','hamming','flattop'], 'hanning')

    sens_mV_g = _perguntar_float("Sensitividade do acelerômetro (mV/g)", 100.0, 1.0)
    sens_V_g = sens_mV_g/1000.0

    n_pas = int(_perguntar_float("Número de pás", 3.0, 2.0, 12.0))
    angulos_pas = np.linspace(0, 360, n_pas, endpoint=False)
    print(f"  Ângulos das pás (Pá 1=0°, cresce HORÁRIO): {[f'{a:.0f}°' for a in angulos_pas]}")

    pa_teste = int(_perguntar_float(f"Pá para massa de teste (1–{n_pas})", 1.0, 1.0, float(n_pas)))
    angulo_teste = angulos_pas[pa_teste - 1]
    massa_teste = _perguntar_float("Massa de teste (g)", 7.55, 0.1)

    raio_mm = _perguntar_float("Raio de teste/correção (mm)", 100.0, 10.0)
    raio = raio_mm / 1000.0

    canal_principal = _perguntar_opcao("Canal principal", ["ch2","ch3","ch4"], "ch2").upper()
    fusao = _perguntar_opcao("Fusão de sensores", ["nenhum","media_wc"], "nenhum")

    # ISO
    print("\n  PARÂMETROS ISO (ISO 1940/21940)")
    rotor_mass_kg = _perguntar_float("Massa do rotor (kg)", 2.0, 0.01)
    G_mm_s = _perguntar_float("Grau ISO G (mm/s) (ex.: 6.3 para fans)", 6.3, 0.01)

    return {
        'rpm_aprox': rpm_aprox,
        'fs': fs,
        'duracao': duracao,
        'janela': janela,
        'sens_V_g': sens_V_g,
        'n_pas': n_pas,
        'angulos_pas': angulos_pas,
        'pa_teste': pa_teste,
        'angulo_teste': angulo_teste,
        'massa_teste': massa_teste,
        'raio_teste': raio,
        'raio_corr': raio,
        'canal_principal': canal_principal,
        'fusao_sensores': fusao,
        'rotor_mass_kg': rotor_mass_kg,
        'G_mm_s': G_mm_s,
    }

# ==============================================================================
# PROCESSAMENTO DE MEDIÇÃO (multi-canal)
# ==============================================================================
def processar_medicao(taco: np.ndarray, canais_acel_V: dict,
                      params: dict, titulo: str = "Etapa") -> dict:
    fs     = params['fs']
    janela = params['janela']
    sens_V_g = params['sens_V_g']

    print(f"\n  {'─'*60}")
    print(f"  Processando: {titulo}")
    print(f"  {'─'*60}")

    pulsos, rpm, diag_taco = detectar_rpm_taquimetro(taco, fs)
    if pulsos is None:
        return None

    N = len(taco)
    fase_ref = construir_referencia_fase(pulsos, N, fs)

    canais = {}
    for ch, sig_V in canais_acel_V.items():
        sig_g = converter_V_para_g(sig_V, sens_V_g)
        sig_f = filtrar_sinal_acelerometro(sig_g, fs, rpm)
        amp_1x, fase_1x, X, Y = extrair_componente_1x(sig_f, fase_ref, janela, verbose=False)

        canais[ch] = {
            'sig_V': sig_V,
            'sig_g': sig_g,
            'sig_f': sig_f,
            'amp_1x': amp_1x,
            'fase_1x': fase_1x,
            'X': X,
            'Y': Y,
            'U': amp_1x*np.exp(1j*np.deg2rad(fase_1x)),
        }

    resultado = {
        'taco': taco,
        'canais': canais,
        'fs': fs,
        'rpm': rpm,
        'pulsos': pulsos,
        'fase_ref': fase_ref,
        't': np.arange(N)/fs,
        'periodos': diag_taco['periodos'],
        'var_pct': diag_taco['var_pct'],
        'janela': janela,
        'sens_V_g': sens_V_g,
        'canal_principal': params['canal_principal'],
        'fusao_sensores': params['fusao_sensores'],
    }

    plotar_analise_etapa_multicanal(resultado, titulo_etapa=titulo, canal_plot=params['canal_principal'])
    return resultado

# ==============================================================================
# PROCEDIMENTO PRINCIPAL
# ==============================================================================
def executar_procedimento_balanceamento() -> None:
    print("\n" + "╔" + "═"*72 + "╗")
    print("║" + "  BALANCEAMENTO — MASSA DE TESTE (PLANO ÚNICO) + ISO".center(72) + "║")
    print("╚" + "═"*72 + "╝")

    params = configurar_parametros()

    fs = params['fs']
    T  = params['duracao']
    print(f"\n  Configuração: fs={fs:.0f} Hz | duração={T:.2f} s | sens={params['sens_V_g']*1000:.0f} mV/g")
    print(f"  Pás: {params['n_pas']} | Wt={params['massa_teste']:.2f} g em {params['angulo_teste']:.0f}° | raio={params['raio_teste']*1000:.0f} mm")
    print(f"  ISO: massa rotor={params['rotor_mass_kg']:.3f} kg | G={params['G_mm_s']:.2f} mm/s")

    # ============================
    # ETAPA 1
    # ============================
    print("\n" + "═"*68)
    print("  ETAPA 1: CONDIÇÃO INICIAL (sem massa de teste)")
    print("═"*68)
    arq1 = input("  Caminho do arquivo (CSV/XLSX) da condição inicial: ").strip()
    taco_1, canais_1 = ler_dados(arq1, verbose=True)
    if taco_1 is None:
        return

    res_1 = processar_medicao(taco_1, canais_1, params, titulo="Etapa 1 — Condição Inicial")
    if res_1 is None:
        return

    # ============================
    # ETAPA 2
    # ============================
    print("\n" + "═"*68)
    print("  ETAPA 2: COM MASSA DE TESTE")
    print("═"*68)
    print(f"  Fixar {params['massa_teste']:.2f} g na Pá #{params['pa_teste']} (ângulo {params['angulo_teste']:.0f}°) no raio {params['raio_teste']*1000:.0f} mm.")
    arq2 = input("  Caminho do arquivo (CSV/XLSX) com a massa de teste: ").strip()
    taco_2, canais_2 = ler_dados(arq2, verbose=True)
    if taco_2 is None:
        return

    res_2 = processar_medicao(taco_2, canais_2, params, titulo="Etapa 2 — Massa de Teste")
    if res_2 is None:
        return

    # ============================
    # ETAPA 3: Correção
    # ============================
    print("\n" + "═"*68)
    print("  ETAPA 3: CÁLCULO DA CORREÇÃO")
    print("═"*68)

    canal_principal = params['canal_principal']
    fusao = params['fusao_sensores']

    correcoes = {}
    for ch in res_1['canais'].keys():
        if ch not in res_2['canais']:
            continue
        A0  = res_1['canais'][ch]['amp_1x']
        p0  = res_1['canais'][ch]['fase_1x']
        At  = res_2['canais'][ch]['amp_1x']
        pt  = res_2['canais'][ch]['fase_1x']

        corr = calcular_correcao_balanceamento(
            A0=A0, phi0=p0,
            At=At, phit=pt,
            Wt=params['massa_teste'], theta_t=params['angulo_teste'],
            raio_teste=params['raio_teste'], raio_correcao=params['raio_corr'],
            verbose=(ch == canal_principal)
        )
        correcoes[ch] = corr

    if fusao == 'media_wc':
        pesos = {ch: res_1['canais'][ch]['amp_1x'] for ch in correcoes.keys()}
        Wc = fundir_Wc_por_canais({ch: correcoes[ch]['Wc'] for ch in correcoes.keys()}, pesos)
        massa_corr = np.abs(Wc)
        ang_corr = np.angle(Wc, deg=True) % 360.0
        print("\n  ── FUSÃO ATIVADA (média vetorial de Wc ponderada) ──")
        print(f"  Wc_fundido = {massa_corr:.2f} g ∠ {ang_corr:.1f}°")
    else:
        Wc = correcoes[canal_principal]['Wc']
        massa_corr = correcoes[canal_principal]['massa_correcao']
        ang_corr = correcoes[canal_principal]['angulo_correcao']

    c0 = correcoes[canal_principal]
    plotar_resultado_vetorial(c0['U0'], c0['Ut'], c0['Wt_vec'], Wc,
                              titulo=f"Resultado Vetorial — {canal_principal} (0° topo, sentido horário)")

    # ============================
    # ISO CHECK (com U ≈ Wc*R)
    # ============================
    Rmm = params['raio_corr']*1000.0
    U_gmm = massa_corr * Rmm

    iso = iso_check(
        U_gmm=U_gmm,
        rpm=res_1['rpm'],
        rotor_mass_kg=params['rotor_mass_kg'],
        G_mm_s=params['G_mm_s']
    )

    print("\n" + "═"*68)
    print("  ISO CHECK (estimativa usando U ≈ Wc·R)")
    print("═"*68)
    print(f"  RPM referência (Etapa 1): {iso['rpm']:.1f}")
    print(f"  U (g·mm): {iso['U_gmm']:.1f}")
    print(f"  e = U/M (g·mm/kg): {iso['e_gmm_per_kg']:.1f}")
    print(f"  e_per (G={iso['G_mm_s']:.2f}) (g·mm/kg): {iso['e_per_gmm_per_kg']:.1f}")
    print("  STATUS:", "OK ✅" if iso['ok'] else "FORA ❌")
    print(f"  Massa mínima p/ atender esse G: {iso['M_min_kg']:.2f} kg")

    # ============================
    # Estratégias em duas pás
    # ============================
    print("\n" + "═"*68)
    print("  ESTRATÉGIAS EM DUAS PÁS")
    print("═"*68)

    print("\n  [A] Duas pás, mesmo raio, massas diferentes")
    solucoes_A = estrategia_duas_pas_mesmo_raio_massas_diferentes(
        angulo_ideal=ang_corr,
        massa_ideal=massa_corr,
        n_pas=params['n_pas'],
        massas_disponiveis=[0.5, 1, 2, 3, 5, 7.55, 10],
        usar_pares_adjacentes=True,
        top_k=5
    )
    for i, sol in enumerate(solucoes_A, 1):
        p1, p2 = sol['pas']
        th1, th2 = sol['angulos']
        m1, m2 = sol['massas_g']
        print(f"   {i}) Pá {p1} ({th1:.0f}°) com {m1:.2f} g + Pá {p2} ({th2:.0f}°) com {m2:.2f} g"
              f"  → erro {sol['erro_pct']:.1f}%")

    print("\n  [B] Duas pás, mesma massa, raios diferentes (ex.: duas moedas)")
    massa_fix = params['massa_teste']
    solucoes_B = estrategia_duas_pas_mesma_massa_raios_diferentes(
        angulo_ideal=ang_corr,
        Uc_equiv_gmm=U_gmm,
        massa_fix_g=massa_fix,
        n_pas=params['n_pas'],
        raio_min_mm=10.0,
        raio_max_mm=Rmm,
        usar_pares_adjacentes=True,
        top_k=5
    )
    for i, sol in enumerate(solucoes_B, 1):
        p1, p2 = sol['pas']
        th1, th2 = sol['angulos']
        r1, r2 = sol['raios_mm']
        ok = "✓" if sol['ok'] else "⚠"
        print(f"   {i}) Pá {p1} ({th1:.0f}°) com {massa_fix:.2f} g no raio {r1:.1f} mm"
              f" + Pá {p2} ({th2:.0f}°) com {massa_fix:.2f} g no raio {r2:.1f} mm"
              f"  → erro {sol['erro_pct']:.1f}%  {ok}")

    print("\n" + "╔" + "═"*72 + "╗")
    print("║" + "  PROCEDIMENTO CONCLUÍDO".center(72) + "║")
    print("╚" + "═"*72 + "╝\n")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    executar_procedimento_balanceamento()