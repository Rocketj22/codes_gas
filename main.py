import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
from scipy.odr import ODR, Model, RealData

# === 0. PARAMETRI ===
sigma_P = 0.03       # incertezza sulla pressione [unità]
sigma_V = 0.03        # incertezza sul volume [unità]
sigma_T = 0.3        # incertezza sulla temperatura [K]
epsilon = 0       # per escludere le prime letture
V_min, V_max = 10.8, 19.0  # intervallo di volume da analizzare [unità]
transformation = 1   # 0 compressione 1 espansione
n_MonteCarlo = 2   # numero di iterazioni del metodo MonteCarlo
barre_errore = 0   # 0 negativo, 1 positivo
data_dir = r"C:\Users\loren\Desktop\ok"
file_pattern = os.path.join(data_dir, "*.txt")


# === Funzioni di pulizia ===
def remove_initial_readings(df, epsilon):
    min_invP = df['invP'].min()
    return df[df['invP'] > min_invP + epsilon].copy()


def clean_by_zscore(df, cols, z_thresh=1.8, n_iter=2):
    for _ in range(n_iter):
        for c in cols:
            m, s = df[c].mean(), df[c].std(ddof=0)
            df = df[np.abs((df[c] - m) / s) <= z_thresh]
    return df


# === Modello e funzioni di fit ===
def linear_model(beta, x):
    """Modello lineare y = beta[0]*x + beta[1]"""
    return beta[0] * x + beta[1]


def fit_with_uncertainty_odr(x, y, sx, sy, beta0=(1.0,0.0)):
    model   = odr.Model(linear_model)
    data    = odr.RealData(x, y, sx=sx, sy=sy)
    odr_obj = odr.ODR(data, model, beta0=beta0)
    out     = odr_obj.run()

    slope, intercept     = out.beta
    dslope, dintercept   = out.sd_beta
    chi2                  = out.sum_square
    chi2_red              = out.res_var
    print(f"slope: {slope:.3e} \t {dslope:.0e} \t {intercept:.3e} \t {dintercept:.0e}")

    return slope, intercept, dslope, dintercept, chi2, chi2_red



# Funzione Monte Carlo per correlazione
def mc_correlated_fit(x, y, cov_mats, beta0, model_fun, n_iter):
    a_samps = np.empty(n_iter)
    b_samps = np.empty(n_iter)

    for k in range(n_iter):
        # genera un dataset sintetico con covarianza
        xy_k = np.array([
            np.random.multivariate_normal([x[i], y[i]], cov_mats[i])
            for i in range(len(x))
        ])
        xk, yk = xy_k[:, 0], xy_k[:, 1]

        # fit ODR sul dataset sintetico
        data_k = RealData(xk, yk)
        odr_k = ODR(data_k, Model(model_fun), beta0=beta0)
        out_k = odr_k.run()

        a_samps[k], b_samps[k] = out_k.beta

    return a_samps, b_samps


def fmt_with_error(val, err, sig=3):
    """
    Ritorna una stringa del tipo:
      2.710±0.020e11
    invece di:
      2.710e11±0.020e11
    """
    if val == 0:
        return f"(0±{err:.{sig}g})"
    exp = int(np.floor(np.log10(abs(val))))
    scale = 10 ** exp
    m = val / scale
    dm = err / scale
    return f"({m:.{sig}f}±{dm:.{sig}f})e{exp}"

# === 1. Raccolta, pulizia e fit dei dati ===
results = []
sigma_P *= 10 ** 4 * 9.80665
sigma_V *= 10 ** -6
sigma_T *= 1
V_min *= 10 ** -6
V_max *= 10 ** -6
for fp in glob.glob(file_pattern):
    df = pd.read_csv(fp, sep='\t', header=None, names=['P', 'V', 'T'])
    df[['P','V','T']] = df[['P','V','T']].apply(pd.to_numeric, errors='coerce')
    df['P'] *= 10 ** 5
    df['V'] *= 10 ** -6
    df['T'] += 273.15
    df['invP'] = 1 / df['P']
    df = remove_initial_readings(df, epsilon)

    # scegli se fare compressione, espansione o entrambi
    # calcolo la differenza istantanea di V
    df['dV'] = df['V'].diff()

    if transformation == 0:
        # compressione: V sta diminuendo --> dV<0
        df_proc = df[df['dV'] < 0].copy()
    elif transformation == 1:
        # espansione: V sta aumentando --> dV>0
        df_proc = df[df['dV'] > 0].copy()
    else:
        # tutto il dataset (dopo il primo scarto)
        df_proc = df.copy()

    # rimuovo eventuali NaN creati dal diff() e continuo con la pulizia
    df_proc = df_proc.dropna(subset=['dV'])
    df_proc = clean_by_zscore(df_proc, ['V', 'invP'])
    df_proc = df_proc[(df_proc['V'] >= V_min) & (df_proc['V'] <= V_max)]

    # ora df_proc è pronto per il fit
    T_mean = df_proc['T'].mean()
    T_std = df_proc['T'].std(ddof=1)
    x = df_proc['invP'].values
    y = df_proc['V'].values
    sx = sigma_P / df_proc['P'].values ** 2
    sy = np.full_like(y, sigma_V)

    # === Inserisci qui il calcolo delle matrici di covarianza ===
    cov_mats = []
    rho = 0.9999633981249841  # sostituire con il valore corretto
    for sx_i, sy_i in zip(sx, sy):
        cov = np.array([[sx_i**2,         rho * sx_i * sy_i],
                        [rho * sx_i * sy_i, sy_i**2       ]])
        cov_mats.append(cov)

    # Stime iniziali per ODR
    slope_guess     = (y[-1] - y[0]) / (x[-1] - x[0])
    intercept_guess = y.mean() - slope_guess * x.mean()

    # Fit ODR classico
    s, i, ds, di, chi2, chi2_red = fit_with_uncertainty_odr(x, y, sx, sy,
                                                            beta0=(slope_guess, intercept_guess))
    idx_min = (df_proc['V'] - V_min).abs().idxmin()
    idx_max = (df_proc['V'] - V_max).abs().idxmin()
    P_Vmin_data = df_proc.loc[idx_min, 'P'] / (10 ** 4 * 9.80665)
    P_Vmax_data = df_proc.loc[idx_max, 'P'] / (10 ** 4 * 9.80665)

    sy_post = sigma_V * np.sqrt(chi2_red)
    #print(f"χ² = {chi2:.2f}, χ²ν = {chi2_red:.8f}, err_y = {sy_post:.2e}, y ={y[-1]:.2e}, P_max = {P_Vmax_data:.2e}, "
          #f"P_min = {P_Vmin_data:.2e}, tempo = {len(x)}")

    # Fit Monte Carlo con correlazione
    a_samps, b_samps = mc_correlated_fit(
        x, y, cov_mats,
        beta0=(slope_guess, intercept_guess),
        model_fun=linear_model,
        n_iter=n_MonteCarlo
    )
    a_mean, a_err = a_samps.mean(), a_samps.std(ddof=1)
    b_mean, b_err = b_samps.mean(), b_samps.std(ddof=1)

    mol = a_mean/ (8.314 * T_mean)
    delta_y = y.max() - y.min()
    delta_x = x.max() - x.min()

    results.append({
        'file':      os.path.basename(fp),
        'mol':       mol,
        'T_mean':    T_mean,
        'T_std':     T_std,
        'slope':     a_mean,
        'intercept': b_mean,
        'dslope':    a_err,
        'dint':      b_err,
        'err_casuale':          a_err/a_mean,
        'err_taraturaV':         2 * sigma_V / (np.sqrt(3) * delta_y),
        'err_taraturaP':         2 * sigma_P / (np.sqrt(3) / delta_x),
        'err_non_isoterma':     T_std / T_mean
    })

# === 1.b Stampa dei risultati su console ===
for res in results:
    a = fmt_with_error(res['slope'], res['dslope'], 3)
    b = fmt_with_error(res['intercept'], res['dint'], 3)
    print(f"{res['file']}: a = {a}, b = {b}\n"
          f"\t\terr_casuale = {res['err_casuale']:.2e}, err_taratura V = {res['err_taraturaV']:.2e},"
          f" err_taratura P = {res['err_taraturaP']:.2e}, non_isoterma = {res['err_non_isoterma']:.2e}, \n"
          f"\t\tmoli = {res['mol']:.4e}, average_temp: {res['T_mean']:.2f} ± {res['T_std']:.2f}\n")

# === 2. Grafico dei risultati ===
plt.figure(figsize=(10,6))
for res in results:
    base = os.path.splitext(res['file'])[0]
    temp_label = f"{base}°C"
    df_plot = pd.read_csv(os.path.join(data_dir, res['file']), sep='\t', header=None, names=['P','V','T'])
    df_plot[['P', 'V', 'T']] = df_plot[['P', 'V', 'T']].apply(pd.to_numeric, errors='coerce')
    df_plot['P'] *= 10 ** 4 * 9.80665
    df_plot['V'] *= 10 ** -6
    df_plot['T'] += 273.15
    df_plot['invP'] = 1 / df_plot['P']
    df_plot = remove_initial_readings(df_plot, epsilon)

    df_plot['dV'] = df_plot['V'].diff()

    if transformation == 0:
        df_plot = df_plot[df_plot['dV'] < 0]
    elif transformation == 1:
        df_plot = df_plot[df_plot['dV'] > 0]
    # else transformation==2: df_plot resta intero
    df_plot = df_plot.dropna(subset=['dV'])
    df_plot = df_plot[(df_plot['V'] >= V_min) & (df_plot['V'] <= V_max)]

    x = df_plot['invP'].values
    y = df_plot['V'].values
    sx_plot = sigma_P / df_plot['P'].values ** 2
    sy_plot = np.full_like(y, sigma_V)

    if barre_errore == 0:
        plt.scatter(x, y, s=5, alpha=0.6)
    else:
        plt.errorbar(x, y, xerr=sx_plot, yerr=sy_plot, fmt='o', capsize=3, elinewidth=1, label=temp_label)
    x_fit = np.array([x.min(), x.max()])
    y_fit = res['slope'] * x_fit + res['intercept']
    a = fmt_with_error(res['slope'], res['dslope'], 3)
    b = fmt_with_error(res['intercept'], res['dint'], 1)
    # plt.plot(x_fit, y_fit, label=f"{temp_label}: a={a}, b={b}")
    plt.plot(x_fit, y_fit, label=f"{temp_label}")
    print(f"{res['file']}: tempo = {len(x)/10}")

plt.xlabel('1/Pressione [1/Pa]', fontsize=14)
plt.ylabel('Volume [m^2]', fontsize=14)
# plt.title('Fit ODR: PV=nRT a T costante', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.yaxis.offsetText.set_fontsize(12)
ax.xaxis.offsetText.set_fontsize(12)
plt.grid(True)
plt.tight_layout()

dest_folder = r"C:\Users\loren\Desktop\PvnRT\Grafici"
#os.makedirs(dest_folder, exist_ok=True)
out_path = os.path.join(dest_folder, "espansione.pdf")
plt.savefig(out_path, format="pdf", dpi=1000, bbox_inches="tight")
plt.show()

# === 3. Salvataggio parametri di fit ===
output_file = r'C:\Users\loren\Desktop\PvnRT\fit_parameters_odr.txt'
with open(output_file, 'w') as f:
    f.write('Temperatura\tcoeff_angolare\tintercetta\tincertezza_coeff_angolare\tincertezza_intercetta\n')
    for res in results:
        temp = os.path.splitext(res['file'])[0]
        f.write(
            f"{temp}\t{res['slope']:.6f}\t{res['intercept']:.6f}\t{res['dslope']:.6f}\t{res['dint']:.6f}\n"
        )
print(f"Parametri di fit ODR salvati in '{output_file}'")
