import numpy as np
import pandas as pd

np.random.seed(42)
markets = ["Power_FR", "Power_DE", "Gas_TTF"]
maturities = np.arange(1, 13)  # échéances M+1..M+12

# Dates : 252 jours ouvrés à partir du 01/01/2024
dates = pd.bdate_range("2024-01-01", periods=252*2, tz='UTC')

# Courbes forward initiales (valeurs de base par marché, ici choix arbitraire)
F0 = {
    "Power_FR": np.array([50.0 + 0.5*(m-1) for m in maturities]),
    "Power_DE": np.array([52.0 + 0.4*(m-1) for m in maturities]),
    "Gas_TTF":  np.array([ 3.0 + 0.1*(m-1) for m in maturities])
}

# Expositions aux facteurs : décroissance exponentielle en T, 3 facteurs (global, électricité, gaz)
alphas = [0.05, 0.20, 0.15]  # paramètres de décroissance
exposures = np.zeros((len(markets), len(maturities), 3))
for i, m in enumerate(maturities):
    expos = np.exp(-alphas[0]*m)   # facteur global
    exposures[0,i,0] = expos  # Power_FR
    exposures[1,i,0] = expos  # Power_DE
    exposures[2,i,0] = expos  # Gas_TTF
    expos_elec = np.exp(-alphas[1]*m)
    exposures[0,i,1] = expos_elec  # facteur électricité sur FR
    exposures[1,i,1] = expos_elec  # sur DE
    expos_gaz = np.exp(-alphas[2]*m)
    exposures[2,i,2] = expos_gaz   # facteur gaz sur Gas_TTF

# Volatilités des facteurs et bruit idiosyncratique
factor_vol = np.array([0.03, 0.03, 0.04])  # volat. journalières des 3 facteurs
idiosyn_vol = 0.005

# DataFrames pour stocker les courbes simulées
forward_prices = {mkt: pd.DataFrame(index=dates, columns=maturities) for mkt in markets}
for mkt in markets:
    forward_prices[mkt] = forward_prices[mkt].astype(float)
    forward_prices[mkt].iloc[0] = F0[mkt]  # courbe initiale au jour 0

# Simulation journalière
for t in range(1, len(dates)):
    # tirer les facteurs du jour t
    factors = np.random.normal(0, 1, 3) * factor_vol
    eps = np.random.normal(0, idiosyn_vol, (len(markets), len(maturities)))
    for i_mkt, mkt in enumerate(markets):
        prev = forward_prices[mkt].iloc[t-1].values.astype(float)
        delta_log = np.zeros(len(maturities))
        for j in range(3):
            delta_log += factors[j] * exposures[i_mkt,:,j]
        delta_log += eps[i_mkt]
        forward_prices[mkt].iloc[t] = prev * np.exp(delta_log)
        

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
for mkt in markets:
    plt.plot(forward_prices[mkt].index, forward_prices[mkt][1], label=f"{mkt} (M+1)", marker='x')
plt.title("Time Series des prix forward (échéance M+1)")
plt.xlabel("Date")
plt.ylabel("Prix Forward")
plt.legend()
plt.grid(linestyle="--", dashes=(5, 10), color="gray", linewidth=0.5)
plt.tight_layout()
plt.show()
