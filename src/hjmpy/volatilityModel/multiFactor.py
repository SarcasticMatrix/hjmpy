from hjmpy.volatilityModel.volatilityModel import VolatilityModel

import numpy as np
from sklearn.decomposition import PCA

class MultiFactorVolatilityModel(VolatilityModel):
    """
    Multi-factor volatility model calibrated using PCA.

    Stores principal components and can reconstruct the volatility matrix.
    """
    def __init__(self, n_factors: int = 3):
        """
        Initialize the multi-factor volatility model.

        :param int n_factors: Number of principal components (factors) to keep.
        """
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors)
        self.components_ = None  # composants propres (n_factors x N)
        self.explained_variance_ = None
    
    def calibrate(self, log_return_matrix: np.ndarray):
        """
        Calibrate the model via PCA on the log return matrix.

        The input matrix should be shaped (n_observations, n_series),
        where each column corresponds to an asset's log returns.

        :param np.ndarray log_return_matrix: Matrix of log returns for calibration.
        :returns: None (updates PCA components and explained variance).
        """
        # log_return_matrix: shape (n_obs, n_series)
        self.pca.fit(log_return_matrix)
        self.components_ = self.pca.components_  # shape (n_factors, n_series)
        self.explained_variance_ = self.pca.explained_variance_ratio_
        # Les composantes principales vont servir à construire les facteurs de volatilité.

    def sigma(self, t: float, T: float):
        """
        Example method returning a volatility vector σ(t, T) of dimension n_factors
        depending on the time horizon (T - t).

        For simplicity, returns a vector with exponentially decaying components.

        :param float t: Current time.
        :param float T: Maturity time.
        :returns: Volatility vector at horizon (T - t).
        :rtype: np.ndarray
        """
        # Ici, on illustre: vol = exp(-k*(T-t)) pour chaque composante
        tau = T - t
        # On peut calibrer k séparément ou utiliser composantes
        return np.array([np.exp(-0.5 * tau)] * self.n_factors)