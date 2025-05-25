from hjmpy.volatilityModel.volatilityModel import VolatilityModel

import numpy as np


class ExponentialVolatilityModel(VolatilityModel):
    """
    Example of an exponential volatility model based on the time horizon (Samuelson effect).

    The volatility function is defined as:
    σ(T - t) = gamma * exp(-k * (T - t))
    """
    def __init__(self, gamma: float, k: float):
        """
        Initialize the exponential volatility model with parameters gamma and k.

        :param float gamma: Scale parameter controlling initial volatility level.
        :param float k: Decay rate parameter controlling how volatility decreases over time.
        """
        self.gamma = gamma
        self.k = k
    
    def sigma(self, t: float, T: float) -> float:
        """
        Compute the volatility σ at time t for maturity T.

        :param float t: Current time.
        :param float T: Maturity time.
        :returns: Volatility at horizon (T - t).
        :rtype: float
        """
        tau = T - t
        return self.gamma * np.exp(-self.k * tau)
    
    def calibrate(self, time_to_maturities: np.ndarray, vols: np.ndarray):
        """
        Calibrate the parameters gamma and k from historical volatilities.

        Uses nonlinear least squares fitting (SciPy's curve_fit) to fit the model
        σ(τ) = gamma * exp(-k * τ) to observed volatilities as a function of τ = T - t.

        :param np.ndarray time_to_maturities: Array of time-to-maturity values (τ).
        :param np.ndarray vols: Corresponding observed volatilities.
        :returns: None (updates gamma and k in place).
        """
        from scipy.optimize import curve_fit
        def model(tau, gamma, k): return gamma * np.exp(-k * tau)
        popt, _ = curve_fit(model, time_to_maturities, vols, p0=[self.gamma, self.k])
        self.gamma, self.k = popt


