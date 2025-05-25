from hjmpy.volatilityModel.volatilityModel import VolatilityModel
from hjmpy.market.market import Market

import numpy as np

class HJMModel:
    """
    Multi-factor HJM model for multiple markets.

    Manages calibration (via PCA) and application of HJM dynamics
    across several markets.
    """
    def __init__(self, volatility_model: VolatilityModel):
        """
        Initialize the HJM model with a volatility model.

        :param VolatilityModel volatility_model: An instance of the volatility model used for calibration and dynamics.
        """
        self.vol_model = volatility_model
        self.markets = {}
    
    def add_market(self, market: Market):
        """
        Add a market (potentially with multiple curves) to the model.

        :param Market market: Market instance to be integrated into the model.
        """
        self.markets[market.name] = market

    def calibrate(self):
        """
        Calibrate the HJM model by performing PCA on log returns of forward curves.

        Constructs the observation-product matrix across all markets' forward curves,
        then uses PCA to calibrate the volatility model.

        :returns: None
        """
        returns = []
        for market in self.markets.values():
            for curve in market.curves.values():
                # On s'assure que la courbe a suffisamment de points pour calculer des rendements
                if len(curve.prices) > 1:
                    r = curve.log_returns()
                    # On prend la m\u00eame taille pour toutes (en tronquant ou remplissant)
                    returns.append(r)
        # Aligner les longueurs : on coupe au plus petit vecteur de rendements
        min_len = min(len(r) for r in returns)
        mat = np.column_stack([r[-min_len:] for r in returns])  # shape (min_len, n_products)
        # PCA
        self.vol_model.calibrate(mat)
        print(f"Explained variance (top {self.vol_model.n_factors} factors): ",
              self.vol_model.explained_variance_)

    def forward_dynamics(self, market_name: str, curve_name: str, t0: float, t1: float):
        """
        Compute analytically the forward price density F(t1,T) from F(t0,T)
        without simulation. Assumes zero drift under the risk-neutral measure Q:

        F(t1) = F(t0) * exp(-0.5 * Var + stochastic_term),

        where Var is computed from the integral of sigma squared.

        :param str market_name: Name of the market containing the forward curve.
        :param str curve_name: Name of the forward curve within the market.
        :param float t0: Initial time.
        :param float t1: Future time at which to evaluate the forward price.

        :returns: Expected forward price at time t1.
        :rtype: float
        """
        market = self.markets[market_name]
        curve = market.get_curve(curve_name)
        T = curve.dates[-1]  # maturit\u00e9 ultime de la courbe (ex: fin de livraison)
        # Exemple simplifi\u00e9 : on utilise un seul facteur constant pour d\u00e9mo
        sigma = self.vol_model.sigma(t0, T)
        if isinstance(sigma, np.ndarray):
            # Multi-facteurs : on somme les variances
            var = np.sum(sigma**2) * (t1 - t0)
        else:
            var = sigma**2 * (t1 - t0)
        F0 = curve.get_forward(T)
        # On ne simule pas W (monte-carlo interdite) : on donne le log-normale sans tirage
        drift = -0.5 * var
        # Valeur attendue : exp(drift), variance log-normal Var
        F1_expected = F0 * np.exp(drift)
        return F1_expected

    def price_forward(self, market_name: str, curve_name: str, t: float):
        """
        Return the forward price at time t for the final delivery of the curve.

        This may integrate the expectation calculation under the HJM dynamics.

        :param str market_name: Name of the market containing the forward curve.
        :param str curve_name: Name of the forward curve.
        :param float t: Time at which to price the forward.

        :returns: Forward price at time t.
        :rtype: float
        """
        # Pour l'instant, on renvoie le prix actuel (pas de temps dynamique)
        market = self.markets[market_name]
        curve = market.get_curve(curve_name)
        T = curve.dates[-1]
        return curve.get_forward(T)
