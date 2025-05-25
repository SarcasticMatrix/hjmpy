class VolatilityModel:
    """
    Abstract class for volatility models.
    """

    def sigma(self, t: float, T: float) -> float:
        """
        Instantaneous volatility σ(t, T) at current time t and maturity T.

        :param float t: Current time.
        :param float T: Maturity time.
        :returns: Instantaneous volatility.
        :rtype: float
        :raises NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def calibrate(self, *args, **kwargs):
        """
        Calibrate the model on historical data (e.g., via PCA).

        :raises NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
