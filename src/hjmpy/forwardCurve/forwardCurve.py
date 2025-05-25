import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class ForwardCurve:
    """
    Represents a forward price curve for a given market or contract.

    Stores maturity dates and observed prices.
    """

    def __init__(self, dates: np.ndarray, prices: np.ndarray):
        """
        Initialize the ForwardCurve with dates and prices.

        :param np.ndarray dates: Array of maturities (datetime or float, e.g., years).
        :param np.ndarray prices: Array of forward prices corresponding to the dates.
        """
        self.dates = np.array(dates)
        self.prices = np.array(prices)
        # Build an interpolator for the forward price at any time T (log-linear interpolation)
        self._interp = interp1d(self.dates, np.log(self.prices),
                                kind='linear', fill_value="extrapolate")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, date_col: str, price_col: str):
        """
        Factory method to create a ForwardCurve from a Pandas DataFrame.

        :param pd.DataFrame df: DataFrame containing date and price columns.
        :param str date_col: Name of the column with dates.
        :param str price_col: Name of the column with prices.
        :returns: ForwardCurve instance.
        :rtype: ForwardCurve
        """
        dates = pd.to_datetime(df[date_col]).astype(np.int64) / (1e9 * 86400)  # convert to days or years
        prices = df[price_col].values
        return cls(dates, prices)

    def get_forward(self, T: float) -> float:
        """
        Get the forward price at maturity T by linearly interpolating (or extrapolating) the log-price.

        :param float T: Maturity time (e.g., years from reference).
        :returns: Forward price at time T.
        :rtype: float
        """
        logp = self._interp(T)
        return np.exp(logp)

    def log_returns(self) -> np.ndarray:
        """
        Compute log returns from stored prices.

        Useful for calibration methods such as PCA.

        :returns: Array of log returns.
        :rtype: np.ndarray
        """
        return np.diff(np.log(self.prices))

    def slice(self, start: float, end: float):
        """
        Extract a sub-curve between start and end maturities.

        :param float start: Start maturity.
        :param float end: End maturity.
        :returns: ForwardCurve instance with the sliced data.
        :rtype: ForwardCurve
        """
        mask = (self.dates >= start) & (self.dates <= end)
        return ForwardCurve(self.dates[mask], self.prices[mask])
