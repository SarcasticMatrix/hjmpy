class Market:
    """
    Represents a commodity market (e.g., electricity in zone A).

    Contains multiple ForwardCurve instances corresponding to various horizons or tenors.
    """

    def __init__(self, name: str, region: str = None, commodity: str = 'electricity'):
        """
        Initialize the Market with a name, optional region, and commodity type.

        :param str name: Name of the market.
        :param str region: Geographic or regulatory region of the market.
        :param str commodity: Type of commodity traded (default is 'electricity').
        """
        self.name = name
        self.region = region
        self.commodity = commodity
        self.curves = {}  # Dictionary of ForwardCurve objects by name or ID

    def add_forward_curve(self, curve_name: str, forward_curve):
        """
        Add a forward curve associated with this market.

        :param str curve_name: Name or identifier of the forward curve.
        :param forward_curve: ForwardCurve instance to add.
        """
        self.curves[curve_name] = forward_curve

    def get_curve(self, curve_name: str):
        """
        Retrieve a forward curve by its name.

        :param str curve_name: Name or identifier of the forward curve.
        :returns: ForwardCurve instance if found, else None.
        """
        return self.curves.get(curve_name)

    def list_curves(self):
        """
        List all forward curve names available in this market.

        :returns: List of forward curve names.
        :rtype: list[str]
        """
        return list(self.curves.keys())
