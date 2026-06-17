def get_multipler(regime, direction):
    """
    Return SIP investment multiplier based on market regime and
    forecasted return direction.

    Parameters:
        regime    : str — "calm" or "volatile"
        direction : str — "up" or "down"

    Returns:
        float — multiplier to apply to base SIP amount
    """

    table = {
        ("calm", "up"): 1.5,   # Invest more in calm up months
        ("calm", "down"): 0.75, # Invest less in calm down months
        ("volatile", "up"): 1.0, # Invest normal amount in volatile up
        ("volatile", "down"): 0.5 # Invest much less in volatile down
    }

    key = (regime.lower(), direction.lower())

    if key not in table:
        raise ValueError(f"Invalid regime/direction combination: {key}")
    
    return table[key]

def get_investment_amount(regime, direction, base=500):
    """
    Return dollar amount to invest this month.

    Parameters:
        regime    : str — "calm" or "volatile"
        direction : str — "up" or "down"
        base      : float — base monthly SIP amount (default $500)

    Returns:
        float — dollar amount to invest
    """

    multiplier = get_multipler(regime, direction)
    return base * multiplier

if __name__ == "__main__":

    cases = [
        ("calm", "up"),
        ("calm", "down"),
        ("volatile", "up"),
        ("volatile", "down"),
    ]

    # Test all combinations of regime and direction
    print(f"{'Regime':<10} {'Forecast':<8} {'Multiplier':<12} {'Amount'}")
    print("-" * 40) # Separator line

    for regime, direction in cases:
        mult = get_multipler(regime, direction)
        amount = get_investment_amount(regime, direction)
        print(f"{regime:<10} {direction:<8} {mult:<12} ${amount}")