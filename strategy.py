def get_multiplier(regime, direction, confidence=1.0):
    """
    Return SIP multiplier for 3-state regime + direction + confidence.

    Regimes:
        calm     = low vol, bull market conditions
        choppy   = elevated vol, unclear direction
        volatile = high vol, stress conditions

    Confidence scales multiplier toward neutral (1.0x) when uncertain.

    Parameters:
        regime     : str   — "calm", "choppy", or "volatile"
        direction  : str   — "up" or "down"
        confidence : float — HMM probability of current regime (0.5–1.0)

    Returns:
        float — multiplier to apply to base SIP amount
    """
    base_table = {
    ("calm",     "up"):   1.25,
    ("calm",     "down"): 0.875,
    ("choppy",   "up"):   0.95,
    ("choppy",   "down"): 1.00,
    ("volatile", "up"):   1.00,
    ("volatile", "down"): 1.30,
}

    key = (regime.lower(), direction.lower())
    if key not in base_table:
        raise ValueError(f"Invalid input: regime='{regime}', direction='{direction}'")

    base    = base_table[key]
    neutral = 1.0
    scaled  = neutral + (base - neutral) * confidence

    return round(scaled, 4)


def get_investment_amount(regime, direction, confidence=1.0, base=500):
    """
    Return dollar amount to invest this month.

    Parameters:
        regime     : str   — "calm", "choppy", or "volatile"
        direction  : str   — "up" or "down"
        confidence : float — HMM probability of current regime
        base       : float — base monthly SIP amount (default $500)

    Returns:
        float — dollar amount to invest
    """
    multiplier = get_multiplier(regime, direction, confidence)
    return round(base * multiplier, 2)


if __name__ == "__main__":
    print(f"{'Regime':<10} {'Forecast':<8} {'Conf':<8} {'Multiplier':<12} {'Amount'}")
    print("-" * 50)

    for regime in ["calm", "choppy", "volatile"]:
        for direction in ["up", "down"]:
            for conf in [1.0, 0.75]:
                mult   = get_multiplier(regime, direction, conf)
                amount = get_investment_amount(regime, direction, conf)
                print(f"{regime:<10} {direction:<8} {conf:<8} {mult:<12} ${amount}")
        print()