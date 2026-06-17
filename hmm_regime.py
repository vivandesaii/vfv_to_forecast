import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os


def load_data(file_path="data/vfv_processed.csv"):
    """
    Load the processed VFV data from CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please run data_loader.py first.")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def add_features(df):
    """
    Enhanced features for HMM:
    - rolling_vol:    21d realized volatility
    - vol_ratio:      21d vol / 63d vol (recent vs longer term)
    - vix:            implied volatility (market fear gauge)
    All features scaled to zero mean / unit variance.
    """
    df = df.copy()

    df["rolling_vol_63d"] = df["daily_return"].rolling(window=63).std()
    df["vol_ratio"] = df["rolling_vol"] / df["rolling_vol_63d"]

    df.dropna(inplace=True)

    # Scale all features
    scaler = StandardScaler()
    feature_cols = ["daily_return", "rolling_vol", "vol_ratio", "vix"]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df

def train_hmm(df, n_states=3, random_state=42):
    """
    Train a Gaussian HMM on 4 features:
    daily_return, rolling_vol, vol_ratio, vix.
    3 states: calm bull, choppy/sideways, volatile bear.

    Returns:
        model        : trained GaussianHMM
        states       : predicted state sequence
        state_probs  : posterior probabilities (n_days x n_states)
        df_enhanced  : DataFrame with added/scaled features
    """
    df_enhanced = add_features(df)
    features = df_enhanced[["daily_return", "rolling_vol", "vol_ratio", "vix"]].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=random_state
    )

    model.fit(features)
    states      = model.predict(features)
    state_probs = model.predict_proba(features)

    return model, states, state_probs, df_enhanced


def identify_calm_state(model, states, df_enhanced):
    """
    With 3 states, identify:
    - calm state     : lowest mean rolling_vol
    - volatile state : highest mean rolling_vol
    - middle state   : everything else (choppy/sideways)

    Returns calm_state index only — backtest uses this for allocation.
    """
    # rolling_vol is index 1 in feature vector
    mean_vols = [
        model.means_[s][1]
        for s in range(model.n_components)
    ]

    calm_state     = int(np.argmin(mean_vols))
    volatile_state = int(np.argmax(mean_vols))
    middle_state   = [s for s in range(model.n_components)
                      if s != calm_state and s != volatile_state][0]

    print(f"State {calm_state}  = CALM       (mean vol: {mean_vols[calm_state]:.4f})")
    print(f"State {middle_state} = CHOPPY     (mean vol: {mean_vols[middle_state]:.4f})")
    print(f"State {volatile_state} = VOLATILE   (mean vol: {mean_vols[volatile_state]:.4f})")

    return calm_state, middle_state, volatile_state


def plot_regimes(df_enhanced, states, calm_state, middle_state, volatile_state,
                 save_path="plots/price_with_regimes.png"):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_enhanced.index, df_enhanced["Close"], color="black", linewidth=0.8, zorder=2)

    color_map = {
        calm_state:     "lightgreen",
        middle_state:   "lightyellow",
        volatile_state: "lightcoral"
    }

    for i in range(len(states)):
        color = color_map.get(states[i], "lightgrey")
        ax.axvspan(df_enhanced.index[i],
                   df_enhanced.index[min(i + 1, len(df_enhanced) - 1)],
                   alpha=0.3, color=color, linewidth=0)

    patches = [
        mpatches.Patch(color="lightgreen",  alpha=0.5, label="Calm"),
        mpatches.Patch(color="lightyellow", alpha=0.5, label="Choppy"),
        mpatches.Patch(color="lightcoral",  alpha=0.5, label="Volatile"),
    ]
    ax.legend(handles=patches, fontsize=11)
    ax.set_title("VFV.TO — 3-State HMM Regime Labels", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (CAD)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")

def run(path="data/vfv_processed.csv"):
    df = load_data(path)
    model, states, state_probs, df_enhanced = train_hmm(df, n_states=3)
    calm_state, middle_state, volatile_state = identify_calm_state(
        model, states, df_enhanced
    )
    df_enhanced["regime"]     = states
    df_enhanced["confidence"] = [
        state_probs[i][states[i]] for i in range(len(states))
    ]
    plot_regimes(df_enhanced, states, calm_state, middle_state, volatile_state)
    return df_enhanced, model, calm_state, middle_state, volatile_state, state_probs

if __name__ == "__main__":
    df_enhanced, model, calm_state, middle_state, volatile_state, state_probs = run()
    states = df_enhanced["regime"].values
    print(f"\nMean confidence: {np.mean([state_probs[i][states[i]] for i in range(len(states))]):.4f}")