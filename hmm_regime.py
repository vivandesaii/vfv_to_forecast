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
    df = df.copy()
    df["rolling_vol_63d"] = df["daily_return"].rolling(window=63).std()
    df["vol_ratio"] = df["rolling_vol"] / df["rolling_vol_63d"]
    df.dropna(inplace=True)

    # Scale features before HMM
    scaler = StandardScaler()
    feature_cols = ["daily_return", "rolling_vol", "vol_ratio"]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df

def train_hmm(df, n_states=2, random_state=42):
    """
    Train a Gaussian HMM on 4 features:
    daily_return, rolling_vol, vol_ratio, momentum.

    Returns:
        model       : trained GaussianHMM
        states      : predicted state sequence (array of 0s and 1s)
        state_probs : posterior probabilities per state (n_days x n_states)
        df_enhanced : DataFrame with added features (may have fewer rows after dropna)
    """
    df_enhanced = add_features(df)
    features = df_enhanced[["daily_return", "rolling_vol", "vol_ratio"]].values

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
    Identify which HMM state is calm by comparing mean rolling_vol.
    Lower mean rolling_vol = calm state.
    """
    # rolling_vol is index 1 in feature vector
    mean_vols = [
        model.means_[s][1]
        for s in range(model.n_components)
    ]

    calm_state     = int(np.argmin(mean_vols))
    volatile_state = 1 - calm_state

    print(f"State {calm_state} = CALM     (mean vol: {mean_vols[calm_state]:.4f})")
    print(f"State {volatile_state} = VOLATILE (mean vol: {mean_vols[volatile_state]:.4f})")

    return calm_state


def plot_regimes(df_enhanced, states, calm_state, save_path="plots/price_with_regimes.png"):
    """
    Plot VFV closing price with calm/volatile regime overlay.
    Saves to plots/price_with_regimes.png.
    """
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_enhanced.index, df_enhanced["Close"], color="black", linewidth=0.8, zorder=2)

    for i in range(len(states)):
        color = "lightgreen" if states[i] == calm_state else "lightcoral"
        ax.axvspan(
            df_enhanced.index[i],
            df_enhanced.index[min(i + 1, len(df_enhanced) - 1)],
            alpha=0.3, color=color, linewidth=0
        )

    calm_patch     = mpatches.Patch(color="lightgreen", alpha=0.5, label="Calm Regime")
    volatile_patch = mpatches.Patch(color="lightcoral", alpha=0.5, label="Volatile Regime")
    ax.legend(handles=[calm_patch, volatile_patch], fontsize=11)

    ax.set_title("VFV.TO Price with HMM Regime Labels (Enhanced Features)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (CAD)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")


def run(path="data/vfv_processed.csv"):
    """
    Full pipeline: load → add features → train HMM → 
    identify calm state → plot regimes.
    """
    df = load_data(path)
    model, states, state_probs, df_enhanced = train_hmm(df)
    calm_state = identify_calm_state(model, states, df_enhanced)

    df_enhanced["regime"]      = states
    df_enhanced["is_calm"]     = (states == calm_state).astype(int)
    df_enhanced["confidence"]  = [
        state_probs[i][states[i]] for i in range(len(states))
    ]

    plot_regimes(df_enhanced, states, calm_state)

    return df_enhanced, model, calm_state, state_probs


if __name__ == "__main__":
    df_enhanced, model, calm_state, state_probs = run()

    print(f"\nFeatures used: daily_return, rolling_vol, vol_ratio, momentum")
    print(f"Mean confidence score: {np.mean([state_probs[i][states[i]] for i in range(len(states))]):.4f}")