import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM
import os


def load_data(file_path="data/vfv_processed.csv"):
    """
    Load the processed VFV data from CSV.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please run data_loader.py first.")
    # Load CSV with date parsing
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def train_hmm(df, n_states=2, random_state=42):
    """
    Train a Gaussian HMM on daily return and rolling volatility.
    Returns the trained model and state sequence.
    """
    features = df[["daily_return", "rolling_vol"]].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=random_state
    )
    
    model.fit(features)
    states = model.predict(features)
    return model, states

def identify_calm_state(model, states):
    """
    Identify which HMM state is calm by comparing mean rolling vol.
    Lower mean rolling vol = calm state.
    """
    mean_vols = [
        model.means_[s][1]
        for s in range(model.n_components)
    ]

    calm_state = np.argmin(mean_vols) # Calm state has lowest mean rolling vol
    volatile_state = 1 - calm_state

    print(f"State {calm_state} is calm (mean vol={mean_vols[calm_state]:.4f})")
    print(f"State {volatile_state} is volatile (mean vol={mean_vols[volatile_state]:.4f})")

    return calm_state

def plot_regimes(df, states, calm_state, save_path="plots/regimes.png"):
    """
    Plot daily returns colored by HMM regime.
    """
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot price
    ax.plot(df.index, df["Close"], color="black", linewidth=0.8, zorder=2)

    # Shade regimes
    for i in range(len(states)):
        color = "lightgreen" if states[i] == calm_state else "lightcoral"
        ax.axvspan(df.index[i], df.index[min(i + 1, len(df) - 1)],
                   alpha=0.3, color=color, linewidth=0)

    # Legend
    calm_patch = mpatches.Patch(color="lightgreen", alpha=0.5, label="Calm Regime")
    volatile_patch = mpatches.Patch(color="lightcoral", alpha=0.5, label="Volatile Regime")
    ax.legend(handles=[calm_patch, volatile_patch], fontsize=11)

    ax.set_title("VFV.TO Price with HMM Regime Labels", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (CAD)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")

def run(path="data/vfv_processed.csv"):
    """Run the full HMM regime analysis pipeline. Load data, train HMM, identify calm state, and plot regimes."""

    df = load_data(path)
    model, states = train_hmm(df)
    calm_state = identify_calm_state(model, states)

    df["regime"] = states
    df["is_calm"] = (states == calm_state).astype(int)

    plot_regimes(df, states, calm_state)

    return df, model, calm_state

if __name__ == "__main__":
    run()