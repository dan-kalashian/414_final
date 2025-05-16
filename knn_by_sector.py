import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('calculated_stocks.csv')

kv = {
    'Industrials': 85,
    'Healthcare': 200,
    'Technology': 200,
    'Utilities': 65,
    'Financial Services': 300,
    'Basic Materials': 50,
    'Consumer Cyclical': 85,
    'Real Estate': 200,
    'Communication Services': 80,
    'Consumer Defensive': 35,
    'Energy': 100
}

for sector in df['Sector'].unique():
    dfx = df[df['Sector'] == sector]
    if len(dfx) < 10:
        continue  # skip very small sectors

    # Features & target arrays
    X = dfx['days_since_ipo'].values.reshape(-1, 1)
    y_vol = dfx['Vol'].values
    y_vv  = dfx['VV'].values

    # 80/20 split
    X_train, X_test, y_vol_train, y_vol_test, y_vv_train, y_vv_test = train_test_split(
        X, y_vol, y_vv, test_size=0.2, random_state=42
    )

    # For smooth curve
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    k = kv[sector]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: Volatility ---
    model_vol = KNeighborsRegressor(n_neighbors=k)
    model_vol.fit(X_train, y_vol_train)

    # Test-set predictions & metrics
    y_vol_pred_test = model_vol.predict(X_test)
    r2_vol = r2_score(y_vol_test, y_vol_pred_test)
    mse_vol = mean_squared_error(y_vol_test, y_vol_pred_test)
    mae_vol = mean_absolute_error(y_vol_test, y_vol_pred_test)

    # Smooth curve for visualization
    y_vol_smooth = model_vol.predict(X_plot)

    axes[0].scatter(X_test, y_vol_test, color='blue', s=5, alpha=0.6, label='Test data')
    axes[0].plot(X_plot, y_vol_smooth, color='black', label='KNN fit')
    axes[0].set_title(f"{sector}: Days Since IPO vs. Volatility")
    axes[0].set_xlabel("Days Since IPO")
    axes[0].set_ylabel("Volatility")
    axes[0].set_ylim(0, 50)
    axes[0].text(
        0.05, 0.95,
        f"k={k}\nR²: {r2_vol:.2f}\nMSE: {mse_vol:.2f}\nMAE: {mae_vol:.2f}",
        transform=axes[0].transAxes, fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.5)
    )

    # --- Plot 2: Variance of Volatility ---
    model_vv = KNeighborsRegressor(n_neighbors=k)
    model_vv.fit(X_train, y_vv_train)

    # Test-set predictions & metrics
    y_vv_pred_test = model_vv.predict(X_test)
    r2_vv = r2_score(y_vv_test, y_vv_pred_test)
    mse_vv = mean_squared_error(y_vv_test, y_vv_pred_test)
    mae_vv = mean_absolute_error(y_vv_test, y_vv_pred_test)

    y_vv_smooth = model_vv.predict(X_plot)

    axes[1].scatter(X_test, y_vv_test, color='blue', s=5, alpha=0.6, label='Test data')
    axes[1].plot(X_plot, y_vv_smooth, color='black', label='KNN fit')
    axes[1].set_title(f"{sector}: Days Since IPO vs. Volatility-of-Volatility")
    axes[1].set_xlabel("Days Since IPO")
    axes[1].set_ylabel("Volatility-of-Volatility")
    axes[1].set_ylim(0, 15)
    axes[1].text(
        0.05, 0.95,
        f"k={k}\nR²:: {r2_vv:.2f}\nMSE: {mse_vv:.2f}\nMAE: {mae_vv:.2f}",
        transform=axes[1].transAxes, fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(f"summary_knn_{sector}.png")
    plt.clf()
