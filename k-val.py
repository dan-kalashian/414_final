import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('calculated_stocks.csv')
k_values = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
             80, 85, 90, 95, 100]
"""
for sector in df['Sector'].unique():
    print(f"\n--- Sector: {sector} ---")
    dfx = df[df['Sector'] == sector]
    X = dfx['days_since_ipo'].values.reshape(-1, 1)

    y_vol = dfx['Vol'].values
    y_vv = dfx['VV'].values

    X_train, X_test, y_vol_train, y_vol_test = train_test_split(X, y_vol, test_size=0.2, random_state=42)
    _, _, y_vv_train, y_vv_test = train_test_split(X, y_vv, test_size=0.2, random_state=42)

    for k in k_values:

        model_vol = KNeighborsRegressor(n_neighbors=k)
        model_vol.fit(X_train, y_vol_train)
        y_vol_pred = model_vol.predict(X_test)
        r2_vol = r2_score(y_vol_test, y_vol_pred)
        mse_vol = mean_squared_error(y_vol_test, y_vol_pred)
        mae_vol = mean_absolute_error(y_vol_test, y_vol_pred)

        model_vv = KNeighborsRegressor(n_neighbors=k)
        model_vv.fit(X_train, y_vv_train)
        y_vv_pred = model_vv.predict(X_test)
        r2_vv = r2_score(y_vv_test, y_vv_pred)
        mse_vv = mean_squared_error(y_vv_test, y_vv_pred)
        mae_vv = mean_absolute_error(y_vv_test, y_vv_pred)

        print(f"k={k:>3} | Vol -> R²: {r2_vol:.3f}, MSE: {mse_vol:.3f}, MAE: {mae_vol:.3f} | "
              f"VV -> R²: {r2_vv:.3f}, MSE: {mse_vv:.3f}, MAE: {mae_vv:.3f}")
"""

for sector in ['Healthcare', 'Technology', 'Financial Services', 'Real Estate', 'Energy']:
    print(f"\n--- Sector: {sector} ---")
    dfx = df[df['Sector'] == sector]
    X = dfx['days_since_ipo'].values.reshape(-1, 1)

    y_vol = dfx['Vol'].values
    y_vv = dfx['VV'].values

    X_train, X_test, y_vol_train, y_vol_test = train_test_split(X, y_vol, test_size=0.2, random_state=42)
    _, _, y_vv_train, y_vv_test = train_test_split(X, y_vv, test_size=0.2, random_state=42)

    for k in [100, 200, 300, 400, 500, 750, 100]:

        model_vol = KNeighborsRegressor(n_neighbors=k)
        model_vol.fit(X_train, y_vol_train)
        y_vol_pred = model_vol.predict(X_test)
        r2_vol = r2_score(y_vol_test, y_vol_pred)
        mse_vol = mean_squared_error(y_vol_test, y_vol_pred)
        mae_vol = mean_absolute_error(y_vol_test, y_vol_pred)

        model_vv = KNeighborsRegressor(n_neighbors=k)
        model_vv.fit(X_train, y_vv_train)
        y_vv_pred = model_vv.predict(X_test)
        r2_vv = r2_score(y_vv_test, y_vv_pred)
        mse_vv = mean_squared_error(y_vv_test, y_vv_pred)
        mae_vv = mean_absolute_error(y_vv_test, y_vv_pred)

        print(f"k={k:>3} | Vol -> R²: {r2_vol:.3f}, MSE: {mse_vol:.3f}, MAE: {mae_vol:.3f} | "
              f"VV -> R²: {r2_vv:.3f}, MSE: {mse_vv:.3f}, MAE: {mae_vv:.3f}")
