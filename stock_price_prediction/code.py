#%%

import pandas as pd
import sklearn as sk 
# %%
df = pd.read_csv("df_prices.csv", delimiter=",")
# %%
print(set(df["Ticker"]))
# %%
df = df[df["Ticker"] == "XUCD.XETRA"]
# %%
df = df[["date", "adjusted_close","volume"]]
df.sort_values(by="date", inplace=True)
df.reset_index(drop=True, inplace=True)
# %%
# Create additional features
for i in range(1, 6):
    df[f"lag_adjusted_close_{i}"] = df["adjusted_close"].shift(i)
    df[f"lag_volume_{i}"] = df["volume"].shift(i)

df["ma_5"] = df["adjusted_close"].rolling(5).mean()
df["volatility_5"] = df["adjusted_close"].rolling(5).std()
df["volume_change"] = df["volume"].pct_change()
df["price_change"] = df["adjusted_close"].pct_change()
df["vol_to_price"] = df["volume"] / df["adjusted_close"]
df.dropna(inplace=True)
# %%
print(df)
# %%
from sklearn.model_selection import train_test_split
features = df.drop(columns=["date", "adjusted_close"])
target = df["adjusted_close"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42, shuffle=False)
# %%
#Random Forest Regressor model
model = sk.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# %%
print(y_pred)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(target.index, target, label="Actual")
shifted_pred = pd.Series(y_pred, index=target.index[-len(y_pred):])

plt.plot(shifted_pred.index, shifted_pred, label="Predicted")
plt.legend()
plt.show()

# %%
# XGBoost Regressor model
model_xgb = sk.ensemble.GradientBoostingRegressor(n_estimators=100, random_state=42)
model_xgb.fit(X_train, y_train)
# %%
y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"Mean Squared Error XGBoost: {mse_xgb}")
# %%
plt.figure(figsize=(10,5))
plt.plot(target.index, target, label="Actual")
shifted_pred_xgb = pd.Series(y_pred_xgb, index=target.index[-len(y_pred_xgb):])
plt.plot(shifted_pred_xgb.index, shifted_pred_xgb, label="Predicted XGBoost")
plt.legend()
plt.show()
# %%
