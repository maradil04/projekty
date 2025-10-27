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
EPS = 1e-8
for i in range(1, 6):
    df[f"lag_adjusted_close_{i}"] = df["adjusted_close"].shift(i)
    df[f"lag_volume_{i}"] = df["volume"].shift(i)

df["ma_5"] = df["adjusted_close"].rolling(5).mean().shift(1)
df["volatility_5"] = df["adjusted_close"].rolling(5).std().shift(1)
df["volume_change"] = df["volume"].pct_change().shift(1)
df["price_change"] = df["adjusted_close"].pct_change().shift(1)
df["vol_to_price"] = (df["volume"] / (df["adjusted_close"] + EPS)).shift(1)
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
# XGBoost Regressor model (not ideal since my dataset only has 1 year of data)
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









#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("indexProcessed.csv", delimiter=",")
df = df[df["Index"] == "HSI"][["Date","Adj Close","Volume"]].copy()
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

df["log_price"] = np.log(df["Adj Close"])
df["log_ret"] = df["log_price"].diff()
df.dropna(inplace=True)

y = df["log_ret"]

n = len(y)
train_size = int(n*0.7)
val_size   = int(n*0.15)

y_train = y.iloc[:train_size]
y_val   = y.iloc[train_size:train_size+val_size]
y_test  = y.iloc[train_size+val_size:]

P_last_trainval = df["Adj Close"].iloc[train_size+val_size-1:]
P_anchor = P_last_trainval.iloc[0]

#%%
def best_arma_order(y_train, max_p=5, max_q=5):
    best_ic = np.inf
    best = (0,0)
    for p in range(max_p+1):
        for q in range(max_q+1):
            if p==0 and q==0: 
                continue
            try:
                res = ARIMA(y_train, order=(p,0,q)).fit()
                if res.bic < best_ic:   
                    best_ic, best = res.bic, (p,q)
            except:
                pass
    return best
#%%
p,q = best_arma_order(pd.concat([y_train, y_val]), max_p=4, max_q=4)
print("Zvolený ARMA(p,q):", (p,q))

model = ARIMA(pd.concat([y_train, y_val]), order=(p,0,q)).fit()
fc_ret = model.forecast(steps=len(y_test)) 

cum_ret = np.cumsum(fc_ret.values)
fc_price = P_anchor * np.exp(cum_ret)
true_price = df["Adj Close"].iloc[train_size+val_size:]

rmse_test = sqrt(mean_squared_error(true_price.values, fc_price))
print(f"Test RMSE (price from predicted returns): {rmse_test:.4f}")

#%%
plt.figure(figsize=(10,5))
plt.plot(df["Adj Close"].iloc[:train_size+val_size], label="Train+Val")
plt.plot(true_price.index, true_price.values, label="Test (true)")
plt.plot(true_price.index, fc_price, label="Forecast (returns→price)")
plt.title(f"ARMA({p},{q}) na log-returnech, RMSE={rmse_test:.2f}")
plt.legend(); plt.show()



# %%

from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 400
p, q = best_order

y_log = np.log(df["Adj Close"])
model_full = ARIMA(y_log, order=(p,1,q), trend="t").fit()

res = model_full.get_forecast(steps=N)
fc_log = res.predicted_mean
ci_log = res.conf_int(alpha=0.05)

# vezmi první a druhý sloupec bez ohledu na názvy
ci_low_log  = ci_log.iloc[:, 0]
ci_high_log = ci_log.iloc[:, 1]

# zpět na ceny
fc_price = np.exp(fc_log)
ci_low   = np.exp(ci_low_log)
ci_high  = np.exp(ci_high_log)

future_dates = pd.date_range(df.index[-1], periods=N+1, freq="B")[1:]

plt.figure(figsize=(10,5))
plt.plot(df.index, df["Adj Close"], label="Historie")
plt.plot(future_dates, fc_price.values, label=f"Predikce {N} dní")
plt.fill_between(future_dates, ci_low.values, ci_high.values, alpha=0.2, label="95% CI")
plt.title(f"ARIMA({p},1,{q}) s lineárním trendem (d=1, trend='t')")
plt.legend(); plt.show()




# %%
