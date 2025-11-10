=INDEX(B$2:D$100, MATCH(F2, A$2:A$100, 0), MATCH(2, 1/(INDEX(B$2:D$100, MATCH(F2, A$2:A$100, 0), )<>"" )))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assume results are stored in DataFrames named:
# results_ar, results_arma, results_ols, results_psf
# each having columns: ['Quarter', 'Forecast', 'Actual']
# and indexed by Quarter

# Combine forecast results for plotting
df_all = pd.DataFrame()
df_all['Actual'] = results_ar['Actual']  # 
Actual is the same in all
df_all['AR'] = results_ar['Forecast']
df_all['ARMA'] = results_arma['Forecast']
df_all['Indicator'] = results_ols['Forecast']
df_all['PSF'] = results_psf['Forecast']

# Plot time series forecasts vs actuals
plt.figure(figsize=(14, 6))
plt.plot(df_all.index, df_all['Actual'], label='Actual', linewidth=3, color='black')
plt.plot(df_all.index, df_all['AR'], label='AR', linestyle='--')
plt.plot(df_all.index, df_all['ARMA'], label='ARMA', linestyle='--')
plt.plot(df_all.index, df_all['Indicator'], label='Indicator OLS', linestyle='-.')
plt.plot(df_all.index, df_all['PSF'], label='PSF', linestyle=':')
plt.title('Nowcasting Forecasts vs Actual GDP')
plt.xlabel('Quarter')
plt.ylabel('GDP Growth')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE for each model
def compute_rmse(forecast, actual):
    return np.sqrt(((forecast - actual)**2).mean())

rmse_results = {
    'AR': compute_rmse(df_all['AR'], df_all['Actual']),
    'ARMA': compute_rmse(df_all['ARMA'], df_all['Actual']),
    'Indicator OLS': compute_rmse(df_all['Indicator'], df_all['Actual']),
    'PSF': compute_rmse(df_all['PSF'], df_all['Actual']),
}

# Plot RMSE bar chart
plt.figure(figsize=(8, 5))
plt.bar(rmse_results.keys(), rmse_results.values(), color='skyblue')
plt.title('RMSE of Nowcasting Models')
plt.ylabel('RMSE')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

