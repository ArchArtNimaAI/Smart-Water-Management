import matplotlib.pyplot as plt
import seaborn as sns

results = {
    'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network'],
    'MAE': [mae, mae_gbr, mae_nn],
    'RMSE': [rmse, rmse_gbr, rmse_nn]
}
results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=results_df)
plt.title('Mean Absolute Error (MAE) Comparison')
plt.ylabel('MAE')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.ylabel('RMSE')
plt.show()
