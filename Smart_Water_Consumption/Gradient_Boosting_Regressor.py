from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_scaled, y_train)


y_pred_gbr = gbr.predict(X_test_scaled)

mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))

print(f"Gradient Boosting MAE: {mae_gbr}")
print(f"Gradient Boosting RMSE: {rmse_gbr}")


# Gradient Boosting MAE: 112.72335895937816
# Gradient Boosting RMSE: 130.15828529427102
