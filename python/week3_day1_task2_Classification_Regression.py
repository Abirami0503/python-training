# %%
# %% [markdown]
# # Week 3 – Day 1
# ## Task 2 – Diabetes Regression
# 
# ### Objective
# To build a regression model and evaluate its performance using MAE, RMSE and R-squared.

# %%
import pandas as pd
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
data = load_diabetes()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# %% [markdown]
# ## Model Training

# %%
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)
baseline_predictions = baseline_model.predict(X_test)

print("Linear Regression Model trained successfully!")
print("Baseline Model trained successfully!")

# %% [markdown]
# ## Model Evaluation

# %%
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
baseline_r2 = r2_score(y_test, baseline_predictions)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Baseline"],
    "MAE": [linear_mae, baseline_mae],
    "RMSE": [linear_rmse, baseline_rmse],
    "R-squared": [linear_r2, baseline_r2]
})

print("Model Comparison:")
print(comparison.to_string(index=False))

# %% [markdown]
# ## Observations
# 
# 1. Linear Regression performs better than the Baseline model.
# 2. Linear Regression has lower MAE and RMSE values.
# 3. Linear Regression has a higher R-squared value of 0.4526.
# 4. The Baseline model has a negative R-squared value.
# 
# ## Conclusion
# 
#Linear Regression is better than the Baseline model for this dataset.
#It provides more accurate predictions and
#explains approximately 45.26% of the variation in the target values.

# %% [markdown]
# ## Model Comparison (3 lines)
# Linear Regression performs better than the Baseline model.
# It has lower MAE and RMSE values.
# It also has a higher R-squared value.

# ## Metric for a Non-Technical Client (2 lines)
# I would show MAE because it is easy to understand.
# It represents the average absolute prediction error in the original target units.

