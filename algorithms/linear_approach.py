from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_prep.dataprep import DataPrep
from utils.data_utils import get_results_linear


models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('Random Forest', RandomForestRegressor()),
    ('Linear Regression', LinearRegression()),
    ('Gradient Boosting', GradientBoostingRegressor()),
]

data = DataPrep(data_path="data_test1")
X, y = data.merged.loc[:, ~data.merged.columns.isin(['timestamp', 'rating'])], data.merged['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_results = {}
for model_name, model in models:
    performance = get_results_linear(model, model_name, X_train, X_test, y_train, y_test)
    model_results[model_name] = {'model': model, 'performance': performance}

# Determine the best model
best_model_name, best_model_info = max(model_results.items(), key=lambda x: x[1]['performance'][0])  # compare based on R2
best_model_rgr = best_model_info['model']
print(f"Best Model: {best_model_name} with R2 Score: {best_model_info['performance']}")