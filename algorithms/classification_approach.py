from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data_prep.dataprep  import DataPrep
from utils.data_utils import get_results_classification
from sklearn.preprocessing import LabelEncoder


models= [
    ('Logistic Regression', LogisticRegression(),  {'C': [0.1, 1, 10]}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [100], 'max_depth':[10,20,30]}),
    ('K-Nearest Neighbors', KNeighborsClassifier(), {'n_neighbors': [5, 7]}),
    ('Gradient Boosting Classifier ', GradientBoostingClassifier(), {'learning_rate': [0.1, 0.05]}),
]


data = DataPrep(data_path="data")
X,y = data.merged.loc[:, ~data.merged.columns.isin(['timestamp', 'rating'])], data.merged['rating']


X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.33, random_state=42)

model_results = {}
for model_name, model, params in models:
    performance = get_results_classification(model, model_name, params,  X_train, X_test, y_train, y_test)
    model_results[model_name] = {'model': model, 'performance': performance}
