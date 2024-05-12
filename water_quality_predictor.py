import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class WaterQualityPredictor:
    def __init__(self, data_path):
        """
        Initialize the WaterQualityPredictor class.

        Parameters:
        - data_path (str): Path to the CSV file containing the dataset.
        """
        self.data_path = data_path
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preprocessor = None
        self.models = None
        self.best_models = None

    def load_data(self):
        """
        Load the dataset from the CSV file and perform initial preprocessing.
        """
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        numeric_data = self.data.select_dtypes(include=['number'])
        self.data_imputed = numeric_data.fillna(numeric_data.mean())
        self.X = self.data_imputed.drop("DissolvedOxygen (mg/L)", axis=1)
        self.y = self.data_imputed["DissolvedOxygen (mg/L)"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def preprocess_data(self):
        """
        Preprocess the data by imputing missing values and scaling features.
        """
        self.preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

    def define_models(self):
        """
        Define the machine learning models to be trained.
        """
        self.models = {
            'XGBoost': xgb.XGBRegressor(),
            'RandomForest': RandomForestRegressor(max_features='log2', bootstrap=True)
        }

    def train_models(self):
        """
        Train each defined model using hyperparameter tuning.
        """
        self.best_models = {}
        for name, model in self.models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            param_grid = {}
            if name == 'XGBoost':
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__max_depth': [3, 4, 5],
                    'model__n_jobs': [4]
                }
            elif name == 'RandomForest':
                param_grid = {
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [3, 4, 5, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=KFold(n_splits=5, shuffle=True), scoring='neg_mean_squared_error')
            grid_search.fit(self.X_train, self.y_train)
            self.best_models[name] = grid_search.best_estimator_

    def evaluate_models(self):
        """
        Evaluate the performance of each trained model.
        """
        for name, model in self.best_models.items():
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"\n{name} Mean Squared Error (MSE):", mse)
            print(f"{name} R-squared (R²) Score:", r2)
