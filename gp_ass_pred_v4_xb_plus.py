import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
# Enhanced Evaluation
from sklearn.model_selection import cross_val_score
from category_encoders import LeaveOneOutEncoder


class MultiTargetEncoder:
    def __init__(self, cols):
        self.encoders = {col: TargetEncoder() for col in cols}
        self.cols = cols
    
    def fit_transform(self, X, y):
        for col in self.cols:
            X[col] = self.encoders[col].fit_transform(X[col], y)
        return X
    
    def transform(self, X):
        for col in self.cols:
            X[col] = self.encoders[col].transform(X[col])
        return X


# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Save target and IDs
train_target = train_df['SalePrice']
test_ids = test_df['Id']
train_df = train_df.drop(columns=['SalePrice', 'Id'])
test_df = test_df.drop(columns=['Id'])

# Identify numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=np.number).columns
categorical_cols = train_df.select_dtypes(include='object').columns

# Handle missing numerical values with train's median
for col in numerical_cols:
    median = train_df[col].median()
    train_df[col].fillna(median, inplace=True)
    test_df[col].fillna(median, inplace=True)

# Handle missing categorical values with train's mode
for col in categorical_cols:
    mode = train_df[col].mode()[0]
    train_df[col].fillna(mode, inplace=True)
    test_df[col].fillna(mode, inplace=True)

# Drop columns with >50% missing in train
missing_percent = train_df.isnull().mean()
columns_to_drop = missing_percent[missing_percent > 0.5].index
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

# Target encode high-cardinality categorical variables
high_cardinality = [col for col in categorical_cols if train_df[col].nunique() > 10]
low_cardinality = [col for col in categorical_cols if train_df[col].nunique() <= 10]

# Target encode high-cardinality features
# encoder = LeaveOneOutEncoder(cols=high_cardinality) causes overfitting
encoder = MultiTargetEncoder(cols=high_cardinality)
train_df[high_cardinality] = encoder.fit_transform(train_df[high_cardinality], np.log1p(train_target))
test_df[high_cardinality] = encoder.transform(test_df[high_cardinality])

# One-hot encode low-cardinality features
train_df = pd.get_dummies(train_df, columns=low_cardinality)
test_df = pd.get_dummies(test_df, columns=low_cardinality)

# Ensure test data has same columns as train
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Drop highly correlated numerical features
corr_matrix = train_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
columns_to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

# Log transform skewed features
skewed = ['GrLivArea', 'LotArea', '1stFlrSF', 'TotalBsmtSF']
for col in skewed:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])


# Feature engineering
temp_df = train_df.copy()
temp_df['SalePrice'] = train_target
neighborhood_median_price = temp_df.groupby('Neighborhood')['SalePrice'].median().to_dict()

for df in [train_df, test_df]:
    df['NeighborhoodMedianPrice'] = df['Neighborhood'].map(neighborhood_median_price)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)

# Log-transform target
y = np.log1p(train_target)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_df, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'early_stopping_rounds': 50
    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, val_preds))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=600, 
               callbacks=[optuna.study.MaxTrialsCallback(100)])
best_params = study.best_params



# Base models
xgb = XGBRegressor(**best_params, random_state=42)
lgbm = LGBMRegressor(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=1000,
            random_state=42,
        )
catboost = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            silent=True,
            random_state=42
        )

# Stacking
final_estimator = make_pipeline(
    StandardScaler(),
    LassoCV(alphas=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], 
            max_iter=10000, cv=5)
)

stacked_model = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('cat', catboost)
    ],
    final_estimator=final_estimator,
    cv=5
)



stacked_model.fit(X_train, y_train)


result = permutation_importance(stacked_model, X_val, y_val, n_repeats=10)
low_importance = result.importances_mean < 0.001

print( train_df.shape, train_df.loc[:, ~low_importance].shape)
print( test_df.shape, test_df.loc[:, ~low_importance].shape)
important_cols = train_df.columns[~low_importance]
train_df_reduced = train_df[important_cols]
test_df_reduced = test_df[important_cols]
x_val_reduced = X_val[important_cols]


scores = cross_val_score(stacked_model, train_df_reduced, y, 
                        cv=5, scoring='neg_root_mean_squared_error')
print(f"Cross-validated RMSE: {-scores.mean():.2f} ± {scores.std():.2f}")

# Final Training
stacked_model.fit(train_df_reduced, y)
val_preds = stacked_model.predict(x_val_reduced)

# Calculate R² (log and original space)
r2_log = r2_score(y_val, val_preds)
r2_original = r2_score(np.expm1(y_val), np.expm1(val_preds))

# Naive RMSE (predicting mean)
naive_pred = [np.expm1(y_val.mean())] * len(y_val)
naive_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), naive_pred))

final_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_preds)))

# MAE in log-transformed space (same scale as model training)
mae_log = mean_absolute_error(y_val, val_preds)
print(f"MAE (log space): {mae_log:.4f}")

mae_original = mean_absolute_error(np.expm1(y_val), np.expm1(val_preds))
print(f"MAE (original prices): ${mae_original:,.2f}")


print(f"R² (log space): {r2_log:.3f}")
print(f"R² (original prices): {r2_original:.3f}")
print(f"Naive RMSE (mean baseline): ${naive_rmse:,.2f}")

print(f"Final Validation RMSE: ${final_rmse:,.2f}")
print(f"Mean Price: ${np.expm1(y).mean():,.2f}")
print(f"Median Price: ${np.expm1(y).median():,.2f}")


residuals = np.expm1(y_val) - np.expm1(val_preds)
plt.scatter(np.expm1(y_val), residuals, alpha=0.3)
plt.axhline(0, color='red')
plt.xlabel("Actual Price")
plt.ylabel("Error ($)")
plt.show()

error_percent = (residuals / np.expm1(y_val)) * 100
plt.hist(error_percent, bins=50)
plt.xlabel("% Error")
plt.ylabel("Count")
plt.show()


# Final predictions
# Get stacked model predictions on training data
train_preds = stacked_model.predict(train_df_reduced)
train_mae = mean_absolute_error(y, train_preds)
print(f"Train MAE (log space): {train_mae:.4f}")

# Train calibrator on the residuals
calibrator = LinearRegression()
calibrator.fit(train_preds.reshape(-1, 1), y)

# Calibrated prediction function
def calibrated_predict(X):
    raw_preds = stacked_model.predict(X)
    return calibrator.predict(raw_preds.reshape(-1, 1))

lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.05)
upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.95)

lower_model.fit(train_df_reduced, y)
upper_model.fit(train_df_reduced, y)

def predict_with_intervals(X):
    pred = calibrated_predict(X)
    return {
        'prediction': np.expm1(pred),
        'lower_bound': np.expm1(lower_model.predict(X)),
        'upper_bound': np.expm1(upper_model.predict(X))
    }


# Generate final submission with confidence intervals
test_preds = predict_with_intervals(test_df_reduced)


submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_preds['prediction'],
    #'SalePrice_Lower': test_preds['lower_bound'],
    #'SalePrice_Upper': test_preds['upper_bound']
})

# Add useful metadata
# submission['Prediction_Error_Percent'] = (
#     (submission['SalePrice_Upper'] - submission['SalePrice_Lower']) / 
#     submission['SalePrice'] * 100
# )

submission.to_csv('final_submission_with_confidence.csv', index=False)





