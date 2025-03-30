import pandas as pd
import numpy as np
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import QuantileTransformer
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from tensorflow.keras.losses import Huber

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
train_df = train_df.drop(columns=['Id', 'SalePrice'])
test_df = test_df.drop(columns=['Id',])

# Create temp df with target for feature engineering
temp_df = train_df.copy()
temp_df['SalePrice'] = train_target

# 1. Feature Engineering (Before Encoding)
# ---------------------------------------
# Neighborhood features (using temp_df with SalePrice)
neighborhood_stats = temp_df.groupby('Neighborhood')['SalePrice'].agg(['mean', 'median']).reset_index()
train_df['Neigh_PriceMean'] = train_df['Neighborhood'].map(neighborhood_stats['mean'])
train_df['Neigh_PriceMedian'] = train_df['Neighborhood'].map(neighborhood_stats['median'])
test_df['Neigh_PriceMean'] = test_df['Neighborhood'].map(neighborhood_stats['mean'])
test_df['Neigh_PriceMedian'] = test_df['Neighborhood'].map(neighborhood_stats['median'])

# Fill any missing neighborhood values
for col in ['Neigh_PriceMean', 'Neigh_PriceMedian']:
    global_mean = train_df[col].mean()
    train_df[col].fillna(global_mean, inplace=True)
    test_df[col].fillna(global_mean, inplace=True)

# Identify numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=np.number).columns
categorical_cols = train_df.select_dtypes(include='object').columns

# Handle missing numerical values with train's median
for col in numerical_cols:
    median = train_df[col].median()
    train_df[col].fillna(median, inplace=True)
    if col in test_df.columns:  # Only fill if column exists in test
        test_df[col].fillna(median, inplace=True)

# Handle missing categorical values with train's mode
for col in categorical_cols:
    mode = train_df[col].mode()[0]
    train_df[col].fillna(mode, inplace=True)
    if col in test_df.columns:
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

# Remove houses with extremely large living areas
train_df = train_df[train_df['GrLivArea'] < 4500]
train_target = train_target[train_df.index]  # Align target with filtered data


# Feature engineering
def add_features(df):
    # Basic features
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['TotalBathrooms'] = df.get('FullBath', 0) + 0.5*df.get('HalfBath', 0)
    df['YearsSinceRemodel'] = df.get('YrSold', 0) - df.get('YearRemodAdd', 0)
    df['Quality_Size'] = df.get('OverallQual', 0) * df.get('GrLivArea', 0)
    df['LivAreaPerRoom'] = df.get('GrLivArea', 0) / (df.get('TotRmsAbvGrd', 0) + 1e-6)
    df['TotalPorchSF'] = df.get('OpenPorchSF', 0) + df.get('EnclosedPorch', 0)
    df['OverallQual_sq'] = df.get('OverallQual', 0) ** 2
    
    # New features
    df['HouseAge'] = df.get('YrSold', 0) - df.get('YearBuilt', 0)
    df['TotalRooms'] = df.get('TotRmsAbvGrd', 0) + df.get('FullBath', 0)
    df['GarageAreaPerCar'] = df.get('GarageArea', 0) / (df.get('GarageCars', 0) + 1e-6)
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)

# Log transform
for col in ['GrLivArea', 'LotArea', 'TotalSF', 'TotalBsmtSF']:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])


# Use 5-Fold CV instead of a single validation split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for train_idx, val_idx in kf.split(train_df):
    X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
    y_train, y_val = np.log1p(train_target.iloc[train_idx]), np.log1p(train_target.iloc[val_idx])


# 2. Data Prep for NN
# -------------------
# Scale features differently for NN
nn_scaler = QuantileTransformer(output_distribution='normal')
X_train_nn = nn_scaler.fit_transform(X_train)
X_val_nn = nn_scaler.transform(X_val)
X_test_nn = nn_scaler.transform(test_df)


# 3. Neural Network Architecture
# -----------------------------
def build_model(input_shape):
    model = models.Sequential([
        layers.Dense(512, activation='swish', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='swish'),
        layers.BatchNormalization(),
        
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=Huber(delta=1.5),  # Proper implementation
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model

# 4. Train with Callbacks
# -----------------------
model_nn = build_model(X_train_nn.shape[1])


history = model_nn.fit(
    X_train_nn, y_train,
    validation_data=(X_val_nn, y_val),
    epochs=500,
    batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(patience=25),
        callbacks.ReduceLROnPlateau(factor=0.3, patience=12)
    ],
    verbose=1
)

# 5. Ensemble with XGBoost
# ------------------------
# Get NN predictions
nn_train_preds = model_nn.predict(X_train_nn).flatten()
nn_val_preds = model_nn.predict(X_val_nn).flatten()

# Convert scaled arrays back to DataFrames for feature stacking
X_train_df = pd.DataFrame(X_train_nn, columns=train_df.columns)
X_val_df = pd.DataFrame(X_val_nn, columns=train_df.columns)
X_test_df = pd.DataFrame(X_test_nn, columns=train_df.columns)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2000, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10),
        'early_stopping_rounds': 50
    }
    
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train_df,y_train, eval_set=[(X_val_df, y_val)],verbose=False)
    val_preds = model.predict(X_val_df)
    return np.sqrt(mean_squared_error(y_val, val_preds))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_xgb_params = study.best_params

# Retrain XGBoost with NN features
final_model = XGBRegressor(**best_xgb_params, random_state=42)
final_model.fit(X_train_df,y_train, eval_set=[(X_val_df, y_val)],verbose=False)

# 6. Evaluate Hybrid Model
# ------------------------
val_preds = final_model.predict(X_val_df)
val_preds_exp = 0.7 * np.expm1(val_preds) + 0.3 * np.expm1(nn_val_preds)
y_val_exp = np.expm1(y_val)

final_rmse = np.sqrt(mean_squared_error(y_val_exp, val_preds_exp))
final_mae = mean_absolute_error(y_val_exp, val_preds_exp)
final_r2 = r2_score(y_val_exp, val_preds_exp)

print(f"\nHybrid Model Validation Metrics:")
print(f"RMSE: {final_rmse:.2f}")
print(f"MAE: {final_mae:.2f}")
print(f"R²: {final_r2:.4f}")
print(f"Mean Price: ${y_val_exp.mean():,.2f}")
print(f"Median Price: ${np.expm1(y_val).median():,.2f}")

# 7. Generate Final Predictions
# ----------------------------
# Get NN predictions on test set
nn_test_preds = model_nn.predict(X_test_nn).flatten()
test_xgb_features = X_test_df.copy()

xgb_test_preds = final_model.predict(test_xgb_features)
test_preds = 0.2 * np.expm1(xgb_test_preds) + 0.8 * np.expm1(nn_test_preds)

final_preds = test_preds

residuals = y_val_exp - val_preds_exp
correction_factor = np.mean(residuals)
final_preds += correction_factor
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds})
submission.to_csv('hybrid_submission.csv', index=False)