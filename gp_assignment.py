# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('train.csv', encoding='utf-8')
# if df.columns[-1].startswith('Unnamed'):
#     df = df.drop(df.columns[-1], axis=1)


# List of columns to drop
columns_to_drop = ['Id']

# Drop unnecessary columns
df = df.drop(columns=columns_to_drop)

y = df['SalePrice'].copy()
df = df.drop('SalePrice', axis=1)


# Data Preprocessing
# Handle missing values
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns

# Log transform ONLY if needed (prices should be >0)
if (y <= 0).any():
    y = y + 1 - y.min()  # Shift to positive values first
y_log = np.log1p(y)

# Numerical missing values
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)
    
# Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Categorical missing values
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove outliers
df = df[df['GrLivArea'] <= 4000]

# Calculate percentage of missing values for each column
missing_percentage = df.isnull().mean() * 100

# Drop columns with more than 50% missing values
columns_to_drop = missing_percentage[missing_percentage > 50].index
df = df.drop(columns=columns_to_drop)

# Identify high cardinality categorical columns
high_cardinality_cols = [col for col in df.select_dtypes(include='object').columns 
                         if df[col].nunique() > 50]

# Drop high cardinality columns
df = df.drop(columns=high_cardinality_cols)

# Calculate correlation matrix
corr_matrix = df.corr().abs()

# Identify highly correlated features (e.g., correlation > 0.9)
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
columns_to_drop = [col for col in upper_triangle.columns 
                   if any(upper_triangle[col] > 0.9)]

# Drop redundant columns
df = df.drop(columns=columns_to_drop)

# Log transform skewed numerical features
skewed_features = ['GrLivArea', 'LotArea']
for feature in skewed_features:
    df[feature] = np.log1p(df[feature])

# Scale numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Create interaction features
df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']

# Create temporal features
df['TimeSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']

# Remove duplicate rows
df = df.drop_duplicates()

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])



# Ensure the input is a 2D array
neighborhood_data = df[['Neighborhood']]


print("Unique Neighborhoods:", df['Neighborhood'].nunique())

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False)
encoded_neigh = encoder.fit_transform(neighborhood_data)

print("Encoded Data Shape:", encoded_neigh.shape)

# Get feature names for the encoded columns
encoded_neigh_columns = encoder.get_feature_names_out(['Neighborhood'])
encoded_neigh = pd.DataFrame(encoded_neigh, columns=encoded_neigh_columns)

# Add target back
df['SalePrice'] = y
df['SalePrice_log'] = y_log


# Create new features
df['PricePerSqFt'] = df['SalePrice'] / df['GrLivArea']
df['HouseAge'] = df['YrSold'] - df['YearBuilt']


plt.figure(figsize=(12, 6))
sns.histplot(df['SalePrice'], kde=True, bins=50)
plt.title('SalePrice Distribution')
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')
plt.show()

# Data Summary
print("Dataset Shape:", df.shape)
print("\nSummary Statistics:")
print(df[['SalePrice', 'GrLivArea', 'OverallQual']].describe())

# Initial Visualizations
plt.figure(figsize=(12, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('Price vs Living Area')
plt.show()

# Data Analysis
# 1. Key Drivers of House Prices
corr_matrix = df.corr()

plt.figure(figsize=(24, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(24, 16))
sns.heatmap(corr_matrix[['SalePrice']].sort_values(by='SalePrice', ascending=False), 
            annot=True, cmap='coolwarm')
plt.title('Correlation with Sale Price')
plt.show()

# Random Forest Feature Importance
X = df[numerical_cols]
y = df['SalePrice']

rf = RandomForestRegressor()
rf.fit(X, y)

feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp.head(10))
plt.title('Top 10 Important Features')
plt.show()



# 2. Affordability vs Amenities (Clustering)
cluster_features = ['BedroomAbvGr', 'FullBath', 'GarageCars', 'GrLivArea']
X_temp = df[cluster_features]
df[cluster_features] = scaler.fit_transform(X_temp)


# Run K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[cluster_features]) 


cluster_summary = df.groupby('Cluster').agg({
    'SalePrice': 'mean',
    'BedroomAbvGr': 'mean',
    'FullBath': 'mean',
    'GrLivArea': 'mean',
    'Neighborhood': lambda x: x.mode()[0]  # Most common neighborhood
}).round(1)

# Rename columns for clarity
cluster_summary.columns = ['Avg_Price', 'Avg_Bedrooms', 'Avg_Bathrooms', 'Avg_SqFt', 'Top_Neighborhood']
cluster_summary['Avg_Price'] = cluster_summary['Avg_Price'].apply(lambda x: f"${x/1000:.0f}K")

print(cluster_summary.sort_values('Avg_Price'))

dream_houses_updated = df[
    (df['Cluster'] == 0) &  # Focus on Cluster 0 (best value)
    (df['BedroomAbvGr'] == 3) & 
    (df['FullBath'].between(1.5, 2)) &  # Allow 1.5 baths for flexibility
    (df['GrLivArea'].between(1500, 1800)) &  # Slightly smaller to save $
    (df['Neighborhood'].isin(['NAmes', 'CollgCr']))  # Target top neighborhoods
]

print(f"Found {len(dream_houses_updated)} dream houses after adjustments.")

# Melt data for plotting
melted = cluster_summary.reset_index().melt(
    id_vars=['Cluster', 'Top_Neighborhood'],
    value_vars=['Avg_Bedrooms', 'Avg_Bathrooms', 'Avg_SqFt'],
    var_name='Metric'
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=melted,
    x='Cluster',
    y='value',
    hue='Metric',
    palette='coolwarm'
)
plt.title('Cluster Features Comparison')
plt.ylabel('Average Value')
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=dream_houses_updated,
    x='GrLivArea',
    y='SalePrice',
    hue='Neighborhood',
    style='FullBath',  # Show bathroom trade-offs
    size='BedroomAbvGr',  # Show bedroom size
    sizes=(50, 200),
    palette='bright'
)
plt.title('Dream Houses: Size vs. Price Trade-offs')
plt.axhline(y=250000, color='red', linestyle='--', label='Budget Cap')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()

# 3. Location Impact (ANOVA)

neighborhood_prices = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
plt.figure(figsize=(15, 6))
sns.barplot(x=neighborhood_prices.index, y=neighborhood_prices.values)
plt.xticks(rotation=90)
plt.title('Median Prices by Neighborhood')
plt.show()

# Plotting interaction between Neighborhood and HouseAge
plt.figure(figsize=(15, 6))
sns.pointplot(x='Neighborhood', y='SalePrice', hue='HouseAge', data=df, palette='viridis', ci=None)
plt.xticks(rotation=90)
plt.title('SalePrice by Neighborhood and HouseAge')
plt.xlabel('Neighborhood')
plt.ylabel('SalePrice')
plt.legend(title='HouseAge', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

import statsmodels.api as sm
from statsmodels.formula.api import ols

# Fit the ANOVA model
model = ols('SalePrice ~ C(Neighborhood) + HouseAge + C(Neighborhood):HouseAge', data=df).fit()

# Get residuals
residuals = model.resid

# Plot residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.fittedvalues, y=residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Get ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)

# Plot F-statistics
plt.figure(figsize=(10, 6))
sns.barplot(x=anova_table.index, y=anova_table['F'], palette='viridis')
plt.title('F-statistics from ANOVA')
plt.xlabel('Features')
plt.ylabel('F-statistic')
plt.xticks(rotation=45)
plt.show()

# 4. Predictive Modeling
# Prepare data

# Reset indices to ensure alignment
df = df.reset_index(drop=True)
encoded_neigh = encoded_neigh.reset_index(drop=True)

X = pd.concat([df[numerical_cols], encoded_neigh], axis=1)
y = df['SalePrice']


# check that the shape of the data
print("Shape of numerical_cols:", df[numerical_cols].shape)
print("Shape of X:", df[numerical_cols].shape)
print("Shape of encoded_neigh:", encoded_neigh.shape)
print("Shape of y:", y.shape)

# Handle missing values (example: dropping rows with missing values)
#  X = X.dropna()
#  y = y.dropna()


imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = y.fillna(y.mean())

# recheck the shape of the data
print("Shape of X after handling missing values:", X.shape)
print("Shape of y after handling missing values:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLinear Regression Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
print(f"R²: {r2_score(y_test, lr_pred):.2f}")

# XGBoost
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, xgb_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, xgb_pred):.2f}")
print(f"Mean Price: ${np.expm1(y_test).mean():,.2f}")
print(f"Median Price: ${np.expm1(y_test).median():,.2f}")
print(f"R²: {r2_score(y_test, xgb_pred):.2f}")

# 5. Identifying Undervalued Homes
df['PredictedPrice'] = xgb_model.predict(X)
df['Residual'] = df['SalePrice'] - df['PredictedPrice']
undervalued = df[df['Residual'] < -0.15*df['PredictedPrice']]


undervalued_details = undervalued[['Neighborhood', 'HouseAge', 'SalePrice', 'PredictedPrice', 'Residual', 
                                   'LotArea', 'OverallQual', 'GrLivArea', 'BedroomAbvGr', 'FullBath']]

undervalued_details['Discount%'] = (-undervalued_details['Residual'] / undervalued_details['PredictedPrice'] * 100).round(1)

# Sort by how undervalued they are (most undervalued first)
undervalued_details = undervalued_details.sort_values(by='Residual')

# Display the details
print(f"\nUndervalued Houses Found: {len(undervalued_details)}")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(undervalued_details.to_string())


