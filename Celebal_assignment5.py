import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading and exploring data...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Store test IDs for later
test_ids = test_data['Id'].copy()

# Combine train and test for consistent preprocessing
all_data = pd.concat([train_data.drop('SalePrice', axis=1), test_data], ignore_index=True)
target = train_data['SalePrice'].copy()

print(f"Combined data shape: {all_data.shape}")

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================

def advanced_feature_engineering(df):
    """Apply advanced feature engineering techniques"""
    df = df.copy()
    
    print("Applying advanced feature engineering...")
    
    # 1. Handle missing values with domain knowledge
    # LotFrontage: fill with median by neighborhood
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Garage features: fill missing with 0 for no garage
    garage_features = ['GarageYrBlt', 'GarageArea', 'GarageCars']
    for feature in garage_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
    
    # Basement features: fill missing with 0 for no basement
    basement_features = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                        'BsmtFullBath', 'BsmtHalfBath']
    for feature in basement_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
    
    # MasVnrArea: fill with 0 for no masonry veneer
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 2. Create new features based on domain knowledge
    
    # Total area features
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (df['FullBath'] + df['BsmtFullBath'] + 
                           0.5 * (df['HalfBath'] + df['BsmtHalfBath']))
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + 
                         df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
    
    # Age features
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    
    # Quality features
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    df['GarageScore'] = df['GarageArea'] * df['GarageCars']
    
    # Ratios and per-unit features
    df['LivingAreaRatio'] = df['GrLivArea'] / df['TotalSF']
    df['BasementRatio'] = df['TotalBsmtSF'] / df['TotalSF']
    df['PricePerSqft'] = df['GrLivArea'] / (df['LotArea'] + 1)  # Add 1 to avoid division by zero
    
    # Categorical combinations
    df['QualCond'] = df['OverallQual'].astype(str) + '_' + df['OverallCond'].astype(str)
    df['ExterQualCond'] = df['ExterQual'].astype(str) + '_' + df['ExterCond'].astype(str)
    
    # Binning continuous variables
    df['LotAreaBin'] = pd.cut(df['LotArea'], bins=5, labels=['Small', 'Medium-Small', 'Medium', 'Medium-Large', 'Large'])
    df['AgeBin'] = pd.cut(df['HouseAge'], bins=5, labels=['New', 'Recent', 'Middle', 'Old', 'Very Old'])
    
    # Boolean features for special cases
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasPorch'] = (df['TotalPorchSF'] > 0).astype(int)
    df['HasSecondFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    
    # Polynomial features for important variables
    df['GrLivArea_sq'] = df['GrLivArea'] ** 2
    df['TotalSF_sq'] = df['TotalSF'] ** 2
    df['OverallQual_sq'] = df['OverallQual'] ** 2
    
    print(f"Feature engineering complete. New shape: {df.shape}")
    return df

# Apply feature engineering
all_data_engineered = advanced_feature_engineering(all_data)

# =============================================================================
# ADVANCED DATA PREPROCESSING
# =============================================================================

def advanced_preprocessing(df, target=None):
    """Apply advanced preprocessing techniques"""
    df = df.copy()
    
    print("Applying advanced preprocessing...")
    
    # Remove outliers from training data only
    if target is not None:
        # Remove extreme outliers using IQR method
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (target >= lower_bound) & (target <= upper_bound)
        print(f"Removing {(~outlier_mask).sum()} outliers from training data")
        
        df = df[outlier_mask]
        target = target[outlier_mask]
    
    # Identify numeric and categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove Id if present
    if 'Id' in numeric_features:
        numeric_features.remove('Id')
        df = df.drop('Id', axis=1)
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    return df, target, numeric_features, categorical_features

# Apply preprocessing
if target is not None:
    # For training data (first len(train_data) rows)
    train_indices = range(len(train_data))
    train_engineered = all_data_engineered.iloc[train_indices].copy()
    train_processed, target_processed, num_features, cat_features = advanced_preprocessing(
        train_engineered, target
    )
    
    # For test data
    test_engineered = all_data_engineered.iloc[len(train_data):].copy()
    test_processed = test_engineered[train_processed.columns]
else:
    all_processed, _, num_features, cat_features = advanced_preprocessing(all_data_engineered)

print(f"Final training data shape: {train_processed.shape}")
print(f"Final test data shape: {test_processed.shape}")

# =============================================================================
# ADVANCED PREPROCESSING PIPELINE
# =============================================================================

# Create preprocessing pipeline with advanced techniques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),  # More robust to outliers than StandardScaler
    ('power', PowerTransformer(method='yeo-johnson'))  # Normalize distributions
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', pd.get_dummies)  # We'll handle this manually for consistency
])

# Manual categorical encoding for better control
def encode_categorical_features(df_train, df_test, cat_features):
    """Encode categorical features consistently across train and test"""
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    
    for feature in cat_features:
        if feature in df_train.columns:
            # Convert to string and handle missing values
            train_values = df_train[feature].astype(str).replace('nan', 'missing')
            test_values = df_test[feature].astype(str).replace('nan', 'missing')
            
            # Get all unique values from both train and test
            all_values = set(train_values.unique()) | set(test_values.unique())
            
            # Create dummy variables
            for value in all_values:
                col_name = f"{feature}_{value}"
                df_train_encoded[col_name] = (train_values == value).astype(int)
                df_test_encoded[col_name] = (test_values == value).astype(int)
            
            # Drop original categorical column
            df_train_encoded = df_train_encoded.drop(feature, axis=1)
            df_test_encoded = df_test_encoded.drop(feature, axis=1)
    
    return df_train_encoded, df_test_encoded

# Apply categorical encoding
X_train_encoded, X_test_encoded = encode_categorical_features(
    train_processed, test_processed, cat_features
)

# Apply numeric transformations
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('power', PowerTransformer(method='yeo-johnson', standardize=True))
])

# Fit and transform numeric features
numeric_features_final = [col for col in X_train_encoded.columns if col in num_features]
X_train_numeric = numeric_pipeline.fit_transform(X_train_encoded[numeric_features_final])
X_test_numeric = numeric_pipeline.transform(X_test_encoded[numeric_features_final])

# Get categorical features
categorical_features_final = [col for col in X_train_encoded.columns if col not in num_features]
X_train_categorical = X_train_encoded[categorical_features_final].values
X_test_categorical = X_test_encoded[categorical_features_final].values

# Combine features
X_train_final = np.hstack([X_train_numeric, X_train_categorical])
X_test_final = np.hstack([X_test_numeric, X_test_categorical])

# Create feature names
feature_names_final = numeric_features_final + categorical_features_final

print(f"Final feature matrix shape: {X_train_final.shape}")
print(f"Total features: {len(feature_names_final)}")

# Apply log transformation to target for better distribution
y_train_log = np.log1p(target_processed)

# =============================================================================
# TRAIN-VALIDATION SPLIT
# =============================================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_final, y_train_log, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# =============================================================================
# HYPERPARAMETER TUNING AND MODEL TRAINING
# =============================================================================

print("\n" + "="*60)
print("HYPERPARAMETER TUNING AND MODEL TRAINING")
print("="*60)

# Define models with hyperparameter grids
models_and_params = {
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        }
    },
    'Lasso': {
        'model': Lasso(random_state=42, max_iter=2000),
        'params': {
            'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        }
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=42, max_iter=2000),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5]
        }
    }
}

# Store results
tuned_models = {}
results = {}

for name, model_info in models_and_params.items():
    print(f"\nTuning {name}...")
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    tuned_models[name] = best_model
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    # Calculate metrics (convert back from log scale)
    y_train_actual = np.expm1(y_train)
    y_val_actual = np.expm1(y_val)
    y_train_pred_actual = np.expm1(y_train_pred)
    y_val_pred_actual = np.expm1(y_val_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
    val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred_actual))
    train_r2 = r2_score(y_train_actual, y_train_pred_actual)
    val_r2 = r2_score(y_val_actual, y_val_pred_actual)
    train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
    val_mae = mean_absolute_error(y_val_actual, y_val_pred_actual)
    
    # Store results
    results[name] = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'best_cv_score': -grid_search.best_score_,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_mae': train_mae,
        'val_mae': val_mae
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: ${np.sqrt(-grid_search.best_score_):,.2f}")
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Validation RMSE: ${val_rmse:,.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Validation R²: {val_r2:.4f}")

# =============================================================================
# MODEL COMPARISON AND RESULTS
# =============================================================================

print("\n" + "="*60)
print("TUNED MODEL COMPARISON")
print("="*60)

print(f"{'Model':<20} {'Best Params':<30} {'Val RMSE':<15} {'Val R²':<12}")
print("-" * 77)

for name, metrics in results.items():
    params_str = str(metrics['best_params'])[:25] + "..." if len(str(metrics['best_params'])) > 25 else str(metrics['best_params'])
    print(f"{name:<20} {params_str:<30} ${metrics['val_rmse']:<14,.0f} {metrics['val_r2']:<11.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
best_model = results[best_model_name]['model']

print(f"\nBest model: {best_model_name}")
print(f"Best validation R²: {results[best_model_name]['val_r2']:.4f}")
print(f"Best validation RMSE: ${results[best_model_name]['val_rmse']:,.2f}")

# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

print("\n" + "="*60)
print("CREATING ENSEMBLE MODEL")
print("="*60)

# Create ensemble predictions (weighted average of top 3 models)
top_models = sorted(results.keys(), key=lambda x: results[x]['val_r2'], reverse=True)[:3]
print(f"Top 3 models for ensemble: {top_models}")

# Ensemble weights based on validation performance
ensemble_weights = {}
total_r2 = sum(results[model]['val_r2'] for model in top_models)
for model in top_models:
    ensemble_weights[model] = results[model]['val_r2'] / total_r2

print("Ensemble weights:")
for model, weight in ensemble_weights.items():
    print(f"  {model}: {weight:.3f}")

# Create ensemble predictions
ensemble_val_pred = np.zeros(len(y_val))
for model_name in top_models:
    model = results[model_name]['model']
    pred = model.predict(X_val)
    ensemble_val_pred += ensemble_weights[model_name] * pred

# Calculate ensemble metrics
ensemble_val_pred_actual = np.expm1(ensemble_val_pred)
y_val_actual = np.expm1(y_val)

ensemble_rmse = np.sqrt(mean_squared_error(y_val_actual, ensemble_val_pred_actual))
ensemble_r2 = r2_score(y_val_actual, ensemble_val_pred_actual)
ensemble_mae = mean_absolute_error(y_val_actual, ensemble_val_pred_actual)

print(f"\nEnsemble Model Performance:")
print(f"Validation RMSE: ${ensemble_rmse:,.2f}")
print(f"Validation R²: {ensemble_r2:.4f}")
print(f"Validation MAE: ${ensemble_mae:,.2f}")

# =============================================================================
# TEST SET PREDICTIONS
# =============================================================================

print("\n" + "="*60)
print("GENERATING TEST SET PREDICTIONS")
print("="*60)

test_predictions = {}

# Individual model predictions
for name, model_info in results.items():
    model = model_info['model']
    test_pred_log = model.predict(X_test_final)
    test_pred = np.expm1(test_pred_log)
    test_predictions[name] = test_pred
    
    print(f"\n{name} Test Predictions:")
    print(f"Mean: ${test_pred.mean():,.2f}")
    print(f"Median: ${np.median(test_pred):,.2f}")
    print(f"Min: ${test_pred.min():,.2f}")
    print(f"Max: ${test_pred.max():,.2f}")

# Ensemble test predictions
ensemble_test_pred = np.zeros(len(X_test_final))
for model_name in top_models:
    model = results[model_name]['model']
    pred_log = model.predict(X_test_final)
    ensemble_test_pred += ensemble_weights[model_name] * pred_log

ensemble_test_pred_actual = np.expm1(ensemble_test_pred)
test_predictions['Ensemble'] = ensemble_test_pred_actual

print(f"\nEnsemble Test Predictions:")
print(f"Mean: ${ensemble_test_pred_actual.mean():,.2f}")
print(f"Median: ${np.median(ensemble_test_pred_actual):,.2f}")
print(f"Min: ${ensemble_test_pred_actual.min():,.2f}")
print(f"Max: ${ensemble_test_pred_actual.max():,.2f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save individual model predictions
for name, predictions in test_predictions.items():
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    filename = f"enhanced_predictions_{name.lower().replace(' ', '_')}.csv"
    submission.to_csv(filename, index=False)
    print(f"Saved {name} predictions to {filename}")

# Save enhanced performance summary
performance_data = []
for name, metrics in results.items():
    performance_data.append({
        'Model': name,
        'Best_Params': str(metrics['best_params']),
        'CV_RMSE': metrics['best_cv_score'],
        'Training_RMSE': metrics['train_rmse'],
        'Validation_RMSE': metrics['val_rmse'],
        'Training_R2': metrics['train_r2'],
        'Validation_R2': metrics['val_r2'],
        'Training_MAE': metrics['train_mae'],
        'Validation_MAE': metrics['val_mae']
    })

# Add ensemble results
performance_data.append({
    'Model': 'Ensemble',
    'Best_Params': f"Weighted average of top 3: {top_models}",
    'CV_RMSE': np.nan,
    'Training_RMSE': np.nan,
    'Validation_RMSE': ensemble_rmse,
    'Training_R2': np.nan,
    'Validation_R2': ensemble_r2,
    'Training_MAE': np.nan,
    'Validation_MAE': ensemble_mae
})

enhanced_performance_df = pd.DataFrame(performance_data)
enhanced_performance_df.to_csv('enhanced_model_performance.csv', index=False)
print("Saved enhanced performance summary to enhanced_model_performance.csv")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Analyze feature importance for tree-based models
for name in ['RandomForest', 'GradientBoosting']:
    if name in results:
        model = results[name]['model']
        importance = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_final,
            'importance': importance
        }).sort_values('importance', ascending=False).head(15)
        
        print(f"\nTop 15 most important features for {name}:")
        print(feature_importance_df.to_string(index=False))

# Analyze coefficients for linear models
for name in ['Ridge', 'Lasso', 'ElasticNet']:
    if name in results:
        model = results[name]['model']
        coefficients = model.coef_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_final,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False).head(15)
        
        print(f"\nTop 15 most important features for {name}:")
        print(feature_importance_df[['feature', 'coefficient']].to_string(index=False))

print("\n" + "="*60)
print("ENHANCED ANALYSIS COMPLETE!")
print("="*60)
print(f"Best individual model: {best_model_name}")
print(f"Best individual R² score: {results[best_model_name]['val_r2']:.4f}")
print(f"Ensemble R² score: {ensemble_r2:.4f}")
print("All enhanced predictions and performance metrics have been saved.")
