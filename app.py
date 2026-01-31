import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import re
import warnings

# [FIX] Import pickle and dill for robust Streamlit serialization
import pickle
try:
    import dill
except ImportError:
    dill = None

# Profiling
from ydata_profiling import ProfileReport


# Scikit-Learn & Modeling

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit

from sklearn.impute import SimpleImputer, KNNImputer

# [FIX] Added MaxAbsScaler for sparse data scaling

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, PolynomialFeatures, FunctionTransformer, label_binarize

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA, TruncatedSVD



# Metrics

from sklearn.metrics import (

    accuracy_score, mean_squared_error, r2_score, mean_absolute_error, 

    classification_report, confusion_matrix, f1_score, precision_score, 

    recall_score, roc_curve, auc, mean_squared_log_error

)



# Imbalanced Learn

try:

    from imblearn.over_sampling import SMOTE

    HAS_IMBLEARN = True

except ImportError:

    HAS_IMBLEARN = False





# Models

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier, SGDRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor, AdaBoostClassifier, AdaBoostRegressor

from sklearn.svm import SVC, SVR

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from xgboost import XGBClassifier, XGBRegressor





# Extra Models

try:

    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LGBM = True

except ImportError: HAS_LGBM = False





try:

    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True

    

    # --- FIX: Define wrapper classes globally here ---

    class SafeCatBoostClassifier(CatBoostClassifier, BaseEstimator): 

        def __init__(self, **kwargs):

            super().__init__(**kwargs)

            

    class SafeCatBoostRegressor(CatBoostRegressor, BaseEstimator): 

        def __init__(self, **kwargs):

            super().__init__(**kwargs)

            

except ImportError: HAS_CATBOOST = False





# --- 1. APP CONFIGURATION ---

st.set_page_config(page_title="NoCodeML Studio", layout="wide", page_icon="📈")

plt.style.use('default')

warnings.filterwarnings('ignore') # Silence sklearn convergence warnings in UI



# Initialize Session State

if 'df' not in st.session_state: st.session_state.df = None

if 'df_raw' not in st.session_state: st.session_state.df_raw = None 

if 'loaded_file_name' not in st.session_state: st.session_state.loaded_file_name = None

if 'model_results' not in st.session_state: st.session_state.model_results = {}

if 'le' not in st.session_state: st.session_state.le = None

if 'preprocessor' not in st.session_state: st.session_state.preprocessor = None

if 'best_model' not in st.session_state: st.session_state.best_model = None

if 'feature_names' not in st.session_state: st.session_state.feature_names = []

if 'pipeline_config' not in st.session_state: st.session_state.pipeline_config = {}

if 'custom_code_content' not in st.session_state: st.session_state.custom_code_content = ""

if 'inference_data' not in st.session_state: st.session_state.inference_data = None

if 'best_params' not in st.session_state: st.session_state.best_params = {}

if 'raw_feature_names' not in st.session_state: st.session_state.raw_feature_names = []



# --- 2. CUSTOM TRANSFORMERS ---

class CustomScriptTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, script_code=None):
        self.script_code = script_code
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if not self.script_code:
            return X
        # Avoid modifying original dataframe if passed by reference
        df = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X).copy()
        
        # Context for execution
        local_vars = {
            'df': df, 
            'pd': pd, 
            'np': np, 
            're': re
        }
        
        try:
            # Execute the user's script
            # The script is expected to modify 'df' in place or reassign it
            exec(self.script_code, globals(), local_vars)
            
            # Retrieve the modified dataframe
            if 'df' in local_vars and isinstance(local_vars['df'], pd.DataFrame):
                return local_vars['df']
            return df
        except Exception as e:
            # In a pipeline, we might want to log this or raise, 
            # but for UI stability we print and return original
            print(f"Custom Script Execution Failed: {e}")
            return X

class MaskingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mask_cols=None, mask_mode='Single Value', mask_val=None, mask_min=None, mask_max=None):
        self.mask_cols = mask_cols
        self.mask_mode = mask_mode
        self.mask_val = mask_val
        self.mask_min = mask_min
        self.mask_max = mask_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not self.mask_cols: return X
        
        for col in self.mask_cols:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                if self.mask_mode == "Single Value" and self.mask_val is not None:
                    try: 
                        val_to_mask = float(self.mask_val)
                    except: 
                        val_to_mask = self.mask_val
                    X[col] = X[col].replace(val_to_mask, np.nan)
                
                elif self.mask_mode == "Range (Min-Max)" and self.mask_min is not None and self.mask_max is not None:
                    condition = (X[col] >= self.mask_min) & (X[col] <= self.mask_max)
                    X.loc[condition, col] = np.nan
        return X

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_cols=None, factor=1.5):
        self.outlier_cols = outlier_cols
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        # Learn the Q1/Q3 bounds from the Training data
        if self.outlier_cols:
            for col in self.outlier_cols:
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    # Calculate bounds
                    lower = Q1 - (self.factor * IQR)
                    upper = Q3 + (self.factor * IQR)
                    self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

class RandomSampleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        self.saved_values = {}
        self.feature_names_in_ = None
        self.random_state = random_state

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else [f"col_{i}" for i in range(X.shape[1])]
        for col in X.columns:
            vals = X[col].dropna().values
            if len(vals) > 0:
                self.saved_values[col] = vals
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        rng = np.random.default_rng(self.random_state)
        for col in X.columns:
            if col in self.saved_values:
                mask = X[col].isna()
                if mask.any():
                    if len(self.saved_values[col]) > 0:
                        fill_values = rng.choice(self.saved_values[col], size=mask.sum())
                        X.loc[mask, col] = fill_values
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.feature_names_in_

class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, binning_config=None):
        self.binning_config = binning_config if binning_config else {}
        self.bins_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist()
        for col, config in self.binning_config.items():
            if col in X.columns:
                try:
                    if config['method'] == 'Automatic (Quantile)':
                        try:
                            _, edges = pd.qcut(X[col], q=config['params'], retbins=True, duplicates='drop')
                        except ValueError:
                            _, edges = pd.cut(X[col], bins=config['params'], retbins=True, duplicates='drop')
                        self.bins_[col] = edges
                    elif config['method'] == 'Manual Ranges':
                        self.bins_[col] = sorted(config['params'])
                except: pass
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, edges in self.bins_.items():
            if col in X.columns and col in self.bins_:
                binned_col_name = f"{col}_binned"
                try:
                    binned_series = pd.cut(X[col], bins=edges, labels=False, include_lowest=True)
                    if hasattr(binned_series, 'cat'):
                        X[binned_col_name] = (binned_series.cat.codes + 1).astype(int)
                    else:
                        X[binned_col_name] = (binned_series.fillna(-1) + 1).astype(int)
                    X = X.drop(columns=[col])
                except: pass
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: input_features = self.feature_names_in_
        out_names = []
        for col in input_features:
            if col in self.binning_config:
                out_names.append(f"{col}_binned")
            else:
                out_names.append(col)
        return out_names

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = [str(c) for c in X.columns] if hasattr(X, "columns") else None
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            try:
                series = pd.to_datetime(X[col], errors='coerce', infer_datetime_format=True)
                X[f"{col}_year"] = series.dt.year
                X[f"{col}_month"] = series.dt.month
                X[f"{col}_day"] = series.dt.day
                X[f"{col}_dow"] = series.dt.dayofweek
                X = X.drop(columns=[col])
            except: pass 
        return X

    def get_feature_names_out(self, input_features=None):
        out_names = []
        if input_features is None: input_features = self.feature_names_in_
        for col in input_features:
            out_names.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dow"])
        return out_names
    

# --- 3. HELPER FUNCTIONS ---

def evaluate_model_performance(model, X_test, y_test, task_type, le=None):

    try:

        preds = model.predict(X_test)

    except Exception as e:

        st.error(f"Prediction failed: {e}")

        return



    if task_type == 'Classification':

        acc = accuracy_score(y_test, preds)

        f1 = f1_score(y_test, preds, average='weighted')

        prec = precision_score(y_test, preds, average='weighted', zero_division=0)

        rec = recall_score(y_test, preds, average='weighted')

        

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{acc:.4f}")

        c2.metric("F1 Score", f"{f1:.4f}")

        c3.metric("Precision", f"{prec:.4f}")

        c4.metric("Recall", f"{rec:.4f}")

        

        # [USED HERE] Classification Report

        with st.expander("📄 View Detailed Classification Report"):

            try:

                class_names = [str(c) for c in le.classes_] if le else None

                report = classification_report(y_test, preds, target_names=class_names)

                st.code(report, language='text')

            except Exception as e:

                st.info(f"Report unavailable: {e}")



        st.divider()

        t1, t2 = st.tabs(["Confusion Matrix", "ROC Curve"])

        

        with t1:

            fig, ax = plt.subplots()

            if le:

                all_labels = range(len(le.classes_))

                names = [str(c) for c in le.classes_]

                sns.heatmap(confusion_matrix(y_test, preds, labels=all_labels), 

                            annot=True, fmt='d', xticklabels=names, yticklabels=names, cmap='Blues', ax=ax)

            else:

                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=ax)

            st.pyplot(fig)

            plt.close(fig)

            

        with t2:

            if hasattr(model, "predict_proba"):

                try:

                    probas = model.predict_proba(X_test)

                    n_classes = len(np.unique(y_test))

                    if n_classes > 2 or probas.shape[1] > 2:

                        fig, ax = plt.subplots()

                        y_bin = label_binarize(y_test, classes=sorted(np.unique(y_test)))

                        for i in range(y_bin.shape[1]):

                            fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, i])

                            ax.plot(fpr, tpr, label=f'Class {i}')

                        ax.plot([0, 1], [0, 1], 'k--')

                        ax.set_xlabel('False Positive Rate')

                        ax.set_ylabel('True Positive Rate')

                        ax.set_title('Multiclass ROC')

                        ax.legend()

                        st.pyplot(fig)

                        plt.close(fig)

                    else:

                        fig, ax = plt.subplots()

                        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])

                        roc_auc = auc(fpr, tpr)

                        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')

                        ax.plot([0, 1], [0, 1], 'k--')

                        ax.set_xlabel('False Positive Rate')

                        ax.set_ylabel('True Positive Rate')

                        ax.legend()

                        st.pyplot(fig)

                        plt.close(fig)

                except Exception as e:

                    st.info(f"ROC not available: {e}")

            else:

                st.info("Model does not support probabilities.")



    else: # Regression

        mse = mean_squared_error(y_test, preds)

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(y_test, preds)

        r2 = r2_score(y_test, preds)

        

        # [USED HERE] Mean Squared Log Error

        try:

            msle = mean_squared_log_error(y_test, preds)

            msle_str = f"{msle:.4f}"

        except:

            msle_str = "N/A (Neg)"



        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("R2 Score", f"{r2:.4f}")

        c2.metric("RMSE", f"{rmse:.4f}")

        c3.metric("MSE", f"{mse:.4f}")

        c4.metric("MAE", f"{mae:.4f}")

        c5.metric("MSLE", msle_str)

        

        st.divider()

        t1, t2 = st.tabs(["Actual vs Predicted", "Residuals"])

        

        with t1:

            fig, ax = plt.subplots()

            ax.scatter(y_test, preds, alpha=0.5)

            min_val = min(np.min(y_test), np.min(preds))

            max_val = max(np.max(y_test), np.max(preds))

            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel("Actual")

            ax.set_ylabel("Predicted")

            st.pyplot(fig)

            plt.close(fig)

            

        with t2:

            residuals = y_test - preds

            fig, ax = plt.subplots()

            sns.histplot(residuals, kde=True, ax=ax)

            ax.set_title("Residual Distribution")

            ax.set_xlabel("Error")

            st.pyplot(fig)

            plt.close(fig)



def get_available_models(task_type):

    models = {}

    if task_type == 'Classification':

        models.update({

            'Logistic Regression': LogisticRegression(max_iter=5000), 

            'Random Forest': RandomForestClassifier(),

            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 

            'SVM': SVC(probability=True),

            'KNN': KNeighborsClassifier(), 

            'Naive Bayes (Gaussian)': GaussianNB(),

            'Naive Bayes (Multinomial)': MultinomialNB(),

            'AdaBoost': AdaBoostClassifier(),

            'Gradient Boosting': GradientBoostingClassifier(),

            'SGD Classifier': SGDClassifier(loss='log_loss')

        })

        if HAS_LGBM: models['LightGBM'] = LGBMClassifier()

        if HAS_CATBOOST:

            # FIX: Use the global class

            models['CatBoost'] = SafeCatBoostClassifier(verbose=0)

            

    else: # Regression

        models.update({

            'Linear Regression': LinearRegression(), 

            'Random Forest': RandomForestRegressor(),

            'XGBoost': XGBRegressor(), 

            'SVR': SVR(), 

            'KNN': KNeighborsRegressor(),

            'Ridge': Ridge(max_iter=5000), 

            'Lasso': Lasso(max_iter=5000), 

            'ElasticNet': ElasticNet(max_iter=5000),

            'AdaBoost': AdaBoostRegressor(),

            'Gradient Boosting': GradientBoostingRegressor(),

            'SGD Regressor': SGDRegressor()

        })

        if HAS_LGBM: models['LightGBM'] = LGBMRegressor()

        if HAS_CATBOOST:

            # FIX: Use the global class

            models['CatBoost'] = SafeCatBoostRegressor(verbose=0)

    

    return models



def preprocess_and_split(df_input, target_col, task_type, is_ts, date_col_sort, 
                         mask_mode, mask_val, mask_min, mask_max, mask_cols, 
                         use_outlier_clip, outlier_cols, outlier_factor,       
                         imp_num_mean, imp_num_median, imp_num_knn, imp_num_random, 
                         imp_cat_mode, imp_cat_const, knn_k,
                         test_size, drop_cols,
                         cols_standard, cols_minmax, cols_robust, cols_maxabs,
                         cols_onehot, cols_ordinal,
                         cols_log, binning_config, cols_date, 
                         use_pca, pca_method, pca_components,
                         use_poly, poly_degree,
                         use_smote, custom_code): 
    
    df = df_input.copy()

    # --- 1. SPLIT (Done first to prevent data leakage in Outlier calculation) ---
    if drop_cols: 
        safe_drop_cols = [c for c in drop_cols if c in df.columns and c != target_col]
        df = df.drop(columns=safe_drop_cols)
    
    df = df.dropna(subset=[target_col])
    if df.empty:
        st.error("Data empty after dropping missing targets.")
        return None, None, None, None, None, None, None, None, None, None, None

    if is_ts and date_col_sort:
        df[date_col_sort] = pd.to_datetime(df[date_col_sort])
        df = df.sort_values(by=date_col_sort)
        X = df.drop(columns=[target_col, date_col_sort])
    else:
        X = df.drop(columns=[target_col])
    y = df[target_col]
    
    raw_feature_names = X.columns.tolist()

    le = None
    if task_type == 'Classification':
        le = LabelEncoder()
        try:
            y = le.fit_transform(y)
            if len(le.classes_) > 50:
                st.warning(f"Target has {len(le.classes_)} classes. Classification might be difficult.")
        except Exception as e:
            st.error(f"Target encoding failed: {e}")
            return None, None, None, None, None, None, None, None, None, None, None
    else:
        y = pd.to_numeric(y, errors='coerce')
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        if len(y) == 0:
            st.error("No valid numeric targets found.")
            return None, None, None, None, None, None, None, None, None, None, None

    if is_ts:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # --- 2. PIPELINE CONSTRUCTION ---
    
    # Step A: Custom Script
    custom_fe_step = CustomScriptTransformer(script_code=custom_code)

    # Step B: Masking
    masking_step = MaskingTransformer(
        mask_cols=mask_cols, mask_mode=mask_mode, 
        mask_val=mask_val, mask_min=mask_min, mask_max=mask_max
    )

    # Step C: Outlier Clipping
    # Only pass columns if the checkbox is checked
    clip_cols = outlier_cols if use_outlier_clip else []
    clipping_step = OutlierClipper(outlier_cols=clip_cols, factor=outlier_factor)
    
    consumed_cols = set()
    def get_unique_cols(cols):
        valid = [c for c in cols if c not in consumed_cols and c in X.columns]
        consumed_cols.update(valid)
        return valid
    
    # Step D: Imputation
    impute_transformers = []
    if imp_num_mean: impute_transformers.append(('imp_mean', SimpleImputer(strategy='mean'), get_unique_cols(imp_num_mean)))
    if imp_num_median: impute_transformers.append(('imp_median', SimpleImputer(strategy='median'), get_unique_cols(imp_num_median)))
    if imp_num_knn: impute_transformers.append(('imp_knn', KNNImputer(n_neighbors=knn_k), get_unique_cols(imp_num_knn)))
    if imp_num_random: impute_transformers.append(('imp_random', RandomSampleImputer(random_state=42), get_unique_cols(imp_num_random)))
    if imp_cat_mode: impute_transformers.append(('imp_mode', SimpleImputer(strategy='most_frequent'), get_unique_cols(imp_cat_mode)))
    if imp_cat_const: impute_transformers.append(('imp_const', SimpleImputer(strategy='constant', fill_value='Missing'), get_unique_cols(imp_cat_const)))

    remaining_cols = [c for c in X.columns if c not in consumed_cols]
    if remaining_cols:
        rem_num = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(X[c])]
        rem_cat = [c for c in remaining_cols if c not in rem_num]
        if rem_num: impute_transformers.append(('imp_default_num', SimpleImputer(strategy='mean'), rem_num))
        if rem_cat: impute_transformers.append(('imp_default_cat', SimpleImputer(strategy='most_frequent'), rem_cat))

    if not impute_transformers:
        imputer_step = ColumnTransformer([('passthrough', 'passthrough', X.columns)], verbose_feature_names_out=False)
    else:
        imputer_step = ColumnTransformer(impute_transformers, verbose_feature_names_out=False)
    imputer_step.set_output(transform="pandas")

    binning_step = BinningTransformer(binning_config)

    # Step E: Scaling/Encoding
    transformers = []
    def resolve_targets(original_list):
        targets = []
        for col in original_list:
            if col in binning_config: targets.append(f"{col}_binned")
            else: targets.append(col)
        return targets

    if cols_date: transformers.append(('date_eng', DateFeatureGenerator(), cols_date))
    if cols_log: transformers.append(('log', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one'), cols_log))
    
    if cols_standard: transformers.append(('std', StandardScaler(with_mean=False), resolve_targets(cols_standard)))
    if cols_minmax: transformers.append(('minmax', MinMaxScaler(), resolve_targets(cols_minmax)))
    if cols_robust: transformers.append(('rob', RobustScaler(with_centering=False), resolve_targets(cols_robust)))
    if cols_maxabs: transformers.append(('maxabs', MaxAbsScaler(), resolve_targets(cols_maxabs)))
    
    final_onehot = resolve_targets(cols_onehot)
    # Force sparse_output=False to ensure Pandas DataFrame compatibility
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20) 
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    if final_onehot: transformers.append(('ohe', ohe, final_onehot))
    
    final_ordinal = resolve_targets(cols_ordinal)
    if final_ordinal: transformers.append(('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), final_ordinal))

    if not transformers:
        mapper_step = ColumnTransformer([('all', 'passthrough', slice(None))], verbose_feature_names_out=False)
    else:
        mapper_step = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
    mapper_step.set_output(transform="pandas")
    
    # --- PIPELINE STEPS ---
    steps = [
        ('custom_fe', custom_fe_step), 
        ('masking', masking_step),      
        ('clipping', clipping_step),    
        ('imputer', imputer_step),
        ('binning', binning_step),
        ('mapper', mapper_step)
    ]
    
    if use_poly:
        steps.append(('poly', PolynomialFeatures(degree=poly_degree, interaction_only=True, include_bias=False)))
    
    if use_pca:
        n_comp = pca_components
        if isinstance(n_comp, int):
            n_comp = min(n_comp, X_train.shape[0], X_train.shape[1])
        
        if "TruncatedSVD" in pca_method:
            steps.append(('pca', TruncatedSVD(n_components=n_comp)))
        else:
            steps.append(('pca', PCA(n_components=n_comp)))

    main_pipeline = Pipeline(steps)

    try:
        X_train_final = main_pipeline.fit_transform(X_train, y_train)
        X_test_final = main_pipeline.transform(X_test)
        
        # --- FEATURE NAME EXTRACTION ---
        processed_feature_names = []
        if use_pca:
            processed_feature_names = [f"PCA_{i}" for i in range(X_train_final.shape[1])]
        else:
            try:
                names_from_mapper = mapper_step.get_feature_names_out()
                if use_poly:
                    poly_step = main_pipeline.named_steps['poly']
                    processed_feature_names = poly_step.get_feature_names_out(names_from_mapper)
                else:
                    processed_feature_names = names_from_mapper
            except:
                if hasattr(X_train_final, 'columns'):
                    processed_feature_names = X_train_final.columns.tolist()
                else:
                    processed_feature_names = [f"Feat_{i}" for i in range(X_train_final.shape[1])]
                    
    except Exception as e:
        st.error(f"Pipeline construction failed: {e}")
        return None, None, None, None, None, None, None, None, None, None, None

    if use_smote and task_type == 'Classification' and not is_ts:
        if HAS_IMBLEARN:
            try:
                min_samples = pd.Series(y_train).value_counts().min()
                k = min(5, min_samples - 1)
                if k > 0:
                    smote = SMOTE(random_state=42, k_neighbors=k)
                    X_train_final, y_train = smote.fit_resample(X_train_final, y_train)
            except Exception as e:
                st.warning(f"SMOTE skipped: {e}")

    return X_train_final, X_test_final, y_train, y_test, main_pipeline, le, None, None, list(processed_feature_names), None, raw_feature_names

# --- CODE GENERATION HELPER ---

def generate_python_code(config, model_name, best_params):
    # (Keep parameter cleaning logic...)
    clean_params = best_params.copy()
    if 'penalty' in clean_params and clean_params['penalty'] != 'elasticnet':
        clean_params.pop('l1_ratio', None)
    
    if "SafeCatBoost" in model_name: model_name = model_name.replace("Safe", "") 
    
    model_import = ""
    # (Keep model import logic...)
    if "LogisticRegression" in model_name: model_import = "from sklearn.linear_model import LogisticRegression"
    elif "RandomForest" in model_name: model_import = f"from sklearn.ensemble import {model_name}"
    elif "XGB" in model_name: model_import = f"from xgboost import {model_name}"
    elif "LGBM" in model_name: model_import = f"from lightgbm import {model_name}"
    elif "CatBoost" in model_name: model_import = f"from catboost import {model_name}"
    elif "SVC" in model_name or "SVR" in model_name: model_import = f"from sklearn.svm import {model_name}"
    elif "KNeighbors" in model_name: model_import = f"from sklearn.neighbors import {model_name}"
    elif "LinearRegression" in model_name: model_import = "from sklearn.linear_model import LinearRegression"
    elif "Ridge" in model_name: model_import = "from sklearn.linear_model import Ridge"
    elif "Lasso" in model_name: model_import = "from sklearn.linear_model import Lasso"
    elif "ElasticNet" in model_name: model_import = "from sklearn.linear_model import ElasticNet"
    elif "AdaBoost" in model_name: model_import = f"from sklearn.ensemble import {model_name}"
    elif "GradientBoosting" in model_name: model_import = f"from sklearn.ensemble import {model_name}"
    elif "Voting" in model_name or "Stacking" in model_name:
        model_import = f"from sklearn.ensemble import {model_name}"
    
    if "Naive Bayes" in model_name or "GaussianNB" in model_name or "MultinomialNB" in model_name:
        model_import = "from sklearn.naive_bayes import GaussianNB, MultinomialNB"

    bin_config_str = str(config.get('binning_config', {}))
    custom_code_str = config.get('custom_code', '').replace('"""', "'''")

    # Config extraction
    mask_mode = config.get('mask_mode', 'Single Value')
    mask_min = config.get('mask_min')
    mask_max = config.get('mask_max')
    mask_cols = config.get('mask_cols', [])
    mask_val = config.get('mask_val')
    use_clip = config.get('use_outlier_clip', False)
    outlier_cols = config.get('outlier_cols', [])
    outlier_factor = config.get('outlier_factor', 1.5)
    
    smote_import = "from imblearn.pipeline import Pipeline as ImbPipeline\nfrom imblearn.over_sampling import SMOTE" if config.get('use_smote') else ""
    pipeline_class = "ImbPipeline" if config.get('use_smote') else "Pipeline"

    target_map_code = ""
    if st.session_state.le:
        mapping = dict(zip(range(len(st.session_state.le.classes_)), [str(c) for c in st.session_state.le.classes_]))
        target_map_code = f"TARGET_MAPPING = {mapping}\n# Decode: lambda x: TARGET_MAPPING.get(x, x)"

    if model_name in ['VotingClassifier', 'VotingRegressor', 'StackingClassifier', 'StackingRegressor']:
        model_instantiation = f"model = {model_name}(estimators=[]) # Needs manual estimator definition"
    else:
        model_instantiation = f"best_params = {clean_params}\nmodel = {model_name}(**best_params)"

    # --- BLOCK 1: IMPORTS & CLASSES (f-string allowed here) ---
    code = f"""import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
{model_import}
{smote_import}

# --- CUSTOM TRANSFORMERS ---
class CustomScriptTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, script_code=None):
        self.script_code = script_code
    def fit(self, X, y=None): return self
    def transform(self, X):
        if not self.script_code: return X
        df = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X).copy()
        local_vars = {{'df': df, 'pd': pd, 'np': np, 're': re}}
        try:
            exec(self.script_code, globals(), local_vars)
            if 'df' in local_vars: return local_vars['df']
            return df
        except: return X

class MaskingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mask_cols=None, mask_mode='Single Value', mask_val=None, mask_min=None, mask_max=None):
        self.mask_cols = mask_cols
        self.mask_mode = mask_mode
        self.mask_val = mask_val
        self.mask_min = mask_min
        self.mask_max = mask_max
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if not self.mask_cols: return X
        for col in self.mask_cols:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                if self.mask_mode == "Single Value" and self.mask_val is not None:
                    try: val_to_mask = float(self.mask_val)
                    except: val_to_mask = self.mask_val
                    X[col] = X[col].replace(val_to_mask, np.nan)
                elif self.mask_mode == "Range (Min-Max)" and self.mask_min is not None and self.mask_max is not None:
                    condition = (X[col] >= self.mask_min) & (X[col] <= self.mask_max)
                    X.loc[condition, col] = np.nan
        return X

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_cols=None, factor=1.5):
        self.outlier_cols = outlier_cols
        self.factor = factor
        self.bounds_ = {{}}
    def fit(self, X, y=None):
        if self.outlier_cols:
            for col in self.outlier_cols:
                if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.bounds_[col] = (Q1 - (self.factor * IQR), Q3 + (self.factor * IQR))
        return self
    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

class RandomSampleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        self.saved_values = {{}}
        self.random_state = random_state
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            vals = X[col].dropna().values
            if len(vals) > 0: self.saved_values[col] = vals
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        rng = np.random.default_rng(self.random_state)
        for col in X.columns:
            if col in self.saved_values:
                mask = X[col].isna()
                if mask.any() and len(self.saved_values[col]) > 0:
                    X.loc[mask, col] = rng.choice(self.saved_values[col], size=mask.sum())
        return X

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            try:
                series = pd.to_datetime(X[col], errors='coerce', infer_datetime_format=True)
                X[f"{{col}}_year"] = series.dt.year
                X[f"{{col}}_month"] = series.dt.month
                X[f"{{col}}_day"] = series.dt.day
                X[f"{{col}}_dow"] = series.dt.dayofweek
                X = X.drop(columns=[col])
            except: pass
        return X

class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, binning_config=None):
        self.binning_config = binning_config if binning_config else {{}}
        self.bins_ = {{}}
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col, config in self.binning_config.items():
            if col in X.columns:
                try:
                    if config['method'] == 'Automatic (Quantile)':
                        try: _, edges = pd.qcut(X[col], q=config['params'], retbins=True, duplicates='drop')
                        except: _, edges = pd.cut(X[col], bins=config['params'], retbins=True, duplicates='drop')
                        self.bins_[col] = edges
                    elif config['method'] == 'Manual Ranges':
                        self.bins_[col] = sorted(config['params'])
                except: pass
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, edges in self.bins_.items():
            if col in X.columns:
                try:
                    b_name = f"{{col}}_binned"
                    res = pd.cut(X[col], bins=edges, labels=False, include_lowest=True)
                    if hasattr(res, 'cat'): X[b_name] = (res.cat.codes + 1).astype(int)
                    else: X[b_name] = (res.fillna(-1) + 1).astype(int)
                    X = X.drop(columns=[col])
                except: pass
        return X

# --- CONFIGURATION ---
TARGET = '{config['target_col']}'
DROP_COLS = {config['drop_cols']}
BIN_CONFIG = {bin_config_str}
CUSTOM_CODE = \"\"\"{custom_code_str}\"\"\"

# MASKING & OUTLIERS
MASK_COLS = {mask_cols}
MASK_MODE = '{mask_mode}'
MASK_VAL = {mask_val if mask_val is not None else 'None'}
MASK_MIN = {mask_min if mask_min is not None else 'None'}
MASK_MAX = {mask_max if mask_max is not None else 'None'}
USE_CLIP = {use_clip}
OUTLIER_COLS = {outlier_cols}
FACTOR = {outlier_factor}

KNN_K = {config.get('imp_num_knn_k', 5)}
IMP_MEAN = {config.get('imp_num_mean', [])}
IMP_MEDIAN = {config.get('imp_num_median', [])}
IMP_KNN = {config.get('imp_num_knn', [])}
IMP_RANDOM = {config.get('imp_num_random', [])}
IMP_MODE = {config.get('imp_cat_mode', [])}
IMP_CONST = {config.get('imp_cat_const', [])}

COLS_STANDARD = {config['cols_standard']}
COLS_MINMAX = {config['cols_minmax']}
COLS_ROBUST = {config['cols_robust']}
COLS_MAXABS = {config.get('cols_maxabs', [])}
COLS_ONEHOT = {config['cols_onehot']}
COLS_ORDINAL = {config['cols_ordinal']}
COLS_LOG = {config['cols_log']}
COLS_DATE = {config['cols_date']}

{target_map_code}

# --- LOAD DATA ---
df = None
loaders = [
    lambda: pd.read_csv('your_dataset.csv'),
    lambda: pd.read_excel('your_dataset.xlsx'),
    lambda: pd.read_json('your_dataset.json'),
    lambda: pd.read_parquet('your_dataset.parquet')
]
for loader in loaders:
    try: 
        df = loader()
        break
    except: continue
if df is None: raise ValueError("Could not load dataset")

# --- SPLIT ---
if DROP_COLS: 
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop)

df = df.dropna(subset=[TARGET])
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PIPELINE COMPONENTS ---
"""

    # --- BLOCK 2: PIPELINE COMPONENTS (Plain string to avoid NameError on {c}) ---
    code += """
imp_transformers = []
if IMP_MEAN: imp_transformers.append(('mean', SimpleImputer(strategy='mean'), IMP_MEAN))
if IMP_MEDIAN: imp_transformers.append(('median', SimpleImputer(strategy='median'), IMP_MEDIAN))
if IMP_KNN: imp_transformers.append(('knn', KNNImputer(n_neighbors=KNN_K), IMP_KNN))
if IMP_RANDOM: imp_transformers.append(('random', RandomSampleImputer(random_state=42), IMP_RANDOM))
if IMP_MODE: imp_transformers.append(('mode', SimpleImputer(strategy='most_frequent'), IMP_MODE))
if IMP_CONST: imp_transformers.append(('const', SimpleImputer(strategy='constant', fill_value='Missing'), IMP_CONST))

all_selected = set(IMP_MEAN + IMP_MEDIAN + IMP_KNN + IMP_RANDOM + IMP_MODE + IMP_CONST)
remaining = [c for c in X.columns if c not in all_selected]
rem_num = [c for c in remaining if pd.api.types.is_numeric_dtype(X[c])]
rem_cat = [c for c in remaining if c not in rem_num]

if rem_num: imp_transformers.append(('def_num', SimpleImputer(strategy='mean'), rem_num))
if rem_cat: imp_transformers.append(('def_cat', SimpleImputer(strategy='most_frequent'), rem_cat))

imputer = ColumnTransformer(imp_transformers, verbose_feature_names_out=False)

def resolve_targets(lst):
    final = []
    for c in lst:
        if c in BIN_CONFIG: final.append(f"{c}_binned")
        else: final.append(c)
    return final

transformers = []
if COLS_DATE: transformers.append(('date', DateFeatureGenerator(), COLS_DATE))
if COLS_LOG: transformers.append(('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), COLS_LOG))

if COLS_STANDARD: transformers.append(('std', StandardScaler(with_mean=False), resolve_targets(COLS_STANDARD)))
if COLS_MINMAX: transformers.append(('minmax', MinMaxScaler(), resolve_targets(COLS_MINMAX)))
if COLS_ROBUST: transformers.append(('rob', RobustScaler(with_centering=False), resolve_targets(COLS_ROBUST)))
if COLS_MAXABS: transformers.append(('maxabs', MaxAbsScaler(), resolve_targets(COLS_MAXABS)))

try: ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=20)
except: ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

if COLS_ONEHOT: transformers.append(('ohe', ohe, resolve_targets(COLS_ONEHOT)))
if COLS_ORDINAL: transformers.append(('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), resolve_targets(COLS_ORDINAL)))

preprocessor = ColumnTransformer(transformers, remainder='passthrough')

"""
    # --- BLOCK 3: MODEL INSTANTIATION (f-string allowed) ---
    code += model_instantiation
    
    # --- BLOCK 4: EXECUTION (Standard string) ---
    code += """
# --- EXECUTION ---
steps = [
    ('custom_fe', CustomScriptTransformer(CUSTOM_CODE)),
    ('masking', MaskingTransformer(MASK_COLS, MASK_MODE, MASK_VAL, MASK_MIN, MASK_MAX)),
    ('clipping', OutlierClipper(OUTLIER_COLS if USE_CLIP else [], FACTOR)),
    ('imputer', imputer),
    ('binning', BinningTransformer(BIN_CONFIG)),
    ('preprocessor', preprocessor)
]
"""
    if config.get('use_smote'): code += f"\nsteps.append(('sampler', SMOTE(random_state=42)))"
    
    if config['use_poly']: 
        deg = config.get('poly_degree', 2)
        code += f"\nsteps.append(('poly', PolynomialFeatures(degree={deg}, interaction_only=True, include_bias=False)))"
    
    if config['use_pca']: 
        n = config.get('pca_components', 5)
        if "TruncatedSVD" in config.get('pca_method', ''):
            code += f"\nsteps.append(('pca', TruncatedSVD(n_components={n})))"
        else:
            code += f"\nsteps.append(('pca', PCA(n_components={n})))"
    
    code += f"""
steps.append(('model', model))
final_pipeline = {pipeline_class}(steps)

try:
    print("Training Pipeline...")
    final_pipeline.fit(X_train, y_train)
    score = final_pipeline.score(X_test, y_test)
    print(f"Test Score: {{score:.4f}}")
except Exception as e:
    print(f"Error: {{e}}")
"""
    return code


# --- 4. UI LAYOUT (SIDEBAR) ---

with st.sidebar:

    st.title("🎛️ Project Controls")

    st.write("Contributors: **Wasiful Haque, Md. Jahirul Islam, Saifuddin Yasir**")

    

    # [FIX] Robust File Uploader with multiple types

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt"])

    

    if uploaded_file:

        try:

            if st.session_state.loaded_file_name != uploaded_file.name:

                df_loaded = None

                fname = uploaded_file.name.lower()

                

                # [FIX] Logic for diverse file types

                try:

                    if fname.endswith('.csv'): 

                        try: df_loaded = pd.read_csv(uploaded_file)

                        except UnicodeDecodeError: df_loaded = pd.read_csv(uploaded_file, encoding='latin1')

                    elif fname.endswith(('.xlsx', '.xls')): 

                        df_loaded = pd.read_excel(uploaded_file)

                    elif fname.endswith('.json'): 

                        df_loaded = pd.read_json(uploaded_file)

                    elif fname.endswith('.parquet'): 

                        df_loaded = pd.read_parquet(uploaded_file)

                    elif fname.endswith(('.tsv', '.txt')): 

                        df_loaded = pd.read_csv(uploaded_file, sep='\t')

                except Exception as load_err:

                    st.error(f"Format recognized but load failed: {load_err}")



                if df_loaded is not None:

                    # Clear state

                    keys_to_clear = [

                        'X_train', 'X_test', 'y_train', 'y_test', 

                        'model_results', 'pipeline_config', 'best_model', 

                        'preprocessor', 'imputer_model', 'pca_model', 

                        'poly_model', 'feature_names', 'le', 'best_params',

                        'inference_data', 'custom_code_content', 'raw_feature_names'

                    ]

                    for key in keys_to_clear:

                        if key in st.session_state: del st.session_state[key]



                    st.session_state.df = df_loaded

                    st.session_state.df_raw = df_loaded.copy()

                    st.session_state.loaded_file_name = uploaded_file.name

                    st.success(f"Loaded {fname} ({len(df_loaded)} rows)!")

                    st.rerun()

                

        except Exception as e: st.error(f"Critical Error: {e}")



# --- FIX: DEFINE TABS HERE ---

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

    "📊 Visualization", "⚙️ Preprocessing", "🧠 Training", 

    "📈 Evaluation", "⚡ Tuning", "🔮 Inference"

])

        

# --- TAB 1: VISUALIZATION ---
with tab1:
    st.subheader("Explore Your Data")
    has_processed = 'X_train' in st.session_state and st.session_state.X_train is not None
    default_idx = 1 if has_processed else 0

    data_source = st.radio("Select Data Stage:", ["Raw Data", "Preprocessed (Train Set)"], index=default_idx, horizontal=True, disabled=not has_processed)

    df_viz = None
    if data_source == "Raw Data":
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df, height=400, use_container_width=True)
            df_viz = st.session_state.df 
        else: st.info("📂 Please upload a dataset in the sidebar to begin.")
        
    elif data_source == "Preprocessed (Train Set)":
        try:
            X_v = st.session_state.X_train
            y_v = st.session_state.y_train
            
            # Use the clean names generated in preprocess_and_split
            feat_names = st.session_state.feature_names
            
            # Safety check if shapes mismatch (e.g. unexpected transform issue)
            if len(feat_names) != X_v.shape[1]:
                feat_names = [f"Feat_{i}" for i in range(X_v.shape[1])]

            # --- Create Visualization DataFrame ---
            # If X_v is already a dataframe (due to set_output=pandas), use it directly
            if hasattr(X_v, 'columns'):
                df_viz = pd.DataFrame(X_v)
            else:
                df_viz = pd.DataFrame(X_v, columns=feat_names)

            # Optional: Add Target column back for context/correlation
            if y_v is not None:
                t_name = st.session_state.get('target_col', 'Target')
                # Reset index to ensure it aligns with X_v
                y_series = pd.Series(y_v, name=t_name).reset_index(drop=True)
                df_viz = df_viz.reset_index(drop=True)
                df_viz[t_name] = y_series
                
            st.dataframe(df_viz, height=400, use_container_width=True)
            
        except Exception as e: st.error(f"Error loading processed data: {e}")

    if df_viz is not None:
        st.divider()
        viz_mode = st.radio("Choose Visualization:", ["Quick Overview", "Correlation Heatmap", "Distribution Plots", "Deep Profiling (ydata)"], horizontal=True)
        numeric_df = df_viz.select_dtypes(include=np.number)
        
        if viz_mode == "Quick Overview":
            st.write(f"**Shape:** {df_viz.shape}")
            st.write(df_viz.describe())
        elif viz_mode == "Correlation Heatmap":
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                # Only show annotations if matrix isn't too huge
                do_annot = df_viz.shape[1] < 20
                sns.heatmap(numeric_df.corr(), annot=do_annot, fmt=".2f", cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else: st.warning("No numeric columns.")
        elif viz_mode == "Distribution Plots":
            col = st.selectbox("Select Column", df_viz.columns)
            if col:
                fig, ax = plt.subplots(figsize=(10, 4))
                if pd.api.types.is_numeric_dtype(df_viz[col]):
                    sns.histplot(df_viz[col], kde=True, ax=ax)
                else:
                    sns.countplot(y=df_viz[col], ax=ax, order=df_viz[col].value_counts().iloc[:15].index)
                st.pyplot(fig)
                plt.close(fig)
        elif viz_mode == "Deep Profiling (ydata)":
            # Guardrail for memory issues on cloud
            if df_viz.shape[0] * df_viz.shape[1] > 1_000_000:
                st.warning("⚠️ Dataset too large for Profiling in Streamlit Cloud. Please subset or run locally.")
            else:
                if st.button("Generate Report"):
                    with st.spinner("Analyzing..."):
                        pr = ProfileReport(df_viz, minimal=True, explorative=False)
                        components.html(pr.to_html(), height=800, scrolling=True)
                        

# --- TAB 2: PREPROCESSING ---
with tab2:
    if st.session_state.df is not None:
        st.subheader("Pipeline Configuration")
        
        # --- 0. CUSTOM FEATURE ENGINEERING ---
        with st.expander("🛠️ Custom Feature Engineering (Python Code)", expanded=False):
            st.info("Write simple Python scripts to create new features. The dataframe is available as `df`.")
           
            fe_template = st.selectbox("Load Template:", ["Empty", "Simple Math"], index=0)
            
            def get_template_code(tmpl):
                if tmpl == "Simple Math": return "df['Ratio'] = df['ColA'] / df['ColB']\ndf['Log_Col'] = np.log1p(df['ColC'])"
                return ""

            # Only update content if it's empty and a template is selected
            if fe_template != "Empty" and not st.session_state.custom_code_content:
                st.session_state.custom_code_content = get_template_code(fe_template)

            # Text Area for Code
            custom_code_input = st.text_area("Python Script", value=st.session_state.custom_code_content, height=300, key="code_area_widget")
            st.session_state.custom_code_content = custom_code_input

            c_btn1, c_btn2 = st.columns([1, 4])
            with c_btn1:
                if st.button("▶️ Apply & Update Columns"):
                    if st.session_state.custom_code_content:
                        try:
                            # Context for execution
                            local_vars = {
                                'df': st.session_state.df.copy(), # Work on a copy first
                                'pd': pd, 
                                'np': np, 
                                're': re
                            }
                            
                            # Execute Code
                            exec(st.session_state.custom_code_content, globals(), local_vars)
                            
                            # Validation and Update
                            if isinstance(local_vars['df'], pd.DataFrame):
                                st.session_state.df = local_vars['df'] 
                                st.success(f"Success! Columns updated. New shape: {st.session_state.df.shape}")
                                st.rerun() # FORCE RERUN TO UPDATE DROPDOWNS BELOW
                            else: 
                                st.error("Error: The code must result in a variable named 'df' that is a pandas DataFrame.")
                        except Exception as e: 
                            st.error(f"Execution Error: {e}")
            with c_btn2:
                if st.button("🔄 Reset to Original"):
                    if st.session_state.df_raw is not None:
                        st.session_state.df = st.session_state.df_raw.copy()
                        st.session_state.custom_code_content = "" 
                        st.success("Reset to raw data!"); 
                        st.rerun()

        # --- REFRESH COLUMN LISTS ---
        all_cols = st.session_state.df.columns.tolist()
        
        c1, c2, c3 = st.columns(3)
        with c1: drop_cols = st.multiselect("Drop Columns", all_cols)
        
        # Calculate remaining columns for Target selection
        remaining_cols = [c for c in all_cols if c not in drop_cols]
        
        with c2: 
            if remaining_cols: 
                target_idx = 0
                if 'target_col' in st.session_state and st.session_state.target_col in remaining_cols:
                    target_idx = remaining_cols.index(st.session_state.target_col)
                target_col = st.selectbox("Target Column", remaining_cols, index=target_idx)
            else: target_col = None
            
        with c3:
            is_ts = st.checkbox("Time Series Mode")
            task_type = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)

        # Identify Feature Types for Widgets
        feature_cols = [c for c in remaining_cols if c != target_col]
        df_features = st.session_state.df[feature_cols]
        auto_num = df_features.select_dtypes(include=np.number).columns.tolist()
        auto_cat = [c for c in feature_cols if c not in auto_num]

        st.divider()
        st.markdown("### 2. Data Cleaning & Imputation")
        
        with st.expander("🎭 Handle Placeholder Values", expanded=True):
            col_mask1, col_mask2 = st.columns(2)
            with col_mask1:
                mask_mode = st.radio("Masking Mode", ["Single Value", "Range (Min-Max)"], horizontal=True)
                mask_val, mask_min, mask_max = None, None, None
                if mask_mode == "Single Value": mask_val = st.text_input("Value to treat as Missing", value="0")
                else:
                    c_min, c_max = st.columns(2)
                    with c_min: mask_min = st.number_input("Min Value", value=0.0)
                    with c_max: mask_max = st.number_input("Max Value", value=0.0)
            with col_mask2: mask_cols = st.multiselect("Apply Masking to Columns", auto_num + auto_cat)

        # --- OUTLIER EXPANDER ---
        with st.expander("🚨 Outlier Handling (IQR Clipping)", expanded=True):
            col_out1, col_out2 = st.columns(2)
            with col_out1:
                use_outlier_clip = st.checkbox("Enable IQR Clipping")
                outlier_factor = st.slider("IQR Factor (k)", 1.0, 3.0, 1.5)
            with col_out2: 
                outlier_cols = st.multiselect("Apply to Numeric", auto_num, disabled=not use_outlier_clip)

        st.write("#### Imputation Strategies")
        i1, i2 = st.columns(2)
        with i1:
            st.caption("Numeric Columns")
            imp_num_mean = st.multiselect("Impute with Mean", auto_num)
            avail_median = [c for c in auto_num if c not in imp_num_mean]
            imp_num_median = st.multiselect("Impute with Median", avail_median)
            
            avail_knn = [c for c in avail_median if c not in imp_num_median]
            imp_num_knn = st.multiselect("Impute with KNN", avail_knn)
            if imp_num_knn: knn_k = st.slider("KNN Neighbors (k)", 1, 20, 5, key="knn_slider")
            else: knn_k = 5

            avail_random = [c for c in avail_knn if c not in imp_num_knn]
            imp_num_random = st.multiselect("Impute with Random", avail_random)
        with i2:
            st.caption("Categorical Columns")
            imp_cat_mode = st.multiselect("Mode", auto_cat)
            avail_const = [c for c in auto_cat if c not in imp_cat_mode]
            imp_cat_const = st.multiselect("Constant", avail_const)

        st.divider()
        st.markdown("### 3. Feature Engineering")
        
        binning_config = {}
        with st.expander("🧩 Feature Generation", expanded=False):
            fe_col1, fe_col2 = st.columns(2)
            with fe_col1: cols_date = st.multiselect("Extract Date", auto_cat + auto_num)
            with fe_col2: cols_log = st.multiselect("Log Transform", auto_num)
            
            st.markdown("**Binning**")
            avail_bin = [c for c in auto_num if c not in cols_date]
            cols_to_bin = st.multiselect("Select Columns to Bin", avail_bin)
            if cols_to_bin:
                bin_method = st.radio("Binning Mode", ["Automatic", "Manual"], horizontal=True)
                if bin_method == "Automatic":
                    n_bins = st.slider("Number of Bins", 2, 20, 5)
                    for c in cols_to_bin: binning_config[c] = {'method': 'Automatic (Quantile)', 'params': n_bins}
                else:
                    for c in cols_to_bin:
                        user_edges = st.text_input(f"Edges for '{c}'", f"{int(df_features[c].min())}, {int(df_features[c].max())}")
                        try: binning_config[c] = {'method': 'Manual Ranges', 'params': sorted([float(x.strip()) for x in user_edges.split(',')])}
                        except: st.error(f"Invalid edges for {c}")

        derived_bin_cols = [f"{c}_binned" for c in cols_to_bin]
        numeric_options = [c for c in auto_num if c not in cols_date and c not in cols_to_bin]
        categorical_options = auto_cat + derived_bin_cols

        st.markdown("**Scaling & Encoding**")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1: cols_standard = st.multiselect("Standard Scaler", numeric_options)
        with col_t2: cols_minmax = st.multiselect("MinMax Scaler", numeric_options)
        with col_t3: cols_robust = st.multiselect("Robust Scaler", numeric_options)
        with col_t4: cols_maxabs = st.multiselect("MaxAbs Scaler", numeric_options)
            
        col_e1, col_e2 = st.columns(2)
        with col_e1: cols_onehot = st.multiselect("One-Hot Encoding", categorical_options, default=auto_cat)
        with col_e2: cols_ordinal = st.multiselect("Ordinal Encoding", categorical_options, default=derived_bin_cols)

        st.divider()
        col_s2, col_s3 = st.columns(2)
        with col_s2: 
            test_size = st.slider("Test Split", 0.1, 0.5, 0.2)
            use_smote = st.checkbox("SMOTE", help="Classif Only")
        with col_s3:
            use_pca = st.checkbox("Dimensionality Reduction")
            pca_method = "PCA (Dense)"
            pca_components = 5
            if use_pca:
                pca_method = st.radio("Method", ["PCA (Dense)", "TruncatedSVD (Sparse)"], horizontal=True)
                pca_components = st.slider("Components", 2, 50, 5)
            
            use_poly = st.checkbox("Poly Features")
            poly_degree = 2
            if use_poly:
                poly_degree = st.slider("Poly Deg", 2, 10, 2)

        date_col_sort = None
        if is_ts: date_col_sort = st.selectbox("Date Sort Col", remaining_cols)

        if st.button("🚀 Process Data"):
            if target_col:
                with st.spinner("Processing..."):
                    try:
                       
                        X_tr, X_te, y_tr, y_te, prep, le, pca, poly, proc_feats, imp, raw_feats = preprocess_and_split(
                            st.session_state.df, target_col, task_type, is_ts, date_col_sort,
                            mask_mode, mask_val, mask_min, mask_max, mask_cols, 
                            use_outlier_clip, outlier_cols, outlier_factor,       
                            imp_num_mean, imp_num_median, imp_num_knn, imp_num_random, 
                            imp_cat_mode, imp_cat_const, knn_k,
                            test_size, drop_cols,
                            cols_standard, cols_minmax, cols_robust, cols_maxabs,
                            cols_onehot, cols_ordinal,
                            cols_log, binning_config, cols_date, 
                            use_pca, pca_method, pca_components,
                            use_poly, poly_degree,
                            use_smote,
                            st.session_state.custom_code_content 
                        )
                        
                        if X_tr is not None:
                            st.session_state.pipeline_config = {
                                'drop_cols': drop_cols, 'target_col': target_col, 'task_type': task_type,
                                'custom_code': st.session_state.custom_code_content, 
                                'mask_mode': mask_mode, 'mask_val': mask_val, 
                                'mask_min': mask_min, 'mask_max': mask_max, 'mask_cols': mask_cols,
                                'use_outlier_clip': use_outlier_clip,
                                'outlier_cols': outlier_cols,
                                'outlier_factor': outlier_factor,
                                'cols_standard': cols_standard, 'cols_minmax': cols_minmax, 'cols_robust': cols_robust, 'cols_maxabs': cols_maxabs,
                                'cols_onehot': cols_onehot, 'cols_ordinal': cols_ordinal,
                                'cols_log': cols_log, 'binning_config': binning_config, 'cols_date': cols_date,
                                'imp_num_mean': imp_num_mean, 'imp_num_median': imp_num_median,
                                'imp_num_knn': imp_num_knn, 'imp_num_random': imp_num_random,
                                'imp_num_knn_k': knn_k,
                                'imp_cat_mode': imp_cat_mode, 'imp_cat_const': imp_cat_const,
                                'use_smote': use_smote, 'use_pca': use_pca, 'pca_components': pca_components,
                                'pca_method': pca_method,
                                'use_poly': use_poly, 'poly_degree': poly_degree
                            }
                            st.session_state.model_results = {}
                            st.session_state.best_model = None
                            st.session_state.X_train = X_tr
                            st.session_state.X_test = X_te
                            st.session_state.y_train = y_tr
                            st.session_state.y_test = y_te
                            st.session_state.target_col = target_col
                            st.session_state.task_type = task_type
                            st.session_state.is_ts = is_ts
                            st.session_state.le = le
                            st.session_state.preprocessor = prep
                            st.session_state.imputer_model = imp
                            st.session_state.pca_model = pca
                            st.session_state.poly_model = poly
                            st.session_state.feature_names = proc_feats
                            st.session_state.raw_feature_names = raw_feats
                            
                            st.success(f"Processed! Train: {X_tr.shape}")
                            st.rerun()
                    except Exception as e: st.error(f"Error: {e}")
            else: st.warning("Select Target.")
            
# --- TAB 3: TRAINING ---

with tab3:

    if 'X_train' in st.session_state:

        st.subheader("Model Training")

        available_models = get_available_models(st.session_state.task_type)

        base_model_names = list(available_models.keys())

        

        c_train1, c_train2 = st.columns(2)

        with c_train1:

            selected_base = st.multiselect("Select Base Models", base_model_names, default=base_model_names[:2])

        

        ensemble_config = {}

        with c_train2:

            st.markdown("**Ensemble Strategy**")

            use_voting = st.checkbox("Train Voting Ensemble")

            use_stacking = st.checkbox("Train Stacking Ensemble")

            

            if use_voting or use_stacking:

                if len(selected_base) < 2:

                    st.error("Select at least 2 Base Models for Ensembles.")

                else:

                    est_list = [(name, available_models[name]) for name in selected_base]

                    if use_voting:

                        if st.session_state.task_type == 'Classification':

                            ensemble_config['Voting'] = VotingClassifier(estimators=est_list, voting='soft')

                        else:

                            ensemble_config['Voting'] = VotingRegressor(estimators=est_list)

                    if use_stacking:

                   

                        final_est = LogisticRegression() if st.session_state.task_type == 'Classification' else LinearRegression()

                        if st.session_state.task_type == 'Classification':

                            ensemble_config['Stacking'] = StackingClassifier(estimators=est_list, final_estimator=final_est)

                        else:

                            ensemble_config['Stacking'] = StackingRegressor(estimators=est_list, final_estimator=final_est)



        if st.button("Train Selected Models"):

            res = {}

            final_models_to_train = {name: available_models[name] for name in selected_base}

            final_models_to_train.update(ensemble_config)


            if not final_models_to_train:

                st.error("Please select at least one model.")

            else:

                bar = st.progress(0)

                for i, (name, model) in enumerate(final_models_to_train.items()):

                    try:

                        model.fit(st.session_state.X_train, st.session_state.y_train)

                        preds = model.predict(st.session_state.X_test)

                        if st.session_state.task_type == 'Classification':

                            score = accuracy_score(st.session_state.y_test, preds)

                            metric = "Accuracy"

                        else:

                            score = r2_score(st.session_state.y_test, preds)

                            metric = "R2 Score"

                        res[name] = {"model": model, "score": score, "preds": preds, "metric": metric}

                    except Exception as e: st.error(f"Failed {name}: {e}")

                    bar.progress((i+1)/len(final_models_to_train))

                

                st.session_state.model_results = res

                st.success("Training Complete!")

                

# --- TAB 4: EVALUATION ---

with tab4:

    if st.session_state.model_results:

        st.subheader("Model Evaluation")

        

        # 1. Summary Table

        res_df = pd.DataFrame([{ "Model": k, v['metric']: v['score']} for k,v in st.session_state.model_results.items()])

        st.table(res_df.sort_values(by=res_df.columns[1], ascending=False))

        

        # 2. Detailed Drill-Down

        sel_model_name = st.selectbox("Select Model for Details", list(st.session_state.model_results.keys()))

        selected_data = st.session_state.model_results[sel_model_name]

        

        st.markdown(f"### Performance: {sel_model_name}")

        evaluate_model_performance(

            selected_data['model'], 

            st.session_state.X_test, 

            st.session_state.y_test, 

            st.session_state.task_type, 

            st.session_state.le

        )


# --- TAB 5: TUNING ---
with tab5:
    if 'X_train' in st.session_state:
        st.subheader("⚡ Automated Tuning")
        
        # Initialize session state for tuning results if not present
        if 'tuning_results' not in st.session_state: 
            st.session_state.tuning_results = None

        if not st.session_state.model_results:
            st.warning("⚠️ No models trained yet. Please go to Tab 3 (Training) and train a model first.")
        else:
            trained_models_list = list(st.session_state.model_results.keys())
            tune_model_name = st.selectbox("Select Model to Tune", trained_models_list)
            
            # --- DETAILED HYPERPARAMETERS ---
            model_params_schema = {
                'Logistic Regression': {'C': {'type': 'float_list', 'default': '0.01, 0.1, 1.0, 10.0', 'label': 'C (Inverse Reg)'}, 'solver': {'type': 'cat_list', 'options': ['saga', 'liblinear', 'lbfgs'], 'default': ['saga'], 'label': 'Solver'}, 'penalty': {'type': 'cat_list', 'options': ['l2', 'l1', 'elasticnet'], 'default': ['l2'], 'label': 'Penalty'}, 'l1_ratio': {'type': 'float_list', 'default': '0.5', 'label': 'L1 Ratio (ElasticNet)'}},
                'Random Forest': {'n_estimators': {'type': 'int_list', 'default': '100, 200, 300', 'label': 'Trees'}, 'max_depth': {'type': 'int_list', 'default': '10, 20, 30, None', 'label': 'Max Depth (None=unlimited)'}, 'min_samples_split': {'type': 'int_list', 'default': '2, 5, 10', 'label': 'Min Samples Split'}, 'min_samples_leaf': {'type': 'int_list', 'default': '1, 2, 4', 'label': 'Min Samples Leaf'}, 'bootstrap': {'type': 'cat_list', 'options': [True, False], 'default': [True], 'label': 'Bootstrap'}},
                'XGBoost': {'n_estimators': {'type': 'int_list', 'default': '100, 300', 'label': 'Trees'}, 'learning_rate': {'type': 'float_list', 'default': '0.01, 0.1, 0.3', 'label': 'Learning Rate'}, 'max_depth': {'type': 'int_list', 'default': '3, 6, 9', 'label': 'Max Depth'}, 'subsample': {'type': 'float_list', 'default': '0.8, 1.0', 'label': 'Subsample'}, 'colsample_bytree': {'type': 'float_list', 'default': '0.8, 1.0', 'label': 'Colsample by Tree'}},
                'LightGBM': {'n_estimators': {'type': 'int_list', 'default': '100, 300', 'label': 'Trees'}, 'learning_rate': {'type': 'float_list', 'default': '0.01, 0.1', 'label': 'Learning Rate'}, 'num_leaves': {'type': 'int_list', 'default': '31, 50, 100', 'label': 'Num Leaves'}, 'max_depth': {'type': 'int_list', 'default': '-1, 10, 20', 'label': 'Max Depth (-1=No Limit)'}},
                'CatBoost': {'iterations': {'type': 'int_list', 'default': '500, 1000', 'label': 'Iterations'}, 'learning_rate': {'type': 'float_list', 'default': '0.01, 0.1', 'label': 'Learning Rate'}, 'depth': {'type': 'int_list', 'default': '4, 6, 10', 'label': 'Depth'}, 'l2_leaf_reg': {'type': 'float_list', 'default': '1, 3, 5', 'label': 'L2 Leaf Reg'}},
                'AdaBoost': {'n_estimators': {'type': 'int_list', 'default': '50, 100, 200', 'label': 'Estimators'}, 'learning_rate': {'type': 'float_list', 'default': '0.01, 0.1, 1.0', 'label': 'Learning Rate'}},
                'Gradient Boosting': {'n_estimators': {'type': 'int_list', 'default': '100, 200', 'label': 'Estimators'}, 'learning_rate': {'type': 'float_list', 'default': '0.01, 0.1', 'label': 'Learning Rate'}, 'max_depth': {'type': 'int_list', 'default': '3, 5', 'label': 'Depth'}},
                'SVM': {'C': {'type': 'float_list', 'default': '0.1, 1, 10', 'label': 'C'}, 'kernel': {'type': 'cat_list', 'options': ['linear', 'rbf', 'poly'], 'default': ['rbf'], 'label': 'Kernel'}, 'gamma': {'type': 'cat_list', 'options': ['scale', 'auto'], 'default': ['scale'], 'label': 'Gamma'}},
                'KNN': {'n_neighbors': {'type': 'int_list', 'default': '3, 5, 7, 9', 'label': 'Neighbors'}, 'weights': {'type': 'cat_list', 'options': ['uniform', 'distance'], 'default': ['uniform'], 'label': 'Weights'}},
                'Linear Regression': {'fit_intercept': {'type': 'cat_list', 'options': [True, False], 'default': [True], 'label': 'Intercept'}},
                'Ridge': {'alpha': {'type': 'float_list', 'default': '0.1, 1.0, 10.0', 'label': 'Alpha'}},
                'Lasso': {'alpha': {'type': 'float_list', 'default': '0.1, 1.0, 10.0', 'label': 'Alpha'}}
            }

            current_schema = model_params_schema.get(tune_model_name, {})
            
            if not current_schema:
                st.info(f"Detailed tuning schema not available for {tune_model_name} (likely an Ensemble). Using simple defaults if possible.")
                
            final_param_grid = {}
            col_p1, col_p2 = st.columns(2)
            cols_list = [col_p1, col_p2]

            for i, (param_name, config) in enumerate(current_schema.items()):
                with cols_list[i % 2]:
                    if config['type'] == 'cat_list':
                        selected_vals = st.multiselect(config['label'], options=config['options'], default=config['default'], key=f"tune_{param_name}")
                        if selected_vals: final_param_grid[param_name] = selected_vals
                    elif config['type'] in ['int_list', 'float_list']:
                        raw_text = st.text_input(config['label'], value=config['default'], key=f"tune_{param_name}")
                        if raw_text.strip():
                            vals = [x.strip() for x in raw_text.split(',')]
                            parsed_vals = []
                            for v in vals:
                                if v.lower() == 'none': parsed_vals.append(None)
                                elif config['type'] == 'int_list': 
                                    try: parsed_vals.append(int(v))
                                    except: pass
                                else: 
                                    try: parsed_vals.append(float(v))
                                    except: pass
                            if parsed_vals: final_param_grid[param_name] = parsed_vals

            st.divider()
            col_set1, col_set2, col_set3 = st.columns(3)
            with col_set1: search_type = st.radio("Search Method", ["Random Search", "Grid Search"])
            with col_set2: n_iter = st.slider("Iterations", 5, 50, 10) if search_type == "Random Search" else 0
            with col_set3: st.write(f"**Grid Size:** {np.prod([len(v) for v in final_param_grid.values()]) if final_param_grid else 0} combinations")

            if st.button("Start Tuning", type="primary"):
                if not final_param_grid: st.warning("⚠️ No parameters selected.")
                else:
                    with st.spinner(f"Tuning {tune_model_name}..."):
                        available_fresh_models = get_available_models(st.session_state.task_type)
                        
                        if tune_model_name in ['Voting', 'Stacking']:
                            st.error("Direct tuning of Ensembles is not supported in this UI. Tune base models individually first.")
                        elif tune_model_name not in available_fresh_models:
                            st.error(f"Model {tune_model_name} not found in registry.")
                        else:
                            base_model = available_fresh_models[tune_model_name]
                            cv = TimeSeriesSplit(n_splits=3) if st.session_state.is_ts else 3
                            try:
                                if search_type == "Grid Search": search = GridSearchCV(base_model, final_param_grid, cv=cv, verbose=1, n_jobs=-1, error_score='raise')
                                else: search = RandomizedSearchCV(base_model, final_param_grid, n_iter=n_iter, cv=cv, verbose=1, n_jobs=-1, error_score='raise')
                                
                                search.fit(st.session_state.X_train, st.session_state.y_train)
                                st.success("Tuning Complete!")
                                
                                # Store results in Session State for persistence
                                st.session_state.tuning_results = {
                                    'model_name': tune_model_name,
                                    'best_score': search.best_score_,
                                    'best_params': search.best_params_,
                                    'best_estimator': search.best_estimator_
                                }
                                
                                # Update Global State
                                st.session_state.best_model = search.best_estimator_
                                st.session_state.best_params = search.best_params_
                                
                            except Exception as e: st.error(f"Tuning Failed: {e}")

            # --- DISPLAY RESULTS ---
    
            if st.session_state.tuning_results is not None:
                res = st.session_state.tuning_results
                
                st.divider()
                st.markdown(f"### Results for: **{res['model_name']}**")
                
                m1, m2 = st.columns(2)
                m1.metric("Best CV Score", f"{res['best_score']:.4f}")
                m2.write("**Best Parameters:**")
                m2.json(res['best_params'])
                
                st.divider()
                st.write("### Test Set Performance (Tuned Model)")
                try:
                    evaluate_model_performance(
                        res['best_estimator'],
                        st.session_state.X_test, 
                        st.session_state.y_test, 
                        st.session_state.task_type, 
                        st.session_state.le
                    )
                except Exception as e:
                    st.error(f"Could not evaluate model: {e}")

# --- TAB 6: INFERENCE ---
with tab6:
    st.subheader("🔮 Inference & Export")
    
    active_model = st.session_state.best_model

    if active_model is None and st.session_state.model_results:
        first_key = list(st.session_state.model_results.keys())[0]
        active_model = st.session_state.model_results[first_key]['model']
        st.info(f"Using base model **{first_key}** (No tuned model found).")
    
    if active_model is None: st.warning("Train a model in Tab 3 first.")
    elif st.session_state.preprocessor is None: st.warning("No Pipeline.")
    else:

        st.markdown("#### 1. Make Predictions")

        input_method = st.radio("Input Method", ["Upload CSV", "Manual Entry"])

        if 'inference_data' not in st.session_state: st.session_state.inference_data = None

        if input_method == "Upload CSV":
            inf_file = st.file_uploader("New Data (CSV)", type=["csv"])
            if inf_file: st.session_state.inference_data = pd.read_csv(inf_file); st.success("Loaded!")
        else:
            with st.form("manual_entry_form"):
                input_data = {}
                cols = st.columns(2)
                feats = st.session_state.raw_feature_names if 'raw_feature_names' in st.session_state and st.session_state.raw_feature_names else st.session_state.feature_names
                
                for i, col_name in enumerate(feats):
                    with cols[i % 2]:
                        input_data[col_name] = st.text_input(f"{col_name}", key=f"in_{i}")
                
                if st.form_submit_button("Generate"):
                    df_input = pd.DataFrame([input_data])
                    if st.session_state.df is not None:
                        ref_df = st.session_state.df
                        for col in df_input.columns:
                            if col in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[col]):
                                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
                    st.session_state.inference_data = df_input
                    st.success("Generated!")

        if st.session_state.inference_data is not None:
            st.divider()
            if st.button("Run Prediction", key="run_pred"):
                try:
                    cfg = st.session_state.get('pipeline_config', {})
                    X_new_raw = st.session_state.inference_data.copy()
                    
                    if cfg.get('drop_cols'):
                        cols_to_drop = [c for c in cfg['drop_cols'] if c in X_new_raw.columns]
                        if cols_to_drop: X_new_raw = X_new_raw.drop(columns=cols_to_drop)
                    
                    target_c = cfg.get('target_col')
                    if target_c and target_c in X_new_raw.columns:
                        X_new_raw = X_new_raw.drop(columns=[target_c])

                    if cfg.get('custom_code'):
                        try:
                            local_vars = {'df': X_new_raw, 'pd': pd, 'np': np, 're': re}
                            exec(cfg['custom_code'], globals(), local_vars)
                            X_new_raw = local_vars['df']
                        except Exception as e: st.error(f"Custom Feature Error: {e}"); st.stop()

                    mask_cols = cfg.get('mask_cols', [])
                    if mask_cols:
                        mask_mode = cfg.get('mask_mode')
                        for col in mask_cols:
                            if col in X_new_raw.columns and pd.api.types.is_numeric_dtype(X_new_raw[col]):
                                if mask_mode == 'Single Value':
                                    val = cfg.get('mask_val')
                                    try: v = float(val)
                                    except: v = val
                                    X_new_raw[col] = X_new_raw[col].replace(v, np.nan)
                                elif mask_mode == 'Range (Min-Max)':
                                    mn, mx = cfg.get('mask_min'), cfg.get('mask_max')
                                    if mn is not None and mx is not None:
                                        cond = (X_new_raw[col] >= mn) & (X_new_raw[col] <= mx)
                                        X_new_raw.loc[cond, col] = np.nan

                    if cfg.get('use_outlier_clip') and len(X_new_raw) > 5:
                        out_cols = cfg.get('outlier_cols', [])
                        factor = cfg.get('outlier_factor', 1.5)
                        for col in out_cols:
                            if col in X_new_raw.columns and pd.api.types.is_numeric_dtype(X_new_raw[col]):
                                Q1 = X_new_raw[col].quantile(0.25)
                                Q3 = X_new_raw[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower = Q1 - (factor * IQR)
                                upper = Q3 + (factor * IQR)
                                X_new_raw[col] = X_new_raw[col].clip(lower=lower, upper=upper)

                    X_new = st.session_state.preprocessor.transform(X_new_raw)
                    preds = active_model.predict(X_new)
                    
                    if st.session_state.le:
                        try:
                            final_preds = st.session_state.le.inverse_transform(preds)
                        except:
                            final_preds = preds 
                            st.warning("Could not map predictions back to labels.")
                    else: final_preds = preds
                    
                    st.success("Prediction Successful!")
                    res_df = X_new_raw.copy()
                    res_df['Prediction'] = final_preds
                    st.dataframe(res_df)
                    
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
                    
                except Exception as e: st.error(f"Prediction Error: {e}")

    st.divider()

    st.markdown("#### 2. Export & Save")
    
    c_ex1, c_ex2 = st.columns(2)
    
    with c_ex1:
        if active_model is not None and 'pipeline_config' in st.session_state:
            if st.button("Generate Python Script"):
                ui_name = type(active_model).__name__
                params_to_pass = st.session_state.get('best_params', {})
                script = generate_python_code(st.session_state.pipeline_config, ui_name, params_to_pass)
                st.code(script, language='python')
    
    with c_ex2:
        if active_model is not None:
            st.write("### Save Model")
          
            model_dict = {
                'model': active_model,
                'pipeline': st.session_state.preprocessor,
                'label_encoder': st.session_state.le,
                'config': st.session_state.pipeline_config,
                'feature_names': st.session_state.feature_names
            }
            
            serialized_model = None
            try:
                if dill is not None:
                    serialized_model = dill.dumps(model_dict)
                else:
                    serialized_model = pickle.dumps(model_dict, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                if "BinningTransformer" in str(e) or "not the same object" in str(e):
                    st.error("⚠️ Export Error: Custom class conflict detected.")
                    st.info("💡 **Fix:** Please install `dill` to solve this Streamlit-specific issue:\n`pip install dill`")
                else:
                    st.error(f"Serialization failed: {e}")

            if serialized_model:
                st.download_button(
                    label="⬇️ Download Trained Model (.pkl)",
                    data=serialized_model,
                    file_name="nocodeml_model.pkl",
                    mime="application/octet-stream"
                )
   
