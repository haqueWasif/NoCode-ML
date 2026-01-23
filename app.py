import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import re

# Profiling
from ydata_profiling import ProfileReport

# Scikit-Learn & Modeling
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix

# Imbalanced Learn
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="NoCodeML Studio", layout="wide", page_icon="📈")

# Initialize Session State
if 'df' not in st.session_state: st.session_state.df = None
if 'model_results' not in st.session_state: st.session_state.model_results = {}
if 'le' not in st.session_state: st.session_state.le = None
if 'preprocessor' not in st.session_state: st.session_state.preprocessor = None
if 'imputer_model' not in st.session_state: st.session_state.imputer_model = None
if 'pca_model' not in st.session_state: st.session_state.pca_model = None
if 'poly_model' not in st.session_state: st.session_state.poly_model = None
if 'best_model' not in st.session_state: st.session_state.best_model = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = []
if 'pipeline_config' not in st.session_state: st.session_state.pipeline_config = {}

# --- 2. CUSTOM TRANSFORMERS ---

class RandomSampleImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.saved_values = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else [f"col_{i}" for i in range(X.shape[1])]
        for col in X.columns:
            self.saved_values[col] = X[col].dropna().values
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            mask = X[col].isna()
            if mask.any() and col in self.saved_values and len(self.saved_values[col]) > 0:
                fill_values = np.random.choice(self.saved_values[col], size=mask.sum())
                X.loc[mask, col] = fill_values
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.feature_names_in_

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = [str(c) for c in X.columns] if hasattr(X, "columns") else None
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            series = pd.to_datetime(X[col], errors='coerce')
            X[f"{col}_year"] = series.dt.year
            X[f"{col}_month"] = series.dt.month
            X[f"{col}_day"] = series.dt.day
            X[f"{col}_dow"] = series.dt.dayofweek
            X = X.drop(columns=[col])
        return X.fillna(0)

    def get_feature_names_out(self, input_features=None):
        out_names = []
        if input_features is None: input_features = self.feature_names_in_
        for col in input_features:
            out_names.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dow"])
        return out_names

# --- 3. CORE LOGIC ---

def preprocess_and_split(df, target_col, task_type, is_ts, date_col_sort, 
                         imp_num_mean, imp_num_median, imp_num_zero, imp_num_random,
                         imp_cat_mode, imp_cat_const,
                         test_size, drop_cols,
                         cols_standard, cols_minmax, cols_robust, cols_onehot, cols_ordinal,
                         cols_log, binning_config, cols_date,
                         use_pca, pca_components,
                         use_poly, poly_degree,
                         use_smote):
    
    # 0. Initial Clean & Drop
    if drop_cols: df = df.drop(columns=drop_cols)
    df = df.dropna(subset=[target_col])
    
    if is_ts and date_col_sort:
        df[date_col_sort] = pd.to_datetime(df[date_col_sort])
        df = df.sort_values(by=date_col_sort)
        X = df.drop(columns=[target_col, date_col_sort])
    else:
        X = df.drop(columns=[target_col])
    y = df[target_col]

    # Target Encoding
    le = None
    if task_type == 'Classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = pd.to_numeric(y, errors='coerce')
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

    # 1. SPLIT
    if is_ts:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    else:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 2. GRANULAR IMPUTATION
    impute_transformers = []
    
    def get_valid_cols(cols, df_cols): return [c for c in cols if c in df_cols]
    
    if imp_num_mean: impute_transformers.append(('imp_mean', SimpleImputer(strategy='mean'), get_valid_cols(imp_num_mean, X.columns)))
    if imp_num_median: impute_transformers.append(('imp_median', SimpleImputer(strategy='median'), get_valid_cols(imp_num_median, X.columns)))
    if imp_num_zero: impute_transformers.append(('imp_zero', SimpleImputer(strategy='constant', fill_value=0), get_valid_cols(imp_num_zero, X.columns)))
    # HOT DECKING
    if imp_num_random: impute_transformers.append(('imp_random', RandomSampleImputer(), get_valid_cols(imp_num_random, X.columns)))
    
    if imp_cat_mode: impute_transformers.append(('imp_mode', SimpleImputer(strategy='most_frequent'), get_valid_cols(imp_cat_mode, X.columns)))
    if imp_cat_const: impute_transformers.append(('imp_const', SimpleImputer(strategy='constant', fill_value='Missing'), get_valid_cols(imp_cat_const, X.columns)))

    selected_cols = set(imp_num_mean + imp_num_median + imp_num_zero + imp_num_random + imp_cat_mode + imp_cat_const)
    remaining_cols = [c for c in X.columns if c not in selected_cols]
    
    if remaining_cols:
        rem_num = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(X[c])]
        rem_cat = [c for c in remaining_cols if c not in rem_num]
        
        if rem_num: impute_transformers.append(('imp_default_num', SimpleImputer(strategy='mean'), rem_num))
        if rem_cat: impute_transformers.append(('imp_default_cat', SimpleImputer(strategy='most_frequent'), rem_cat))

    if not impute_transformers:
        global_imputer = ColumnTransformer([('passthrough', 'passthrough', X.columns)], verbose_feature_names_out=False)
    else:
        global_imputer = ColumnTransformer(impute_transformers, verbose_feature_names_out=False)
    
    global_imputer.set_output(transform="pandas") 

    try:
        X_train_imp = global_imputer.fit_transform(X_train_raw)
        X_test_imp = global_imputer.transform(X_test_raw)
    except Exception as e:
        st.error(f"Imputation Failed: {e}")
        return None, None, None, None, None, None, None, None, None, None

    # 3. Binning (Destructive - drops original)
    def apply_binning(data_df):
        df_out = data_df.copy()
        if binning_config:
            for col, config in binning_config.items():
                if col in df_out.columns:
                    binned_col_name = f"{col}_binned"
                    try:
                        if config['method'] == 'Automatic (Quantile)':
                            res = pd.qcut(df_out[col], q=config['params'], labels=False, duplicates='drop')
                        elif config['method'] == 'Manual Ranges':
                            res = pd.cut(df_out[col], bins=config['params'], labels=False, include_lowest=True)
                        
                        # SHIFT LOGIC: NaN=-1 -> 0, Bin0 -> 1, Bin1 -> 2
                        df_out[binned_col_name] = (res.fillna(-1) + 1).astype(int)
                        
                        # DROP ORIGINAL COLUMN
                        df_out = df_out.drop(columns=[col])
                    except: pass
        return df_out

    X_train_binned = apply_binning(X_train_imp)
    X_test_binned = apply_binning(X_test_imp)

    # 4. Pipeline Construction
    current_cols = X_train_binned.columns.tolist()
    
    # Filter selected columns to ensure we don't look for dropped columns
    final_standard = [c for c in cols_standard if c in current_cols]
    final_minmax = [c for c in cols_minmax if c in current_cols]
    final_robust = [c for c in cols_robust if c in current_cols]
    final_log = [c for c in cols_log if c in current_cols]
    final_date = [c for c in cols_date if c in current_cols]
    
    # Handle OneHot logic for binned columns
    final_onehot = []
    for c in cols_onehot:
        binned_name = f"{c}_binned"
        if binned_name in current_cols:
            final_onehot.append(binned_name)
            X_train_binned[binned_name] = X_train_binned[binned_name].astype(str)
            X_test_binned[binned_name] = X_test_binned[binned_name].astype(str)
        elif c in current_cols:
            final_onehot.append(c)
    
    final_ordinal = []
    for c in cols_ordinal:
        binned_name = f"{c}_binned"
        if binned_name in current_cols:
            final_ordinal.append(binned_name)
            X_train_binned[binned_name] = X_train_binned[binned_name].astype(str)
            X_test_binned[binned_name] = X_test_binned[binned_name].astype(str)
        elif c in current_cols:
            final_ordinal.append(c)

    transformers = []

    if final_date: transformers.append(('date_eng', DateFeatureGenerator(), final_date))
    
    if final_log: transformers.append(('log', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one'), final_log))

    if final_standard: transformers.append(('std', StandardScaler(), final_standard))
    if final_minmax: transformers.append(('minmax', MinMaxScaler(), final_minmax))
    if final_robust: transformers.append(('rob', RobustScaler(), final_robust))

    if final_onehot: transformers.append(('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_onehot))
    if final_ordinal: transformers.append(('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), final_ordinal))

    if not transformers:
        main_preprocessor = ColumnTransformer([('all', 'passthrough', current_cols)], verbose_feature_names_out=False)
    else:
        main_preprocessor = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=True)

    X_train_final = main_preprocessor.fit_transform(X_train_binned)
    X_test_final = main_preprocessor.transform(X_test_binned)
    
    poly_model = None
    if use_poly:
        poly = PolynomialFeatures(degree=poly_degree, interaction_only=True, include_bias=False)
        X_train_final = poly.fit_transform(X_train_final)
        X_test_final = poly.transform(X_test_final)
        poly_model = poly

    pca_model = None
    if use_pca:
        pca = PCA(n_components=pca_components)
        X_train_final = pca.fit_transform(X_train_final)
        X_test_final = pca.transform(X_test_final)
        pca_model = pca

    if use_smote and task_type == 'Classification' and not is_ts:
        if HAS_IMBLEARN:
            try:
                smote = SMOTE(random_state=42)
                X_train_final, y_train = smote.fit_resample(X_train_final, y_train)
                st.success(f"SMOTE applied: Rows {X_train_final.shape[0]}")
            except Exception as e: st.warning(f"SMOTE failed: {e}")
        else: st.warning("imbalanced-learn not installed.")
            
    feature_names = current_cols 

    return X_train_final, X_test_final, y_train, y_test, main_preprocessor, le, pca_model, poly_model, feature_names, global_imputer

# --- HELPERS ---
def get_available_models(task_type):
    if task_type == 'Classification':
        return {
            'Logistic Regression': LogisticRegression(), 'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(), 'Naive Bayes': GaussianNB()
        }
    else:
        return {
            'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(), 'SVR': SVR(), 'KNN': KNeighborsRegressor(),
            'Ridge': Ridge(), 'Lasso': Lasso(), 'ElasticNet': ElasticNet()
        }

def get_ensemble_models(base_models, task_type):
    estimators = [(name, model) for name, model in list(base_models.items())[:3]]
    if task_type == 'Classification':
        return {'Voting Classifier': VotingClassifier(estimators=estimators, voting='soft'),
                'Stacking Classifier': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())}
    else:
        return {'Voting Regressor': VotingRegressor(estimators=estimators),
                'Stacking Regressor': StackingRegressor(estimators=estimators, final_estimator=LinearRegression())}

def plot_time_series_results(y_test, y_pred, is_class=False, le=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    y_test_reset = y_test if isinstance(y_test, np.ndarray) else y_test.reset_index(drop=True)
    if is_class and le:
        ax.step(range(len(y_test_reset)), y_test_reset, label='Actual Class', where='mid', color='blue', alpha=0.6)
        ax.step(range(len(y_pred)), y_pred, label='Predicted Class', where='mid', color='orange', alpha=0.6, linestyle='--')
        class_indices = np.arange(len(le.classes_))
        ax.set_yticks(class_indices)
        ax.set_yticklabels(le.classes_)
    else:
        ax.plot(range(len(y_test_reset)), y_test_reset, label='Actual', color='blue')
        ax.plot(range(len(y_pred)), y_pred, label='Predicted', color='orange', linestyle='--')
    ax.legend()
    return fig

# --- CODE GENERATION HELPER ---
def generate_python_code(config, model_name, best_params):
    clean_params = best_params.copy()
    if 'penalty' in clean_params and clean_params['penalty'] != 'elasticnet':
        clean_params.pop('l1_ratio', None)
    
    bin_config_str = str(config.get('binning_config', {}))

    code = f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# --- CUSTOM TRANSFORMERS ---
class RandomSampleImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.saved_values = {{}}
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.saved_values[col] = X[col].dropna().values
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            mask = X[col].isna()
            if mask.any() and col in self.saved_values:
                fill_values = np.random.choice(self.saved_values[col], size=mask.sum())
                X.loc[mask, col] = fill_values
        return X

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            series = pd.to_datetime(X[col], errors='coerce')
            X[f"{{col}}_year"] = series.dt.year
            X[f"{{col}}_month"] = series.dt.month
            X[f"{{col}}_day"] = series.dt.day
            X = X.drop(columns=[col])
        return X.fillna(0)

# --- CONFIGURATION ---
TARGET = '{config['target_col']}'
DROP_COLS = {config['drop_cols']}
BIN_CONFIG = {bin_config_str}

# Selected Columns for Strategies
IMP_MEAN = {config.get('imp_num_mean', [])}
IMP_MEDIAN = {config.get('imp_num_median', [])}
IMP_ZERO = {config.get('imp_num_zero', [])}
IMP_RANDOM = {config.get('imp_num_random', [])} # Hot-Deck
IMP_MODE = {config.get('imp_cat_mode', [])}
IMP_CONST = {config.get('imp_cat_const', [])}

COLS_STANDARD = {config['cols_standard']}
COLS_MINMAX = {config['cols_minmax']}
COLS_ROBUST = {config['cols_robust']}
COLS_ONEHOT_RAW = {config['cols_onehot']}
COLS_ORDINAL_RAW = {config['cols_ordinal']}
COLS_LOG = {config['cols_log']}
COLS_DATE = {config['cols_date']}

# --- LOAD DATA ---
df = pd.read_csv('your_dataset.csv') 

if DROP_COLS: 
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop)

df = df.dropna(subset=[TARGET])
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --- 1. GRANULAR IMPUTATION ---
imp_transformers = []
if IMP_MEAN: imp_transformers.append(('mean', SimpleImputer(strategy='mean'), IMP_MEAN))
if IMP_MEDIAN: imp_transformers.append(('median', SimpleImputer(strategy='median'), IMP_MEDIAN))
if IMP_ZERO: imp_transformers.append(('zero', SimpleImputer(strategy='constant', fill_value=0), IMP_ZERO))
if IMP_RANDOM: imp_transformers.append(('random', RandomSampleImputer(), IMP_RANDOM))
if IMP_MODE: imp_transformers.append(('mode', SimpleImputer(strategy='most_frequent'), IMP_MODE))
if IMP_CONST: imp_transformers.append(('const', SimpleImputer(strategy='constant', fill_value='Missing'), IMP_CONST))

# Auto-fill remaining
all_selected = set(IMP_MEAN + IMP_MEDIAN + IMP_ZERO + IMP_RANDOM + IMP_MODE + IMP_CONST)
remaining = [c for c in X.columns if c not in all_selected]
rem_num = [c for c in remaining if pd.api.types.is_numeric_dtype(X[c])]
rem_cat = [c for c in remaining if c not in rem_num]

if rem_num: imp_transformers.append(('def_num', SimpleImputer(strategy='mean'), rem_num))
if rem_cat: imp_transformers.append(('def_cat', SimpleImputer(strategy='most_frequent'), rem_cat))

imputer = ColumnTransformer(imp_transformers, verbose_feature_names_out=False)
imputer.set_output(transform="pandas")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Imputation
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# --- 2. BINNING ---
for col, cfg in BIN_CONFIG.items():
    if col in X_train.columns:
        b_name = f"{{col}}_binned"
        try:
            if cfg['method'] == 'Automatic (Quantile)':
                X_train[b_name] = pd.qcut(X_train[col], q=cfg['params'], labels=False, duplicates='drop')
                X_test[b_name] = pd.qcut(X_test[col], q=cfg['params'], labels=False, duplicates='drop')
            else:
                X_train[b_name] = pd.cut(X_train[col], bins=cfg['params'], labels=False, include_lowest=True)
                X_test[b_name] = pd.cut(X_test[col], bins=cfg['params'], labels=False, include_lowest=True)
            
            # Shift by 1: NaNs become 0, Bin 0 becomes 1...
            X_train[b_name] = (X_train[b_name].fillna(-1) + 1).astype(int)
            X_test[b_name] = (X_test[b_name].fillna(-1) + 1).astype(int)
            
            # Drop Original
            X_train.drop(columns=[col], inplace=True)
            X_test.drop(columns=[col], inplace=True)
        except: pass

# --- 3. MAIN PIPELINE ---
transformers = []
curr_cols = X_train.columns.tolist()
def filter_c(lst): return [c for c in lst if c in curr_cols]

# Logic to swap user selection for binned cols
def resolve_binned(lst):
    final = []
    for c in lst:
        bn = f"{{c}}_binned"
        if bn in curr_cols: final.append(bn)
        elif c in curr_cols: final.append(c)
    return final

COLS_ONEHOT = resolve_binned(COLS_ONEHOT_RAW)
COLS_ORDINAL = resolve_binned(COLS_ORDINAL_RAW)

if COLS_DATE: transformers.append(('date', DateFeatureGenerator(), filter_c(COLS_DATE)))
if COLS_LOG: transformers.append(('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), filter_c(COLS_LOG)))

if COLS_STANDARD: transformers.append(('std', StandardScaler(), filter_c(COLS_STANDARD)))
if COLS_MINMAX: transformers.append(('minmax', MinMaxScaler(), filter_c(COLS_MINMAX)))
if COLS_ROBUST: transformers.append(('rob', RobustScaler(), filter_c(COLS_ROBUST)))

if COLS_ONEHOT: transformers.append(('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), COLS_ONEHOT))
if COLS_ORDINAL: transformers.append(('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), COLS_ORDINAL))

preprocessor = ColumnTransformer(transformers, remainder='passthrough')

# --- MODEL ---
best_params = {clean_params}
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

model_map = {{
    'Logistic Regression': LogisticRegression, 'Random Forest': RandomForestClassifier, 'XGBoost': XGBClassifier, 'SVM': SVC, 'KNN': KNeighborsClassifier,
    'Linear Regression': LinearRegression, 'Ridge': Ridge, 'Lasso': Lasso, 'ElasticNet': ElasticNet, 'SVR': SVR
}}

ModelClass = model_map.get('{model_name}', RandomForestClassifier)
model = ModelClass(**best_params)

# --- EXECUTION ---
steps = [('preprocessor', preprocessor)]
"""
    if config['use_poly']: code += f"\nsteps.append(('poly', PolynomialFeatures(degree={config['poly_degree']}, interaction_only=True)))"
    if config['use_pca']: code += f"\nsteps.append(('pca', PCA(n_components={config['pca_components']})))"
    
    code += """
steps.append(('model', model))
final_pipeline = Pipeline(steps)

try:
    print("Training Pipeline...")
    final_pipeline.fit(X_train, y_train)
    score = final_pipeline.score(X_test, y_test)
    print(f"Test Score: {score:.4f}")
    print("Success! Pipeline is ready.")
except Exception as e:
    print(f"Error: {e}")
"""
    return code

# --- 4. UI LAYOUT ---

with st.sidebar:
    st.title("🎛️ Project Controls")
    st.write("Contributors: **Wasiful Haque, Md. Jahirul Islam, Saifuddin Yasir**")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): st.session_state.df = pd.read_csv(uploaded_file)
            else: st.session_state.df = pd.read_excel(uploaded_file)
            st.success("Loaded!")
        except Exception as e: st.error(f"Error: {e}")

st.title("🤖 NoCodeML: Advanced ML Studio")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visualization", "⚙️ Preprocessing", "🧠 Training", "📈 Evaluation", "⚡ Tuning", "🔮 Inference"
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
            # FIX: FULL DATASET VISUALIZATION
            st.dataframe(st.session_state.df, height=400, use_container_width=True)
            st.caption("Tip: Hover over the top-right of the table and click the 'Maximize' arrows to see the full screen view.")
            df_viz = st.session_state.df 
        else: st.info("📂 Please upload a dataset in the sidebar to begin.")
        
    elif data_source == "Preprocessed (Train Set)":
        try:
            X_v = st.session_state.X_train
            y_v = st.session_state.y_train
            
            # Name Cleaning
            feat_names = []
            if st.session_state.preprocessor:
                try: raw_names = st.session_state.preprocessor.get_feature_names_out()
                except: raw_names = [f"Feat_{i}" for i in range(X_v.shape[1])]
            else: raw_names = [f"Feat_{i}" for i in range(X_v.shape[1])]

            seen_names = {}
            clean_names = []
            for name in raw_names:
                clean_name = re.sub(r'^\w+__', '', name) 
                if clean_name in seen_names:
                    seen_names[clean_name] += 1
                    clean_names.append(f"{clean_name}_{seen_names[clean_name]}")
                else:
                    seen_names[clean_name] = 0
                    clean_names.append(clean_name)
            feat_names = clean_names

            if st.session_state.poly_model:
                try: feat_names = st.session_state.poly_model.get_feature_names_out(feat_names)
                except: feat_names = [f"Poly_{i}" for i in range(X_v.shape[1])]
            if st.session_state.pca_model: feat_names = [f"PCA_{i}" for i in range(X_v.shape[1])]
            
            if len(feat_names) != X_v.shape[1]: feat_names = [f"Feat_{i}" for i in range(X_v.shape[1])]

            df_viz = pd.DataFrame(X_v, columns=feat_names)
            if y_v is not None:
                t_name = st.session_state.get('target_col', 'Target')
                df_viz[t_name] = y_v
                
            st.dataframe(df_viz, height=400, use_container_width=True)
            
        except Exception as e: st.error(f"Error loading data: {e}")

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
                sns.heatmap(numeric_df.corr(), annot=df_viz.shape[1] < 20, fmt=".2f", cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else: st.warning("No numeric columns.")
        elif viz_mode == "Distribution Plots":
            col = st.selectbox("Select Column", df_viz.columns)
            if col:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df_viz[col], kde=True, ax=ax)
                st.pyplot(fig)
        elif viz_mode == "Deep Profiling (ydata)":
            if st.button("Generate Report"):
                with st.spinner("Analyzing..."):
                    pr = ProfileReport(df_viz, minimal=df_viz.shape[1] > 30, explorative=True)
                    pr.config.html.navbar_show = False
                    components.html(pr.to_html(), height=800, scrolling=True)

# --- TAB 2: PREPROCESSING ---
with tab2:
    if st.session_state.df is not None:
        st.subheader("Pipeline Configuration")
        all_cols = st.session_state.df.columns.tolist()
        
        # 1. Global Drops & Type
        c1, c2, c3 = st.columns(3)
        with c1: drop_cols = st.multiselect("Drop Columns", all_cols)
        remaining_cols = [c for c in all_cols if c not in drop_cols]
        with c2: 
            if remaining_cols: target_col = st.selectbox("Target Column", remaining_cols)
            else: target_col = None
        with c3:
            is_ts = st.checkbox("Time Series Mode")
            task_type = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)

        feature_cols = [c for c in remaining_cols if c != target_col]
        df_features = st.session_state.df[feature_cols]
        auto_num = df_features.select_dtypes(include=np.number).columns.tolist()
        auto_cat = [c for c in feature_cols if c not in auto_num]

        st.divider()
        st.markdown("### 2. Imputation Strategies (Granular)")
        
        i1, i2 = st.columns(2)
        with i1:
            st.caption("Numeric Columns")
            imp_num_mean = st.multiselect("Impute with Mean", auto_num)
            avail_median = [c for c in auto_num if c not in imp_num_mean]
            imp_num_median = st.multiselect("Impute with Median", avail_median)
            avail_zero = [c for c in avail_median if c not in imp_num_median]
            imp_num_zero = st.multiselect("Impute with Zero", avail_zero)
            
            # --- HOT DECKING (RANDOM) ADDED HERE ---
            avail_random = [c for c in avail_zero if c not in imp_num_zero]
            imp_num_random = st.multiselect("Impute with Random (Hot-Deck)", avail_random)
            
        with i2:
            st.caption("Categorical Columns")
            imp_cat_mode = st.multiselect("Impute with Mode (Most Frequent)", auto_cat)
            avail_const = [c for c in auto_cat if c not in imp_cat_mode]
            imp_cat_const = st.multiselect("Impute with 'Missing' Constant", avail_const)

        st.divider()
        st.markdown("### 3. Feature Engineering")
        
        # --- ADVANCED BINNING UI (Dynamic Update) ---
        binning_config = {}
        with st.expander("🧩 Feature Generation", expanded=False):
            fe_col1, fe_col2 = st.columns(2)
            with fe_col1: cols_date = st.multiselect("Extract Date", auto_cat + auto_num)
            with fe_col2: cols_log = st.multiselect("Log Transform", auto_num)
            
            st.divider()
            st.markdown("**Binning / Discretization**")
            st.caption("Creates new `_binned` columns. Originals are DROPPED.")
            avail_bin = [c for c in auto_num if c not in cols_date]
            cols_to_bin = st.multiselect("Select Columns to Bin", avail_bin)
            
            if cols_to_bin:
                bin_method = st.radio("Binning Mode", ["Automatic (Quantile)", "Manual Ranges"], horizontal=True)
                if bin_method == "Automatic (Quantile)":
                    n_bins = st.slider("Number of Bins", 2, 20, 5)
                    for c in cols_to_bin:
                        binning_config[c] = {'method': 'Automatic (Quantile)', 'params': n_bins}
                else:
                    st.caption("Enter edges (e.g., `0, 18, 50, 100`).")
                    for c in cols_to_bin:
                        c_min, c_max = df_features[c].min(), df_features[c].max()
                        def_val = f"{int(c_min)}, {int((c_min+c_max)/2)}, {int(c_max)}"
                        user_edges = st.text_input(f"Edges for '{c}' (Min:{c_min:.1f} Max:{c_max:.1f})", def_val)
                        try:
                            edges = [float(x.strip()) for x in user_edges.split(',')]
                            edges.sort()
                            binning_config[c] = {'method': 'Manual Ranges', 'params': edges}
                        except: st.error(f"Invalid edges for {c}")

        # --- DYNAMIC COLUMN ALLOCATION ---
        derived_bin_cols = [f"{c}_binned" for c in cols_to_bin]
        
        # Remove binned originals from numeric options
        numeric_options = [c for c in auto_num if c not in cols_date and c not in cols_to_bin]
        
        # Add binned to categorical options
        categorical_options = auto_cat + derived_bin_cols

        st.markdown("**Numeric Scaling**")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1: cols_standard = st.multiselect("Standard Scaler", numeric_options)
        with col_t2: cols_minmax = st.multiselect("MinMax Scaler", numeric_options)
        with col_t3: cols_robust = st.multiselect("Robust Scaler", numeric_options)
            
        st.markdown("**Categorical Encoding**")
        col_e1, col_e2 = st.columns(2)
        with col_e1: cols_onehot = st.multiselect("One-Hot Encoding", categorical_options, default=auto_cat)
        with col_e2: cols_ordinal = st.multiselect("Ordinal Encoding", categorical_options, default=derived_bin_cols)

        st.divider()
        st.markdown("### 4. Model Prep")
        col_s2, col_s3 = st.columns(2)
        with col_s2: 
            test_size = st.slider("Test Split", 0.1, 0.5, 0.2, 0.05)
            use_smote = st.checkbox("SMOTE", help="Classif Only")
        with col_s3:
            use_pca = st.checkbox("PCA")
            pca_components = st.slider("PCA Var", 0.5, 0.99, 0.95) if use_pca else 0.95
            use_poly = st.checkbox("Poly Features")
            poly_degree = st.slider("Poly Deg", 2, 3, 2) if use_poly else 2

        date_col_sort = None
        if is_ts: date_col_sort = st.selectbox("Date Sort Col", remaining_cols)

        if st.button("🚀 Process Data"):
            if target_col:
                with st.spinner("Processing..."):
                    try:
                        X_train, X_test, y_train, y_test, prep, le, pca, poly, feat_names, imp_model = preprocess_and_split(
                            st.session_state.df, target_col, task_type, is_ts, date_col_sort,
                            imp_num_mean, imp_num_median, imp_num_zero, imp_num_random,
                            imp_cat_mode, imp_cat_const,
                            test_size, drop_cols,
                            cols_standard, cols_minmax, cols_robust, cols_onehot, cols_ordinal,
                            cols_log, binning_config, cols_date, 
                            use_pca, pca_components,
                            use_poly, poly_degree,
                            use_smote
                        )
                        if X_train is not None:
                            st.session_state.pipeline_config = {
                                'drop_cols': drop_cols, 'target_col': target_col, 'task_type': task_type,
                                'cols_standard': cols_standard, 'cols_minmax': cols_minmax, 'cols_robust': cols_robust,
                                'cols_onehot': cols_onehot, 'cols_ordinal': cols_ordinal,
                                'cols_log': cols_log, 'binning_config': binning_config, 'cols_date': cols_date,
                                'imp_num_mean': imp_num_mean, 'imp_num_median': imp_num_median, 'imp_num_zero': imp_num_zero,
                                'imp_num_random': imp_num_random,
                                'imp_cat_mode': imp_cat_mode, 'imp_cat_const': imp_cat_const,
                                'use_smote': use_smote, 
                                'use_pca': use_pca, 'pca_components': pca_components,
                                'use_poly': use_poly, 'poly_degree': poly_degree
                            }
                            
                            # --- CRITICAL FIX START ---
                            # Clear old model results and tuned models to prevent size mismatch errors
                            st.session_state.model_results = {}  
                            st.session_state.best_model = None
                            st.session_state.best_params = None
                            # --- CRITICAL FIX END ---

                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.target_col = target_col
                            st.session_state.task_type = task_type
                            st.session_state.is_ts = is_ts
                            st.session_state.le = le
                            st.session_state.preprocessor = prep
                            st.session_state.imputer_model = imp_model
                            st.session_state.pca_model = pca
                            st.session_state.poly_model = poly
                            st.session_state.feature_names = feat_names 
                            
                            st.success(f"Processed! Train: {X_train.shape} | Test: {X_test.shape}")
                            st.rerun()
                    except Exception as e: st.error(f"Error: {e}")
            else: st.warning("Select Target.")

# --- TAB 3: TRAINING ---
with tab3:
    if 'X_train' in st.session_state:
        st.subheader("Model Training")
        base_models = get_available_models(st.session_state.task_type)
        ensemble_models = get_ensemble_models(base_models, st.session_state.task_type)
        all_models = {**base_models, **ensemble_models}
        
        selected = st.multiselect("Select Models", list(all_models.keys()), default=list(base_models.keys())[0:2])
        
        if st.button("Train Selected Models"):
            res = {}
            bar = st.progress(0)
            for i, name in enumerate(selected):
                model = all_models[name]
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
                bar.progress((i+1)/len(selected))
            st.session_state.model_results = res
            st.success("Training Complete!")

# --- TAB 4: EVALUATION ---
with tab4:
    if st.session_state.model_results:
        st.subheader("Results")
        res_df = pd.DataFrame([{ "Model": k, v['metric']: v['score']} for k,v in st.session_state.model_results.items()])
        st.table(res_df.sort_values(by=res_df.columns[1], ascending=False))
        
        sel_model = st.selectbox("Visualize Model", list(st.session_state.model_results.keys()))
        preds = st.session_state.model_results[sel_model]['preds']
        y_test = st.session_state.y_test
        
        if st.session_state.is_ts:
            is_class = st.session_state.task_type == 'Classification'
            st.pyplot(plot_time_series_results(y_test, preds, is_class, st.session_state.le))
        elif st.session_state.task_type == "Regression":
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.5); ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            if st.session_state.le:
                names = [str(c) for c in st.session_state.le.classes_]
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', xticklabels=names, yticklabels=names)
            else:
                sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d')
            st.pyplot(fig)
            st.text(classification_report(y_test, preds))

# --- TAB 5: TUNING ---
with tab5:
    if 'X_train' in st.session_state:
        st.subheader("⚡ Automated Tuning")
        
        if not st.session_state.model_results:
            st.warning("⚠️ No models trained yet. Please go to Tab 3 (Training) and train a model first.")
        else:
            trained_models_list = list(st.session_state.model_results.keys())
            tune_model_name = st.selectbox("Select Model to Tune", trained_models_list)
            available_fresh_models = get_available_models(st.session_state.task_type)
            
            model_params_schema = {
                'Logistic Regression': {'C': {'type': 'float_list', 'default': '0.1, 1.0, 10.0', 'label': 'C'}, 'solver': {'type': 'cat_list', 'options': ['saga', 'liblinear'], 'default': ['saga'], 'label': 'Solver'}, 'penalty': {'type': 'cat_list', 'options': ['l2', 'l1'], 'default': ['l2'], 'label': 'Penalty'}},
                'Random Forest': {'n_estimators': {'type': 'int_list', 'default': '50, 100', 'label': 'Trees'}, 'max_depth': {'type': 'int_list', 'default': '10, 20', 'label': 'Depth'}},
                'XGBoost': {'n_estimators': {'type': 'int_list', 'default': '50, 100', 'label': 'Trees'}, 'learning_rate': {'type': 'float_list', 'default': '0.01, 0.1', 'label': 'Rate'}, 'max_depth': {'type': 'int_list', 'default': '3, 6', 'label': 'Depth'}},
                'SVM': {'C': {'type': 'float_list', 'default': '0.1, 1', 'label': 'C'}, 'kernel': {'type': 'cat_list', 'options': ['linear', 'rbf'], 'default': ['rbf'], 'label': 'Kernel'}},
                'KNN': {'n_neighbors': {'type': 'int_list', 'default': '3, 5', 'label': 'Neighbors'}},
                'Linear Regression': {'fit_intercept': {'type': 'cat_list', 'options': [True, False], 'default': [True], 'label': 'Intercept'}},
                'Ridge': {'alpha': {'type': 'float_list', 'default': '0.1, 1.0', 'label': 'Alpha'}},
                'Lasso': {'alpha': {'type': 'float_list', 'default': '0.1, 1.0', 'label': 'Alpha'}}
            }

            current_schema = model_params_schema.get(tune_model_name, {})
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
                                if config['type'] == 'int_list': parsed_vals.append(int(v))
                                else: parsed_vals.append(float(v))
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
                        base_model = available_fresh_models[tune_model_name]
                        cv = TimeSeriesSplit(n_splits=3) if st.session_state.is_ts else 3
                        try:
                            if search_type == "Grid Search": search = GridSearchCV(base_model, final_param_grid, cv=cv, verbose=1, n_jobs=-1, error_score='raise')
                            else: search = RandomizedSearchCV(base_model, final_param_grid, n_iter=n_iter, cv=cv, verbose=1, n_jobs=-1, error_score='raise')
                            
                            search.fit(st.session_state.X_train, st.session_state.y_train)
                            st.success("Tuning Complete!")
                            st.session_state.best_model = search.best_estimator_
                            st.session_state.best_params = search.best_params_
                            
                            m1, m2 = st.columns(2)
                            m1.metric("Best CV Score", f"{search.best_score_:.4f}")
                            m2.write("**Best Parameters:**")
                            m2.json(search.best_params_)
                        except Exception as e: st.error(f"Tuning Failed: {e}")

# --- TAB 6: INFERENCE ---
with tab6:
    st.subheader("🔮 Inference & Export")
    
    if st.session_state.best_model is None: st.warning("Tune a model in Tab 5 first.")
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
                original_feats = st.session_state.feature_names
                for i, col_name in enumerate(original_feats):
                    with cols[i % 2]:
                        input_data[col_name] = st.text_input(f"{col_name}", key=f"in_{i}")
                if st.form_submit_button("Generate"):
                    st.session_state.inference_data = pd.DataFrame([input_data])
                    st.success("Generated!")

        if st.session_state.inference_data is not None:
            st.divider()
            st.write("Preview:", st.session_state.inference_data.head())
            if st.button("Run Prediction", key="run_pred"):
                try:
                    # Pipeline: Impute -> Bin -> Preprocess -> Model
                    X_new_raw = st.session_state.inference_data.copy()
                    
                    # 1. Global Impute
                    if st.session_state.imputer_model:
                        X_new_imp = st.session_state.imputer_model.transform(X_new_raw)
                    else:
                        X_new_imp = X_new_raw 
                        
                    # 2. Manual Binning
                    cfg = st.session_state.get('pipeline_config', {})
                    bin_cfg = cfg.get('binning_config', {})
                    
                    if bin_cfg:
                        for col, c_conf in bin_cfg.items():
                            if col in X_new_imp.columns:
                                b_name = f"{col}_binned"
                                try:
                                    if c_conf['method'] == 'Automatic (Quantile)':
                                        res = pd.qcut(X_new_imp[col], q=c_conf['params'], labels=False, duplicates='drop')
                                    else:
                                        res = pd.cut(X_new_imp[col], bins=c_conf['params'], labels=False, include_lowest=True)
                                    
                                    # Shift by +1
                                    X_new_imp[b_name] = (res.fillna(-1) + 1).astype(int)
                                    # Drop Original
                                    X_new_imp = X_new_imp.drop(columns=[col])
                                except: pass

                    # 3. Main Transform
                    X_new = st.session_state.preprocessor.transform(X_new_imp)
                    if st.session_state.poly_model: X_new = st.session_state.poly_model.transform(X_new)
                    if st.session_state.pca_model: X_new = st.session_state.pca_model.transform(X_new)
                    
                    preds = st.session_state.best_model.predict(X_new)
                    if st.session_state.le: final_preds = st.session_state.le.inverse_transform(preds)
                    else: final_preds = preds
                    st.write("Prediction:", final_preds)
                except Exception as e: st.error(f"Error: {e}")

    st.divider()
    st.markdown("#### 2. Export Code 💾")
    if st.session_state.best_model is not None and 'pipeline_config' in st.session_state:
        if st.button("Generate Training Script"):
            ui_name = "Random Forest"
            script = generate_python_code(st.session_state.pipeline_config, ui_name, st.session_state.best_params)
            st.code(script, language='python')
