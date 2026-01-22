import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import ast

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

# Imbalanced Learn (Try/Except for robustness)
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
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
if 'pca_model' not in st.session_state: st.session_state.pca_model = None
if 'poly_model' not in st.session_state: st.session_state.poly_model = None
if 'best_model' not in st.session_state: st.session_state.best_model = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = []

# --- 2. CUSTOM TRANSFORMERS (FEATURE ENGINEERING) ---

### FEATURE ENGINEERING START ###
class RandomSampleImputer(BaseEstimator, TransformerMixin):
    """Hot-Deck Imputation: Fills NaNs with random values from the train set."""
    def __init__(self):
        self.saved_values = {}

    def fit(self, X, y=None):
        # Store non-nan values for each column to sample from later
        X = pd.DataFrame(X)
        for col in X.columns:
            self.saved_values[col] = X[col].dropna().values
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            mask = X[col].isna()
            if mask.any() and col in self.saved_values and len(self.saved_values[col]) > 0:
                # Sample random values to fill the nans
                fill_values = np.random.choice(self.saved_values[col], size=mask.sum())
                X.loc[mask, col] = fill_values
        return X.values

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    """Extracts features from datetime columns."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            # Force conversion to datetime
            series = pd.to_datetime(X[col], errors='coerce')
            X[f"{col}_year"] = series.dt.year
            X[f"{col}_month"] = series.dt.month
            X[f"{col}_day"] = series.dt.day
            X[f"{col}_dow"] = series.dt.dayofweek
            # Drop original date column as models can't handle it
            X = X.drop(columns=[col])
        # Fill NaNs generated from coercion with -1 or median (simplified here to 0)
        return X.fillna(0)
### FEATURE ENGINEERING END ###

# --- 3. CORE LOGIC ---

def build_advanced_pipeline(X, num_impute_strat, cat_impute_strat,
                            cols_standard, cols_minmax, cols_robust,
                            cols_onehot, cols_ordinal,
                            cols_log, cols_binning, cols_date, binning_n_bins):
    transformers = []

    ### FEATURE ENGINEERING START ###
    # 0. Date Features (Happens first)
    if cols_date:
        transformers.append(('date_eng', DateFeatureGenerator(), cols_date))

    # 1. Log Transform Group
    if cols_log:
        # Log1p is safe for zeros, but we assume non-negative for simplicity
        transformers.append(('num_log', FunctionTransformer(np.log1p, validate=False), cols_log))

    # 2. Binning (Discretization) - Converts Numeric to Categorical (Ordinal)
    if cols_binning:
        transformers.append(('num_bin', KBinsDiscretizer(n_bins=binning_n_bins, encode='ordinal', strategy='quantile'), cols_binning))
    ### FEATURE ENGINEERING END ###

    # --- NUMERIC PIPELINES ---
    # Helper to get imputer based on selection
    def get_imputer(strat):
        if strat == "Mean": return SimpleImputer(strategy='mean')
        elif strat == "Median": return SimpleImputer(strategy='median')
        elif strat == "Zero": return SimpleImputer(strategy='constant', fill_value=0)
        elif strat == "Hot-Deck (Random)": return RandomSampleImputer()
        return SimpleImputer(strategy='mean')

    if cols_standard:
        steps = [('imputer', get_imputer(num_impute_strat)), ('scaler', StandardScaler())]
        transformers.append(('num_standard', Pipeline(steps), cols_standard))
    if cols_minmax:
        steps = [('imputer', get_imputer(num_impute_strat)), ('scaler', MinMaxScaler())]
        transformers.append(('num_minmax', Pipeline(steps), cols_minmax))
    if cols_robust:
        steps = [('imputer', get_imputer(num_impute_strat)), ('scaler', RobustScaler())]
        transformers.append(('num_robust', Pipeline(steps), cols_robust))

    # --- CATEGORICAL PIPELINES ---
    if cols_onehot:
        steps = [('imputer', SimpleImputer(strategy='most_frequent' if cat_impute_strat == 'Mode' else 'constant')),
                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))] # Force dense for stability
        transformers.append(('cat_onehot', Pipeline(steps), cols_onehot))
    if cols_ordinal:
        steps = [('imputer', SimpleImputer(strategy='most_frequent' if cat_impute_strat == 'Mode' else 'constant')),
                 ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]
        transformers.append(('cat_ordinal', Pipeline(steps), cols_ordinal))

    if not transformers:
        return ColumnTransformer(transformers=[('identity', 'passthrough', X.columns)], verbose_feature_names_out=False)

    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0, verbose_feature_names_out=False)

def preprocess_and_split(df, target_col, task_type, is_ts, date_col_sort, 
                         num_impute_strat, cat_impute_strat, test_size, drop_cols,
                         cols_standard, cols_minmax, cols_robust, cols_onehot, cols_ordinal,
                         cols_log, cols_binning, cols_date, binning_n_bins,
                         use_pca, pca_components,
                         use_poly, poly_degree,
                         use_smote):
    
    # 0. Drop Columns
    if drop_cols: df = df.drop(columns=drop_cols)
    
    # Drop rows with missing Target
    df = df.dropna(subset=[target_col])
        
    # 1. Sort for Time Series
    if is_ts and date_col_sort:
        df[date_col_sort] = pd.to_datetime(df[date_col_sort])
        df = df.sort_values(by=date_col_sort)
        X = df.drop(columns=[target_col, date_col_sort])
    else:
        X = df.drop(columns=[target_col])
    y = df[target_col]

    # Type Enforcement
    # Note: Logic slightly modified to allow overlap for Feature Engineering (e.g. Log + Scaler)
    # But for safety in this NoCode app, we enforce numeric on Scaler cols
    all_scale_cols = cols_standard + cols_minmax + cols_robust + cols_log
    for col in all_scale_cols:
        if col in X.columns: X[col] = pd.to_numeric(X[col], errors='coerce')
            
    all_cat_cols = cols_onehot + cols_ordinal
    for col in all_cat_cols:
        if col in X.columns: X[col] = X[col].astype(str)

    # 2. Target Encoding
    le = None
    if task_type == 'Classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = pd.to_numeric(y, errors='coerce')
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

    # 3. Split
    if is_ts:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    else:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 4. Build Pipeline
    preprocessor = build_advanced_pipeline(X, num_impute_strat, cat_impute_strat,
                                           cols_standard, cols_minmax, cols_robust,
                                           cols_onehot, cols_ordinal,
                                           cols_log, cols_binning, cols_date, binning_n_bins)
    
    # 5. Apply Transformations (Fit on Train, Transform Test)
    # This generates the Feature Engineered numpy arrays
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    ### FEATURE ENGINEERING START (Post-Preprocessing) ###
    
    # A. Polynomial Features
    poly_model = None
    if use_poly:
        # Warning: interaction_only=True prevents explosion
        poly = PolynomialFeatures(degree=poly_degree, interaction_only=True, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        poly_model = poly

    # B. PCA
    pca_model = None
    if use_pca:
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        pca_model = pca

    # C. SMOTE (Resampling - Train Only)
    if use_smote and task_type == 'Classification' and not is_ts:
        if HAS_IMBLEARN:
            try:
                # SMOTE works on numeric data (which X_train is now)
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                st.success(f"SMOTE applied: Training size increased to {X_train.shape[0]}")
            except Exception as e:
                st.warning(f"SMOTE failed (likely sparse matrix or class size issue): {e}")
        else:
            st.warning("imbalanced-learn library not installed. SMOTE skipped.")
    elif use_smote:
        st.warning("SMOTE skipped. (Disabled for Regression or Time Series)")
        
    ### FEATURE ENGINEERING END ###
        
    return X_train, X_test, y_train, y_test, preprocessor, le, pca_model, poly_model

# ... [Keep helper functions get_default_hyperparams, get_available_models, get_ensemble_models, plot_time_series_results from previous code] ...
# (Included here for completeness)

def get_default_hyperparams(model_name, task_type):
    if task_type == "Classification":
        if model_name == "Logistic Regression": return "{'C': [0.1, 1, 10], 'penalty': ['l2']}"
        elif "Random Forest" in model_name: return "{'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}"
        elif "XGBoost" in model_name: return "{'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}"
        elif "SVM" in model_name: return "{'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}"
        elif "KNN" in model_name: return "{'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}"
    else: 
        if model_name == "Linear Regression": return "{}"
        elif "Random Forest" in model_name: return "{'n_estimators': [50, 100], 'max_depth': [None, 10]}"
        elif "XGBoost" in model_name: return "{'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}"
        elif "SVR" in model_name: return "{'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}"
        elif "KNN" in model_name: return "{'n_neighbors': [3, 5, 7]}"
    return "{}"

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
    if st.session_state.df is not None:
        st.subheader("Explore Your Data")
        viz_mode = st.radio("Choose Visualization:", ["Quick Overview", "Correlation Heatmap", "Distribution Plots", "Deep Profiling (ydata)"], horizontal=True)
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=np.number)
        
        if viz_mode == "Quick Overview":
            st.dataframe(df.head())
            st.write(df.describe())
        elif viz_mode == "Correlation Heatmap":
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else: st.warning("No numeric columns.")
        elif viz_mode == "Distribution Plots":
            col = st.selectbox("Select Column", numeric_df.columns)
            if col:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)
        elif viz_mode == "Deep Profiling (ydata)":
            if st.button("Generate Report"):
                with st.spinner("Analyzing..."):
                    pr = ProfileReport(df, explorative=True)
                    pr.config.html.navbar_show = False
                    components.html(pr.to_html(), height=800, scrolling=True)
    else: st.info("Upload data to start.")

# --- TAB 2: PREPROCESSING ---
with tab2:
    if st.session_state.df is not None:
        st.subheader("Pipeline Configuration")
        all_cols = st.session_state.df.columns.tolist()
        
        # 1. Drop
        drop_cols = st.multiselect("Select Columns to Drop", all_cols)
        remaining_cols = [c for c in all_cols if c not in drop_cols]
        
        # 2. Target & Task
        c1, c2, c3 = st.columns(3)
        if remaining_cols: target_col = c1.selectbox("Target Column", remaining_cols)
        else: target_col = None
        feature_cols = [c for c in remaining_cols if c != target_col]
        
        is_ts = c2.checkbox("Time Series Mode")
        if is_ts:
            c3.markdown("**Time Series Active**")
            task_type = c3.radio("Task Type", ["Regression", "Classification"], help="Classification for Up/Down trend")
        else:
            task_type = c3.radio("Task Type", ["Classification", "Regression"])

        # 3. Feature Selection
        st.divider()
        st.markdown("### 2. Feature Processing")
        
        # Auto-detect types
        df_features = st.session_state.df[feature_cols]
        auto_num = df_features.select_dtypes(include=np.number).columns.tolist()
        auto_cat = [c for c in feature_cols if c not in auto_num]

        # A. Feature Engineering Expander
        ### FEATURE ENGINEERING START ###
        with st.expander("🧩 Feature Engineering (Domain & Generation)", expanded=False):
            st.caption("Apply these before scaling/encoding.")
            fe_col1, fe_col2, fe_col3 = st.columns(3)
            with fe_col1:
                cols_date = st.multiselect("Extract Date Features (Day/Month/Year)", auto_cat + auto_num)
            with fe_col2:
                cols_log = st.multiselect("Log Transform (Skewed Data)", auto_num)
            with fe_col3:
                cols_binning = st.multiselect("Binning (Numeric -> Categories)", auto_num)
                binning_n_bins = st.slider("Number of Bins", 3, 10, 5)
        ### FEATURE ENGINEERING END ###

        # B. Standard Processing
        st.markdown("**Numeric Scaling**")
        # Remove cols used in Feature Engineering from defaults to avoid errors
        used_in_fe = cols_date + cols_binning
        remaining_num = [c for c in auto_num if c not in used_in_fe]
        remaining_cat = [c for c in auto_cat if c not in used_in_fe]

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1: cols_standard = st.multiselect("Standard Scaler", feature_cols, default=remaining_num)
        with col_t2: cols_minmax = st.multiselect("MinMax Scaler", feature_cols, default=[])
        with col_t3: cols_robust = st.multiselect("Robust Scaler", feature_cols, default=[])
            
        st.markdown("**Categorical Encoding**")
        col_e1, col_e2 = st.columns(2)
        with col_e1: cols_onehot = st.multiselect("One-Hot Encoding", feature_cols, default=remaining_cat)
        with col_e2: cols_ordinal = st.multiselect("Ordinal Encoding", feature_cols, default=[])

        # 4. Settings
        st.divider()
        st.markdown("### 3. Settings & Decompositions")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: 
            num_impute_strat = st.selectbox("Numeric Imputation", ["Mean", "Median", "Zero", "Hot-Deck (Random)"])
            cat_impute_strat = st.selectbox("Categorical Imputation", ["Mode", "Constant"])
        with col_s2: 
            test_size = st.slider("Test Split", 0.1, 0.5, 0.2, 0.05)
            use_smote = st.checkbox("Use SMOTE (Fix Imbalance)", help="Only for Classification. Disabled in Time Series.")
        with col_s3:
            st.write("**Decomposition / Interactions**")
            use_pca = st.checkbox("Apply PCA")
            pca_components = st.slider("PCA Variance", 0.5, 0.99, 0.95) if use_pca else 0.95
            use_poly = st.checkbox("Polynomial Features (x²)")
            poly_degree = st.slider("Degree", 2, 3, 2) if use_poly else 2

        date_col_sort = None
        if is_ts: date_col_sort = st.selectbox("Date Column (Sorting)", remaining_cols)

        if st.button("🚀 Process Data"):
            if target_col:
                with st.spinner("Processing..."):
                    try:
                        X_train, X_test, y_train, y_test, prep, le, pca, poly = preprocess_and_split(
                            st.session_state.df, target_col, task_type, is_ts, date_col_sort,
                            num_impute_strat, cat_impute_strat, test_size, drop_cols,
                            cols_standard, cols_minmax, cols_robust, cols_onehot, cols_ordinal,
                            cols_log, cols_binning, cols_date, binning_n_bins,
                            use_pca, pca_components,
                            use_poly, poly_degree,
                            use_smote
                        )
                        if X_train is not None:
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.task_type = task_type
                            st.session_state.is_ts = is_ts
                            st.session_state.le = le
                            st.session_state.preprocessor = prep
                            st.session_state.pca_model = pca
                            st.session_state.poly_model = poly
                            
                            # Keep track of column names is hard after complex transformation, 
                            # so we save the Raw Input Features for the inference form
                            # NOTE: This is a simplification. Real feature mapping is complex.
                            st.session_state.feature_names = feature_cols
                            
                            st.success(f"Processed! Train: {X_train.shape} | Test: {X_test.shape}")
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
        tunable_models = get_available_models(st.session_state.task_type)
        tune_model_name = st.selectbox("Select Model", list(tunable_models.keys()))
        
        st.markdown("##### Edit Hyperparameters")
        default_params = get_default_hyperparams(tune_model_name, st.session_state.task_type)
        param_str = st.text_area("Parameter Grid (JSON)", value=default_params, height=100)
        
        try: param_grid = ast.literal_eval(param_str)
        except: param_grid = {}; st.error("Invalid Syntax")

        search_type = st.radio("Method", ["Random Search", "Grid Search"])
        n_iter = st.slider("Iterations", 5, 50, 10) if search_type == "Random Search" else 0
        
        if st.button("Start Tuning"):
            if not param_grid: st.error("Empty Grid")
            else:
                with st.spinner(f"Tuning {tune_model_name}..."):
                    base_model = tunable_models[tune_model_name]
                    cv = TimeSeriesSplit(n_splits=3) if st.session_state.is_ts else 3
                    try:
                        if search_type == "Grid Search": search = GridSearchCV(base_model, param_grid, cv=cv, verbose=1, n_jobs=-1)
                        else: search = RandomizedSearchCV(base_model, param_grid, n_iter=n_iter, cv=cv, verbose=1, n_jobs=-1)
                        search.fit(st.session_state.X_train, st.session_state.y_train)
                        st.success(f"Best: {search.best_params_}")
                        st.metric("CV Score", f"{search.best_score_:.4f}")
                        st.session_state.best_model = search.best_estimator_
                    except Exception as e: st.error(f"Failed: {e}")

# --- TAB 6: INFERENCE ---
with tab6:
    st.subheader("🔮 Make Predictions")
    if st.session_state.best_model is None: st.warning("Tune a model first.")
    elif st.session_state.preprocessor is None: st.warning("No Pipeline.")
    else:
        input_method = st.radio("Input Method", ["Upload CSV", "Manual Entry"])
        if 'inference_data' not in st.session_state: st.session_state.inference_data = None

        if input_method == "Upload CSV":
            inf_file = st.file_uploader("New Data (CSV)", type=["csv"])
            if inf_file: st.session_state.inference_data = pd.read_csv(inf_file); st.success("Loaded!")
        else:
            with st.form("manual_entry_form"):
                input_data = {}
                cols = st.columns(2)
                for i, col_name in enumerate(st.session_state.feature_names):
                    # Attempt to get type
                    col_type = object
                    if st.session_state.df is not None and col_name in st.session_state.df.columns:
                        col_type = st.session_state.df[col_name].dtype
                    with cols[i % 2]:
                        if pd.api.types.is_numeric_dtype(col_type):
                            input_data[col_name] = st.number_input(f"{col_name}", value=0.0, key=f"in_{i}")
                        else:
                            input_data[col_name] = st.text_input(f"{col_name}", key=f"in_{i}")
                if st.form_submit_button("Generate"):
                    st.session_state.inference_data = pd.DataFrame([input_data])
                    st.success("Generated!")

        if st.session_state.inference_data is not None:
            st.divider()
            st.write("Preview:", st.session_state.inference_data.head())
            if st.button("Run Prediction", key="run_pred"):
                try:
                    # 1. Preprocess
                    X_new = st.session_state.preprocessor.transform(st.session_state.inference_data)
                    # 2. Poly
                    if st.session_state.poly_model: X_new = st.session_state.poly_model.transform(X_new)
                    # 3. PCA
                    if st.session_state.pca_model: X_new = st.session_state.pca_model.transform(X_new)
                    
                    preds = st.session_state.best_model.predict(X_new)
                    if st.session_state.le: final_preds = st.session_state.le.inverse_transform(preds)
                    else: final_preds = preds
                    
                    st.write("Prediction:", final_preds)
                except Exception as e: st.error(f"Error: {e}")
