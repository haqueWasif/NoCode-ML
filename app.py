import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

# Profiling
from ydata_profiling import ProfileReport

# Scikit-Learn & Modeling
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="NoCode ML Studio", layout="wide", page_icon="🤖")

# Initialize Session State
if 'df' not in st.session_state: st.session_state.df = None
if 'model_results' not in st.session_state: st.session_state.model_results = {}

# --- 2. CORE LOGIC ---

def preprocess_and_split(df, target_col, task_type, is_ts, date_col, 
                         impute_strat, scale_method, test_size):
    """
    Advanced Preprocessing with user-selected methods.
    """
    # 1. Handling Time Series Sorting
    if is_ts and date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        X = df.drop(columns=[target_col, date_col])
    else:
        X = df.drop(columns=[target_col])
        
    y = df[target_col]

    # 2. Detect Columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Dynamic Pipeline Construction
    
    # -- Numerical Pipeline --
    num_steps = []
    # Imputation
    if impute_strat == "Mean":
        num_steps.append(('imputer', SimpleImputer(strategy='mean')))
    elif impute_strat == "Median":
        num_steps.append(('imputer', SimpleImputer(strategy='median')))
    elif impute_strat == "Zero":
        num_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
    
    # Scaling
    if scale_method == "StandardScaler":
        num_steps.append(('scaler', StandardScaler()))
    elif scale_method == "MinMaxScaler":
        num_steps.append(('scaler', MinMaxScaler()))
    elif scale_method == "RobustScaler":
        num_steps.append(('scaler', RobustScaler()))
    
    num_transformer = Pipeline(steps=num_steps)
    
    # -- Categorical Pipeline --
    cat_steps = []
    cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
    cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
    cat_transformer = Pipeline(steps=cat_steps)
    
    # Combine
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    # 4. Target Encoding
    le = None
    if task_type == 'Classification' and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 5. Apply Transformations
    X_processed = preprocessor.fit_transform(X)

    # 6. Split Data
    if is_ts:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42)
        
    return X_train, X_test, y_train, y_test, preprocessor, le

def get_available_models(task_type):
    if task_type == 'Classification':
        return {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'SVM': SVC()
        }
    else:
        return {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'SVR': SVR()
        }

def plot_time_series_forecast(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(12, 6))
    y_test_reset = y_test if isinstance(y_test, np.ndarray) else y_test.reset_index(drop=True)
    ax.plot(range(len(y_test_reset)), y_test_reset, label='Actual', color='blue')
    ax.plot(range(len(y_pred)), y_pred, label='Predicted', color='orange', linestyle='--')
    ax.set_title('Time Series Forecast')
    ax.legend()
    return fig

# --- 3. UI LAYOUT ---

with st.sidebar:
    st.title("🎛️ Project Controls")
    st.write("User: **Wasiful Haque**")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): st.session_state.df = pd.read_csv(uploaded_file)
            else: st.session_state.df = pd.read_excel(uploaded_file)
            st.success("Loaded!")
        except Exception as e: st.error(f"Error: {e}")

st.title("🤖 NoCodeML: Advanced ML Studio")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "⚙️ Preprocessing", "🧠 Training", "📈 Evaluation"])

# --- TAB 1: VISUALIZATION CONTROLS ---
with tab1:
    if st.session_state.df is not None:
        st.subheader("Explore Your Data")
        
        # New: Visualization Mode Selector
        viz_mode = st.radio("Choose Visualization Type:", 
                           ["Quick Overview", "Correlation Heatmap", "Distribution Plots", "Deep Profiling (ydata)"], 
                           horizontal=True)
        
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
            else:
                st.warning("No numeric columns for correlation.")

        elif viz_mode == "Distribution Plots":
            col_to_plot = st.selectbox("Select Column", numeric_df.columns)
            if col_to_plot:
                fig, ax = plt.subplots()
                sns.histplot(df[col_to_plot], kde=True, ax=ax)
                st.pyplot(fig)

        elif viz_mode == "Deep Profiling (ydata)":
            if st.button("Generate Full Report"):
                with st.spinner("Generating report..."):
                    pr = ProfileReport(df, explorative=True)
                    report_html = pr.to_html()
                    components.html(report_html, height=800, scrolling=True)
    else:
        st.info("Upload data to visualize.")

# --- TAB 2: PREPROCESSING CONTROLS ---
with tab2:
    if st.session_state.df is not None:
        st.subheader("Build Your Pipeline")
        
        c1, c2, c3 = st.columns(3)
        target_col = c1.selectbox("Target Column", st.session_state.df.columns)
        is_ts = c2.checkbox("Time Series Mode")
        
        if is_ts:
            task_type = "Regression"
            c3.markdown("**Task:** Regression (Time Series)")
        else:
            task_type = c3.radio("Task Type", ["Classification", "Regression"])

        # NEW: Advanced Controls
        st.divider()
        st.write("🔧 **Advanced Settings**")
        
        col_sets1, col_sets2, col_sets3 = st.columns(3)
        
        with col_sets1:
            impute_strat = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Zero"])
        with col_sets2:
            scale_method = st.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
        with col_sets3:
            test_size = st.slider("Test Split Size", 0.1, 0.5, 0.2, 0.05)

        date_col = None
        if is_ts:
            date_col = st.selectbox("Date Column", st.session_state.df.columns)

        if st.button("🚀 Process Data"):
            with st.spinner("Processing..."):
                try:
                    # Pass the new arguments to the function
                    X_train, X_test, y_train, y_test, prep, le = preprocess_and_split(
                        st.session_state.df, target_col, task_type, is_ts, date_col,
                        impute_strat, scale_method, test_size
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.task_type = task_type
                    st.session_state.is_ts = is_ts
                    
                    st.success(f"Processed! Train: {X_train.shape} | Test: {X_test.shape}")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 3 & 4 (Training/Eval) ---
# (Logic remains largely the same, just compressed for brevity)
with tab3:
    if 'X_train' in st.session_state:
        st.subheader("Train Models")
        models = get_available_models(st.session_state.task_type)
        selected = st.multiselect("Select Models", list(models.keys()), default=list(models.keys())[0])
        
        if st.button("Train"):
            res = {}
            bar = st.progress(0)
            for i, name in enumerate(selected):
                model = models[name]
                model.fit(st.session_state.X_train, st.session_state.y_train)
                preds = model.predict(st.session_state.X_test)
                
                if st.session_state.task_type == 'Classification':
                    score = accuracy_score(st.session_state.y_test, preds)
                    metric = "Accuracy"
                else:
                    score = r2_score(st.session_state.y_test, preds)
                    metric = "R2 Score"
                
                res[name] = {"model": model, "score": score, "preds": preds, "metric": metric}
                bar.progress((i+1)/len(selected))
            st.session_state.model_results = res
            st.success("Done!")

with tab4:
    if st.session_state.model_results:
        st.subheader("Evaluation")
        res_df = pd.DataFrame([{ "Model": k, v['metric']: v['score']} for k,v in st.session_state.model_results.items()])
        st.table(res_df.sort_values(by=res_df.columns[1], ascending=False))
        
        sel_model = st.selectbox("Visualize Model", list(st.session_state.model_results.keys()))
        preds = st.session_state.model_results[sel_model]['preds']
        y_test = st.session_state.y_test
        
        if st.session_state.is_ts:
            st.pyplot(plot_time_series_forecast(y_test, preds))
        elif st.session_state.task_type == "Regression":
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
