import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
st.set_page_config(page_title="Credit risk project", layout="centered")

# dictionary and functions below are copied from notebook
imput_values={'loan_amnt': 12000.0, 'term': 36, 'fico_range_low': 690.0, 
 'fico_range_high': 694.0, 'pub_rec': 0, 'emp_length': 10, 
 'home_ownership': 'MORTGAGE', 'annual_inc': 65000.0,
   'purpose': 'debt_consolidation', 'dti': 17.67, 'revol_util': 52.1, 'mort_acc': 1.0, 
   'earliest_cr_line': 2001, 'open_acc': 11.0, 'issue_d': 2015, 'credit_hist_years': 13}

def cleaner(data):
    df = data.copy()
    
    if 'mort_acc' in df.columns: 
        df['mort_acc'] = df['mort_acc'].fillna(0)
    if 'emp_length' in df.columns: 
        df['emp_length'] = df['emp_length'].fillna(0)
    
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        df['issue_d'] = df['issue_d'].apply(lambda x: int(x.split('-')[1]) if type(x)==str else np.nan)
        df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x: int(x.split('-')[1]) if type(x)==str else np.nan)
        df['credit_hist_years'] = df['issue_d'] - df['earliest_cr_line']
        df = df.drop(columns=['issue_d', 'earliest_cr_line'])

    if 'emp_length' in df.columns:
        if df['emp_length'].dtype == 'O' or df['emp_length'].dtype == 'string':
            df['emp_length'] = df['emp_length'].astype(str).str.replace(r'years?|<|\+', '', regex=True).str.strip()
        df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')
    
    if 'term' in df.columns:
        if df['term'].dtype == 'O' or df['term'].dtype == 'string':
            df['term'] = df['term'].str.replace('months', '').str.strip()
        df['term'] = pd.to_numeric(df['term'], errors='coerce')

    for col in df.columns:
        if col in imput_values:
            df[col] = df[col].fillna(imput_values[col])

    if 'annual_inc' in df.columns: 
        df['annual_inc'] = df['annual_inc'].clip(upper=300000)
    if 'pub_rec' in df.columns:
        df['pub_rec'] = (df['pub_rec'] > 0).astype(int)
    if 'dti' in df.columns:
        df['dti'] = df['dti'].clip(lower=0, upper=60)
    if 'revol_util' in df.columns:
        df['revol_util'] = df['revol_util'].clip(upper=150)
    if 'open_acc' in df.columns:
        df['open_acc'] = df['open_acc'].clip(upper=39)
    if 'mort_acc' in df.columns:
        df['mort_acc'] = df['mort_acc'].clip(upper=10)
    if 'credit_hist_years' in df.columns:
        df['credit_hist_years'] = df['credit_hist_years'].clip(upper=59)

    if 'home_ownership' in df.columns:
        to_keep = ['MORTGAGE', 'RENT', 'OWN']
        df['home_ownership'] = df['home_ownership'].apply(lambda x: x if x in to_keep else 'MORTGAGE')

    if 'purpose' in df.columns:
        to_merge = ['vacation', 'wedding', 'renewable_energy', 'educational', 'moving', 'house']
        df.loc[df['purpose'].isin(to_merge), 'purpose']='other'
    
    return df

def engineer(data):
    df = data.copy()
    
    if 'term' in df.columns:
        if df['term'].isin([36, 60]).any():
            df['term'] = df['term'].apply(lambda x: 0 if x == 36 else 1)
            
    if 'fico_range_high' in df.columns and 'fico_range_low' in df.columns:
        df['fico'] = (df['fico_range_high'] + df['fico_range_low']) / 2
        df = df.drop(['fico_range_high', 'fico_range_low'], axis=1)
        
    to_log = ['annual_inc', 'mort_acc', 'fico', 'open_acc', 'credit_hist_years']
    for col in to_log:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            
    return df

#  loading data and model
@st.cache_data
def load_model():
    return joblib.load('credit_model.pkl')
model = load_model()


def load_data():
    return pd.read_csv('train_small.csv')
data = load_data()


def tab_calculator():
    if 'history_data' not in st.session_state:
        st.session_state['history_data'] = pd.DataFrame()

    with st.form("loan_form"):
        st.header('💲 Calculator')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Financial")
            loan_amnt = st.number_input("Loan amount ($)", min_value=1000, max_value=50000, value=15000, step=500, help="Listed amount of the loan.")
            annual_inc = st.number_input("Annual income ($)", min_value=0, value=60000, step=1000, help="Borrower's annual income.")
            dti = st.slider("DTI", 0.0, 100.0, 15.0, help="Ratio calculated using the borrower’s total monthly debt payments divided by monthly gross income.")
            
            st.subheader("Credit history")
            fico = st.slider("FICO", 300, 850, 700, help="Borrower's FICO score.")
            credit_hist_years = st.number_input("Credit history years", 0, 50, 10, help="Number of years since the borrower's earliest reported credit line was opened.")
            pub_rec = st.radio("Any recorded bankruptcies", ["No",'Yes'], help="Any derogatory public records.")
            pub_rec_val = 1 if pub_rec=='Yes' else 0

        with col2:
            st.subheader("Credit information")
            term = st.radio("Term", ["36 months", "60 months"], help="Number of payments on the loan.")
            term_val = 0 if term == "36 months" else 1
            
            purpose_options = {
                'debt_consolidation': 'Debt consolidation',
                'credit_card': 'Credit card',
                'home_improvement': 'Home improvement',
                'major_purchase': 'Major purchase',
                'medical': 'Medical',
                'small_business': 'Small business',
                'car': 'Car',
                'other': 'Other'
            }
            sorted_purpose = sorted(purpose_options.keys(), key=lambda x: purpose_options[x])

            purpose = st.selectbox(
                "Purpose", 
                sorted_purpose, 
                format_func=lambda x: purpose_options[x],
                help="Category provided by the borrower for the loan request."
            )
            
            home_ownership = st.selectbox("Home ownership", ['MORTGAGE', 'RENT', 'OWN'], help="Home ownership status.")
            
            st.subheader("Other")
            emp_length =st.select_slider(
                "Employment length (years)", 
                options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                value=5,
                format_func=lambda x: "10+" if x == 10 else str(x),
                help="Employment duration."
            )
            open_acc = st.number_input("Opened accounts", 0, 50, 1, 
                                       help="Number of open credit lines in the borrower's credit file.")
            mort_acc = st.number_input("Opened mortgage accounts", 0, 20, 1, help="Number of mortgage accounts")
            revol_util = st.slider("Revolving utilization", 0.0, 150.0, 40.0, help="Amount of credit the borrower is using relative to all available revolving credit.")

        submitted = st.form_submit_button("Calculate risk")

    if submitted:

            input_data = pd.DataFrame({
                'loan_amnt': [loan_amnt],
                'term': [term_val],
                'fico':[fico],
                'pub_rec': [pub_rec_val],
                'emp_length': [emp_length],
                'home_ownership': [home_ownership],
                'annual_inc': [annual_inc],
                'purpose': [purpose],
                'dti': [dti],
                'revol_util': [revol_util],
                'mort_acc': [mort_acc],
                'open_acc': [open_acc],
                'credit_hist_years': [credit_hist_years]
            })

            try:
                prob = model.predict_proba(input_data)[:, 1][0]
                prob_percent = prob * 100
                if prob < 0.4:
                    risk_category = "LOW RISK"
                elif prob < 0.6:
                    risk_category = "MODERATE RISK"
                else:
                    risk_category = "HIGH RISK"
                st.divider()
                st.subheader(f"Results")
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.metric(label="Default probability", value=f"{prob_percent:.2f}%")
                with col_res2:
                    if risk_category == "LOW RISK":
                        st.success(f"🟢 **{risk_category}**")
                    elif risk_category == "MODERATE RISK":
                        st.warning(f"🟡 **{risk_category}**")
                    else:
                        st.error(f"🔴 **{risk_category}**")
                current_summary = input_data.copy()
                current_summary.insert(1, 'risk', risk_category)
                current_summary.insert(2, 'probability', f"{round(prob_percent,2)}%")
                current_summary['term']=current_summary['term'].apply(lambda x: 60 if x==1 else 36)

                st.session_state['history_data'] = pd.concat(
                    [current_summary, st.session_state['history_data']], 
                    ignore_index=True
                )
            except Exception as e:
                st.error(f"Error {e}")

    if not st.session_state['history_data'].empty:
        st.divider()
        st.subheader("History")
        st.write("It resets after every app refresh.")
        st.dataframe(st.session_state['history_data'], use_container_width=True)


def tab_about():
    st.header(f"ℹ️ About")
    st.markdown("""
    ### Model
    This calculator uses **XGBoost Classifier**. The whole process of data analysis, preprocessing and
    model creating is documented here in [Jupyter Notebook](https://github.com/olek2852/Pre-approval-credit-risk-scorer/blob/main/credit_notebook.ipynb).
    
    **Metrics:**
    * **ROC-AUC score:** 0.701
    * **GINI:** 0.402
    * **RECALL**: 0.83 (threshold: 0.4).
    """)
    st.image("feature_importances.png")
    st.divider()
    st.subheader("Data")
    st.markdown("""
    Dataset used: [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).  (Kaggle)         
        
    **Note:** 
    These plots use a representative sample of 50.000 records for performance. 
    The full EDA and observations are available in the Jupter Notebook mentioned above.
                
    """)
    st.subheader("Target (loan status)")

    counts = data['loan_status'].value_counts()
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6})
    colors = sns.color_palette("Set2")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = sns.countplot(data=data, x='loan_status', palette=colors[:2][::-1], ax=axes[0])
    axes[0].set_title('Target Count')
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=10)
    axes[0].set_facecolor('#f9f9f9')

    axes[1].pie(x=counts.values, labels=counts.index, colors=colors, 
                autopct='%1.2f%%', pctdistance=0.85, explode=(0.03, 0))
    axes[1].set_title('Target Distribution')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    data['fico'] = (data['fico_range_low'] + data['fico_range_high']) / 2
    num_cols = list(data.select_dtypes(exclude='object').columns.drop(['target_num'], errors='ignore'))
    to_drop = ['issue_d', 'earliest_cr_line', 'fico_range_low', 'fico_range_high']
    num_cols = [c for c in num_cols if c not in to_drop]
    cat_cols = list(data.select_dtypes(exclude=np.number).columns.drop('loan_status', errors='ignore'))
    
    selected_col = st.selectbox("Select feature to visualize", num_cols + cat_cols)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=100)

    if selected_col in num_cols:
        if selected_col in ['term', 'emp_length', 'pub_rec']:
            sns.countplot(data=data, x=selected_col, ax=axes[0], color="#4a69bd", edgecolor='white')
            axes[0].set_title(f'Count by {selected_col}', fontsize=26)
            axes[0].set_xlabel(selected_col, fontsize=22)
            axes[0].set_ylabel('Count', fontsize=22)
            axes[0].tick_params(axis='both', labelsize=18)
        
            sns.barplot(data=data, x=selected_col, y='target_num', ax=axes[1], color=colors[1],errorbar=None)
            axes[1].set_title(f'Default rate by {selected_col}', fontsize=26)
            axes[1].set_xlabel(selected_col, fontsize=22)
            axes[1].set_ylabel('Probability of Default', fontsize=22)
            axes[1].tick_params(axis='both', labelsize=18)
        
            for container in axes[1].containers:
                axes[1].bar_label(container, fmt='%.2f', padding=3, fontsize=20)
        else:
            sns.histplot(data, x=selected_col, kde=True, ax=axes[0], bins=70, color="#4a69bd", edgecolor='white')
            axes[0].set_title(f'Distribution of {selected_col}', fontsize=26)
            axes[0].set_xlabel(selected_col, fontsize=22)
            axes[0].set_ylabel('Count', fontsize=22)
            axes[0].tick_params(axis='both', labelsize=18)
            sns.boxplot(data=data, x=selected_col, y='loan_status', ax=axes[1], palette=colors[:2][::-1], showfliers=False)
            axes[1].set_title(f'{selected_col} vs Loan Status', fontsize=26)
            axes[1].set_xlabel(selected_col, fontsize=22)
            axes[1].set_ylabel('Loan Status', fontsize=22)
            axes[1].tick_params(axis='both', labelsize=18)

    elif selected_col in cat_cols:
        order1 = data.groupby(selected_col)['target_num'].count().sort_values(ascending=False).index
        order2 = data.groupby(selected_col)['target_num'].mean().sort_values(ascending=False).index
        sns.countplot(data=data, x=selected_col, ax=axes[0], color="#4a69bd", edgecolor='white', order=order1)
        axes[0].set_title(f'Count by {selected_col}', fontsize=26)
        axes[0].set_xlabel(selected_col, fontsize=22)
        axes[0].set_ylabel('Count', fontsize=22)
        axes[0].tick_params(axis='both', labelsize=18)
        sns.barplot(data=data, x=selected_col, y='target_num', color=colors[1], order=order2, ax=axes[1],errorbar=None)
        axes[1].set_title(f'Default rate by {selected_col}', fontsize=26)
        axes[1].set_xlabel(selected_col, fontsize=22)
        axes[1].set_ylabel('Probability of Default', fontsize=22)
        axes[1].tick_params(axis='both', labelsize=18)
        
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.2f', padding=3, fontsize=18)
            
        if selected_col == 'purpose':
            axes[0].tick_params(axis='x', rotation=45, labelsize=18)
            axes[1].tick_params(axis='x', rotation=45, labelsize=18)
    axes[0].set_facecolor('#f9f9f9')
    axes[1].set_facecolor('#f9f9f9')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.title("Pre-approval credit risk scorer")
    
    tab1, tab2 = st.tabs(["💲 Calculator", "ℹ️ About"])
    with tab1:
        tab_calculator()
    
    with tab2:
        tab_about()

if __name__ == "__main__":
    main()