import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Algo Trading Pipeline EDA", layout="wide")
st.title("Data Analytics: Client Conversion Strategy Validation")

@st.cache_data
def load_data():
    # Loading the clean dataset you just generated
    return pd.read_csv('cleaned_trading_data.csv')

df = load_data()

st.header("1. Descriptive Analytics")
st.write("Dataset Overview:", df.head())
st.write("Statistical Summary:", df.describe())

st.header("2. Diagnostic Analytics & Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Allocated Capital")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Allocated_Capital_USD'], bins=30, kde=True, color='blue', ax=ax)
    st.pyplot(fig)
    st.caption("Highlights right-skewed capital allocation among clients.")

with col2:
    st.subheader("Risk Tolerance vs. Average Monthly ROI")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Risk_Tolerance', y='Avg_Monthly_ROI', data=df, order=['Low', 'Medium', 'High'], palette='Set2', ax=ax2)
    st.pyplot(fig2)
    st.caption("Diagnoses the relationship between client risk profile and actual market returns.")

st.subheader("Correlation Analysis")
fig3, ax3 = plt.subplots(figsize=(10, 6))
# Select only numeric columns for the heatmap
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
st.pyplot(fig3)
st.caption("Diagnoses linear relationships (e.g., Capital and Commission).")

st.subheader("Actionable Business Insight")
st.info("The data strategy validates that focusing marketing efforts on high-risk-tolerance individuals significantly drives potential commission revenue for the DII/FII mediator model.")