# Set page configuration - MUST BE FIRST STREAMLIT COMMAND

# Importing all the libraries 
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib
import seaborn as sns
import datetime as dt
from PIL import Image
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Mutual Funds Prediction",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Minimal CSS that won't hide sidebar
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Rest of your code continues...
# Title for the web app 
st.markdown("<h1 style='text-align: center; color: white;'>📈 Mutual Fund Returns Predictor</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained deployment pipeline."""
    try:
        return joblib.load('mutual_fund_returns_pipeline.pkl')
    except Exception as e:
        try:
            with open('mutual_fund_returns_pipeline.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

RISK_SCORE_MAPPING = {
    'Very High': 5,
    'High': 4,
    'Moderately High': 3,
    'Moderate': 2,
    'Low to Moderate': 1.5,
    'Moderately Low': 1,
    'Low': 0.5
}

FUND_TYPE_ENCODING = {
    'Debt': 0,
    'Equity': 1,
    'Hybrid': 2,
    'Other': 3,
    'Solution Oriented': 4
}

MODEL_PERIOD_KEYS = {
    "one_year": "1_year",
    "three_year": "3_year",
    "five_year": "5_year"
}

def build_model_features(fund_data, feature_columns):
    """Create the engineered feature row expected by the saved models."""
    aum = fund_data['aum']
    nav = fund_data['nav']
    equity_per = fund_data['equity_per']
    debt_per = fund_data['debt_per']
    fund_type = fund_data['fund_type']

    engineered = {
        'aum_funds_individual_lst': aum,
        'nav_funds_individual_lst': nav,
        'rating_of_funds_individual_lst': fund_data['rating'],
        'minimum_funds_individual_lst': 500,
        'debt_per': debt_per,
        'equity_per': equity_per,
        'risk_score': RISK_SCORE_MAPPING.get(fund_data['risk'], 2),
        'fund_type_encoded': FUND_TYPE_ENCODING.get(fund_type, FUND_TYPE_ENCODING['Other']),
        'aum_nav_ratio': aum / (nav + 1),
        'equity_concentration': equity_per / 100,
        'debt_concentration': debt_per / 100,
        'is_equity_fund': int(fund_type == 'Equity'),
        'is_hybrid_fund': int(fund_type == 'Hybrid')
    }

    return pd.DataFrame([{column: engineered.get(column, 0) for column in feature_columns}])

def predict_returns(fund_data, return_type):
    """Predict returns using the saved model pipeline, with a heuristic fallback."""
    try:
        pipeline = load_model()
        if isinstance(pipeline, dict):
            model_key = MODEL_PERIOD_KEYS.get(return_type)
            models = pipeline.get('models', {})
            scalers = pipeline.get('scalers', {})
            feature_columns = pipeline.get('feature_columns', [])

            if model_key in models and model_key in scalers and feature_columns:
                prediction_df = build_model_features(fund_data, feature_columns)
                prediction_scaled = scalers[model_key].transform(prediction_df)
                prediction = models[model_key].predict(prediction_scaled)
                return prediction[0]

        if pipeline is not None and hasattr(pipeline, "predict"):
            prediction_df = build_model_features(
                fund_data,
                [
                    'aum_funds_individual_lst',
                    'nav_funds_individual_lst',
                    'rating_of_funds_individual_lst',
                    'debt_per',
                    'equity_per',
                    'risk_score',
                    'fund_type_encoded'
                ]
            )
            prediction = pipeline.predict(prediction_df)
            return prediction[0]

        return calculate_fallback_return(fund_data, return_type)
    except Exception as e:
        return calculate_fallback_return(fund_data, return_type)

def calculate_fallback_return(fund_data, return_type):
    """Fallback calculation using fund characteristics"""
    base_return = 8.0
    equity_bonus = (fund_data['equity_per'] - 50) * 0.2
    rating_bonus = (fund_data['rating'] - 3) * 2.0
    
    # Risk adjustment
    risk_map = {"Very High": 3, "High": 2, "Moderately High": 1, "Moderate": 0,
                "Low to Moderate": -1, "Moderately Low": -2, "Low": -3}
    risk_bonus = risk_map.get(fund_data['risk'], 0) * 1.5
    
    # Fund type adjustment
    type_map = {"Equity": 3, "Hybrid": 1, "Solution Oriented": 0, "Other": -1, "Debt": -2}
    type_bonus = type_map.get(fund_data['fund_type'], 0)
    
    # Time factor based on return type
    time_factors = {"one_year": 1.0, "three_year": 1.3, "five_year": 1.6}
    time_factor = time_factors.get(return_type, 1.0)
    
    prediction = (base_return + equity_bonus + rating_bonus + risk_bonus + type_bonus) * time_factor
    return max(-10, min(prediction, 40))

def get_return_interpretation(prediction_value, return_period):
    """Get interpretation based on prediction value and return period"""
    if return_period == "One Year":
        if prediction_value > 15:
            return "Excellent for short-term investment"
        elif prediction_value > 8:
            return "Good for short-term gains"
        elif prediction_value > 0:
            return "Moderate short-term returns"
        else:
            return "Not recommended for short-term"
    
    elif return_period == "Three Years":
        if prediction_value > 20:
            return "Excellent for medium-term investment"
        elif prediction_value > 12:
            return "Good for medium-term growth"
        elif prediction_value > 5:
            return "Suitable for medium-term"
        else:
            return "Low medium-term potential"
    
    else:  # Five Years
        if prediction_value > 25:
            return "Outstanding for long-term wealth creation"
        elif prediction_value > 15:
            return "Strong long-term growth potential"
        elif prediction_value > 8:
            return "Good for long-term portfolio"
        else:
            return "Consider other long-term options"

def get_prediction_range(prediction_value):
    """Convert an exact prediction into a simple presentation-friendly range."""
    if prediction_value < 0:
        lower = np.floor(prediction_value / 10) * 10
        upper = lower + 10
    else:
        lower = np.floor(prediction_value / 10) * 10
        upper = lower + 10

    return int(lower), int(upper)

def get_confidence_label(prediction_value, fund_data):
    """Estimate confidence from input completeness and model-friendly ranges."""
    confidence_score = 0

    if fund_data['aum'] > 0:
        confidence_score += 1
    if fund_data['nav'] > 0:
        confidence_score += 1
    if fund_data['rating'] >= 3:
        confidence_score += 1
    if 0 <= fund_data['equity_per'] <= 100:
        confidence_score += 1
    if -10 <= prediction_value <= 40:
        confidence_score += 1

    if confidence_score >= 4:
        return "High"
    if confidence_score >= 3:
        return "Medium"
    return "Low"

def get_category_average(return_column, fund_type):
    """Read local data and return the category average for comparison."""
    try:
        df = pd.read_excel('data/cleaned_data.xlsx')
        if return_column not in df.columns or 'type_of_fund' not in df.columns:
            return None

        category_df = df[df['type_of_fund'] == fund_type]
        if category_df.empty:
            return None

        return category_df[return_column].mean()
    except Exception:
        return None

def get_category_comparison(prediction_value, category_average, fund_type):
    """Compare model prediction against dataset category average."""
    if category_average is None or pd.isna(category_average):
        return "Category comparison is not available for this selection."

    difference = prediction_value - category_average
    if difference > 2:
        return f"This prediction is above the dataset average for {fund_type} funds by about {difference:.1f} percentage points."
    if difference < -2:
        return f"This prediction is below the dataset average for {fund_type} funds by about {abs(difference):.1f} percentage points."
    return f"This prediction is close to the dataset average for {fund_type} funds."

def get_prediction_reasons(fund_data):
    """Generate simple user-facing reasons from input characteristics."""
    reasons = []

    if fund_data['equity_per'] >= 70:
        reasons.append("High equity allocation can increase return potential, but it also increases market risk.")
    elif fund_data['equity_per'] >= 40:
        reasons.append("Balanced equity allocation gives moderate growth potential with lower volatility than aggressive equity funds.")
    else:
        reasons.append("Lower equity allocation usually reduces return potential but may provide more stability.")

    if fund_data['rating'] >= 4:
        reasons.append("A higher fund rating supports a stronger prediction because it reflects better historical quality indicators.")
    elif fund_data['rating'] <= 2:
        reasons.append("A lower fund rating weakens the prediction because it may indicate weaker historical quality indicators.")

    if fund_data['risk'] in ['Very High', 'High']:
        reasons.append("The selected risk profile suggests higher volatility and higher possible return movement.")
    elif fund_data['risk'] in ['Low', 'Moderately Low', 'Low to Moderate']:
        reasons.append("The selected risk profile suggests a more conservative return pattern.")

    if fund_data['aum'] >= 1000:
        reasons.append("Higher AUM may indicate fund scale and investor confidence.")

    return reasons[:4]

def main_prediction_page():
    st.subheader("Mutual Fund Returns Prediction")
    
    # Create a clean form layout
    with st.form(key='prediction_form'):
        # Input fields in vertical layout only
        st.write("### Fund Information")
        
        user_AUM = st.number_input(
            'Assets Under Management (AUM) in Crores',
            value=1711.78,
            min_value=0.0,
            help="Total assets managed by the fund"
        )
        
        user_NAV = st.number_input(
            'Net Asset Value (NAV)',
            value=127.22,
            min_value=0.0,
            help="Price per unit of the fund"
        )
        
        user_rating = st.selectbox(
            'Fund Rating',
            ['1', '2', '3', '4', '5'],
            index=4
        )
        
        user_equity = st.number_input(
            'Equity Allocation Percentage',
            min_value=0,
            max_value=100,
            value=70,
            help="Percentage invested in stocks (0 to 100)"
        )
        
        # Removed debt allocation display
        
        risk_of_the_fund_user = st.selectbox(
            'Risk Profile',
            ['Very High', 'High', 'Moderately High', 'Moderate', 'Low to Moderate', 'Moderately Low', 'Low'],
            index=0
        )
        
        type_of_fund = st.selectbox(
            'Fund Category',
            ['Equity', 'Hybrid', 'Solution Oriented', 'Debt', 'Other'],
            index=0
        )
        
        # Return period selection using dropdown
        st.write("### Prediction Period")
        return_period = st.selectbox(
            "Select time period for prediction:",
            ["One Year", "Three Years", "Five Years"],
            index=0,  # Default to One Year
            help="Choose the investment period for return prediction"
        )

        show_exact_prediction = st.checkbox(
            "Show exact model prediction",
            value=False,
            help="Enable this if you want to see the precise value predicted by the model."
        )
        
        submitted = st.form_submit_button('Calculate Returns', use_container_width=True)

    # Handle prediction after form submission
    if submitted:
        with st.spinner('Analyzing fund performance...'):
            try:
                # Prepare fund data
                fund_data = {
                    'aum': user_AUM,
                    'nav': user_NAV,
                    'rating': int(user_rating),
                    'equity_per': user_equity,
                    'debt_per': 100 - user_equity,  # Still calculate internally but don't show
                    'risk': risk_of_the_fund_user,
                    'fund_type': type_of_fund
                }
                
                # Map return period to prediction type
                period_map = {
                    "One Year": "one_year",
                    "Three Years": "three_year", 
                    "Five Years": "five_year"
                }
                return_column_map = {
                    "One Year": "one_year_returns",
                    "Three Years": "three_year_returns",
                    "Five Years": "five_year_returns"
                }
                
                return_type = period_map[return_period]
                
                # Predict returns
                prediction = predict_returns(fund_data, return_type)
                prediction_value = prediction
                all_predictions = {
                    "One Year": predict_returns(fund_data, "one_year"),
                    "Three Years": predict_returns(fund_data, "three_year"),
                    "Five Years": predict_returns(fund_data, "five_year")
                }
                
                # Get interpretation
                interpretation = get_return_interpretation(prediction_value, return_period)
                range_lower, range_upper = get_prediction_range(prediction_value)
                confidence_label = get_confidence_label(prediction_value, fund_data)
                category_average = get_category_average(return_column_map[return_period], type_of_fund)
                category_comparison = get_category_comparison(prediction_value, category_average, type_of_fund)
                prediction_reasons = get_prediction_reasons(fund_data)
                
                # Display result with improved styling
                st.markdown("---")
                
                # Determine color and arrow based on prediction
                if prediction_value >= 0:
                    color = "#00cc66"  # Bright green
                    arrow = "▲"
                    sign = "+"
                else:
                    color = "#ff4d4d"  # Bright red
                    arrow = "▼"
                    sign = ""
                
                # Improved display with better styling
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 20px;'>
                        <h3 style='color: #FFFFFF; margin-bottom: 5px; font-size: 18px; font-weight: 600;'>
                            Estimated {return_period} Return Range
                        </h3>
                        <div style='font-size: 52px; color: {color}; font-weight: bold; margin: 10px 0;'>
                            {arrow} {range_lower}% to {range_upper}%
                        </div>
                        <div style='color: #FFFFFF; font-size: 16px; font-weight: 500; margin-top: 10px;'>
                            {interpretation}
                        </div>
                        <div style='color: #D9D9D9; font-size: 14px; margin-top: 8px;'>
                            Confidence: {confidence_label}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                if show_exact_prediction:
                    st.info(f"Exact model prediction: {prediction_value:.2f}%")

                st.write("### Category Comparison")
                if category_average is not None and not pd.isna(category_average):
                    st.metric(
                        f"Average {return_period} Return for {type_of_fund} Funds",
                        f"{category_average:.2f}%"
                    )
                st.write(category_comparison)

                st.write("### Return Forecast Chart")
                chart_df = pd.DataFrame({
                    "Period": list(all_predictions.keys()),
                    "Predicted Return (%)": list(all_predictions.values())
                })

                fig, ax = plt.subplots(figsize=(8, 4))
                bar_colors = [
                    "#00cc66" if period == return_period else "#4ECDC4"
                    for period in chart_df["Period"]
                ]
                bars = ax.bar(
                    chart_df["Period"],
                    chart_df["Predicted Return (%)"],
                    color=bar_colors
                )
                ax.set_title("Predicted Returns Across Investment Horizons")
                ax.set_ylabel("Predicted Return (%)")
                ax.set_xlabel("Investment Horizon")
                ax.axhline(0, color="#333333", linewidth=0.8)
                ax.grid(axis="y", alpha=0.25)

                for bar in bars:
                    height = bar.get_height()
                    label_y = height if height >= 0 else 0
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                        fontsize=10
                    )

                plt.tight_layout()
                st.pyplot(fig)

                st.write("### Why This Prediction?")
                for reason in prediction_reasons:
                    st.write(f"- {reason}")

                st.warning(
                    "Educational use only. This prediction is based on historical data and selected fund characteristics; it is not financial advice."
                )
                
            except Exception as e:
                st.error("An error occurred during prediction. Please try again.")

def analytics():
    st.subheader("Mutual Funds Analytics Dashboard")
    
    try:
        def load_google_sheets_data():
            SHEET_ID = "1aIdCAOWZhRp1rEbr4H-kqocHB9oQn2i31gAAH8nsXYY"
            csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
            try:
                return pd.read_csv(csv_url)
            except Exception:
                return pd.read_excel('data/cleaned_data.xlsx')
        
        # Load the data
        with st.spinner('Loading data from Google Sheets...'):
            df = load_google_sheets_data()
        
        # st.success("✅ Data loaded successfully from Google Sheets!")
        
        # # Debug: Show data info
        # st.write(f"**Dataset loaded:** {len(df)} rows, {len(df.columns)} columns")
        # st.write("**First few rows:**")
        # st.dataframe(df.head(3))
        
        # Rest of your existing analytics code continues here...
        st.sidebar.write("**Data Columns:**", list(df.columns))
        
        # Create tabs for different types of analysis
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Overview", 
            "Risk Analysis", 
            "Top Performers", 
            "Data Explorer"
        ])
        
        with tab1:
            st.write("### Fund Performance Overview")
            
            # Key metrics - using safe column access
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'one_year_returns' in df.columns:
                    avg_1yr = df['one_year_returns'].mean()
                    st.metric("Average 1-Year Return", f"{avg_1yr:.2f}%")
                else:
                    st.metric("Average 1-Year Return", "N/A")
            
            with col2:
                if 'three_year_returns' in df.columns:
                    avg_3yr = df['three_year_returns'].mean()
                    st.metric("Average 3-Year Return", f"{avg_3yr:.2f}%")
                else:
                    st.metric("Average 3-Year Return", "N/A")
            
            with col3:
                if 'five_year_returns' in df.columns:
                    avg_5yr = df['five_year_returns'].mean()
                    st.metric("Average 5-Year Return", f"{avg_5yr:.2f}%")
                else:
                    st.metric("Average 5-Year Return", "N/A")
            
            # Returns distribution if columns exist
            returns_columns = [col for col in ['one_year_returns', 'three_year_returns', 'five_year_returns'] if col in df.columns]
            if returns_columns:
                st.write("#### Returns Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                returns_data = df[returns_columns]
                sns.boxplot(data=returns_data, palette="Set2")
                ax.set_title('Returns Distribution Across Time Periods')
                ax.set_ylabel('Returns (%)')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            # Performance by fund type if column exists
            if 'type_of_fund' in df.columns and returns_columns:
                st.write("#### Performance by Fund Category")
                category_performance = df.groupby('type_of_fund')[returns_columns].mean().round(2)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                category_performance.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(returns_columns)])
                ax.set_title('Average Returns by Fund Category')
                ax.set_ylabel('Returns (%)')
                ax.set_xlabel('Fund Category')
                plt.xticks(rotation=45)
                plt.legend(title='Time Period')
                st.pyplot(fig)
        
        with tab2:
            st.write("### Risk-Return Analysis")
            
            # Risk vs Returns scatter plot if columns exist
            if 'three_year_returns' in df.columns and 'equity_per' in df.columns:
                st.write("#### Equity Allocation vs Returns")
                
                # Clean the data first
                plot_df = df.dropna(subset=['three_year_returns', 'equity_per']).copy()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # SIMPLIFIED VERSION - Remove size mapping to AUM
                if 'risk_of_the_fund' in df.columns:
                    # Create scatter plot without size mapping
                    scatter = sns.scatterplot(
                        data=plot_df, 
                        x='three_year_returns', 
                        y='equity_per', 
                        hue='risk_of_the_fund',
                        s=60,  # Fixed size
                        ax=ax,
                        palette='viridis'
                    )
                    # Move legend outside the plot
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Risk Level')
                else:
                    # Basic scatter plot without risk coloring
                    scatter = sns.scatterplot(
                        data=plot_df, 
                        x='three_year_returns', 
                        y='equity_per', 
                        s=60,
                        ax=ax
                    )
                
                ax.set_title('Equity Allocation vs 3-Year Returns', fontsize=14, fontweight='bold')
                ax.set_xlabel('3-Year Returns (%)', fontsize=12)
                ax.set_ylabel('Equity Percentage (%)', fontsize=12)
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3)
                
                # Adjust layout to prevent legend cutoff
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show some statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Funds Plotted", len(plot_df))
                with col2:
                    correlation = plot_df['three_year_returns'].corr(plot_df['equity_per'])
                    st.metric("Correlation", f"{correlation:.2f}")
                with col3:
                    st.metric("Average Equity %", f"{plot_df['equity_per'].mean():.1f}%")
            
            else:
                st.warning("Required columns ('three_year_returns' and 'equity_per') not found for risk-return analysis.")
            
            # Risk category performance if column exists
            if 'risk_of_the_fund' in df.columns:
                st.write("#### Average Returns by Risk Category")
                risk_columns = [col for col in ['one_year_returns', 'three_year_returns', 'five_year_returns', 'equity_per'] if col in df.columns]
                if risk_columns:
                    risk_performance = df.groupby('risk_of_the_fund')[risk_columns].mean().round(2)
                    st.dataframe(risk_performance)
                
                # Risk distribution - SIMPLIFIED
                st.write("#### Risk Category Distribution")
                risk_counts = df['risk_of_the_fund'].value_counts()
                
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Pie chart
                ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                       startangle=90, colors=sns.color_palette("viridis", len(risk_counts)))
                ax1.set_title('Distribution by Risk Category')
                
                # Bar chart
                risk_counts.sort_values(ascending=False).plot(kind='bar', ax=ax2, color='skyblue')
                ax2.set_title('Number of Funds by Risk Category')
                ax2.set_ylabel('Number of Funds')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig2)
        
        with tab3:
            st.write("### Top Performing Funds")
            
            # Find identifier column (use first column if fund_name doesn't exist)
            identifier_col = 'fund_name' if 'fund_name' in df.columns else df.columns[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'three_year_returns' in df.columns:
                    st.write("#### Top 10 Funds - 3 Year Returns")
                    top_3yr = df.nlargest(10, 'three_year_returns')[[identifier_col, 'three_year_returns']]
                    if 'type_of_fund' in df.columns:
                        top_3yr['type_of_fund'] = df.loc[top_3yr.index, 'type_of_fund']
                    if 'risk_of_the_fund' in df.columns:
                        top_3yr['risk_of_the_fund'] = df.loc[top_3yr.index, 'risk_of_the_fund']
                    
                    top_3yr = top_3yr.rename(columns={
                        identifier_col: 'Fund',
                        'three_year_returns': '3-Year Returns (%)'
                    })
                    st.dataframe(top_3yr.reset_index(drop=True), use_container_width=True)
                else:
                    st.write("3-year returns data not available")
            
            with col2:
                if 'five_year_returns' in df.columns:
                    st.write("#### Top 10 Funds - 5 Year Returns")
                    top_5yr = df.nlargest(10, 'five_year_returns')[[identifier_col, 'five_year_returns']]
                    if 'type_of_fund' in df.columns:
                        top_5yr['type_of_fund'] = df.loc[top_5yr.index, 'type_of_fund']
                    if 'risk_of_the_fund' in df.columns:
                        top_5yr['risk_of_the_fund'] = df.loc[top_5yr.index, 'risk_of_the_fund']
                    
                    top_5yr = top_5yr.rename(columns={
                        identifier_col: 'Fund',
                        'five_year_returns': '5-Year Returns (%)'
                    })
                    st.dataframe(top_5yr.reset_index(drop=True), use_container_width=True)
                else:
                    st.write("5-year returns data not available")
        
        with tab4:
            st.write("### Data Explorer")
            
            # Show raw data with filters
            st.write("#### Fund Data Overview")
            
            # Column selector for display
            available_columns = list(df.columns)
            default_cols = [col for col in ['fund_name', 'type_of_fund', 'risk_of_the_fund', 
                                          'one_year_returns', 'three_year_returns', 'five_year_returns',
                                          'equity_per', 'aum_funds_individual_lst'] if col in df.columns]
            
            selected_columns = st.multiselect(
                "Select columns to display:",
                available_columns,
                default=default_cols[:5] if default_cols else available_columns[:5]
            )
            
            if selected_columns:
                # Add filters
                st.write("#### Filters")
                col1, col2 = st.columns(2)
                
                filtered_df = df[selected_columns].copy()
                
                # Fund type filter
                if 'type_of_fund' in selected_columns:
                    with col1:
                        fund_types = ['All'] + list(df['type_of_fund'].unique())
                        selected_fund_type = st.selectbox("Fund Type", fund_types)
                        if selected_fund_type != 'All':
                            filtered_df = filtered_df[filtered_df['type_of_fund'] == selected_fund_type]
                
                # Risk level filter
                if 'risk_of_the_fund' in selected_columns:
                    with col2:
                        risk_levels = ['All'] + list(df['risk_of_the_fund'].unique())
                        selected_risk = st.selectbox("Risk Level", risk_levels)
                        if selected_risk != 'All':
                            filtered_df = filtered_df[filtered_df['risk_of_the_fund'] == selected_risk]
                
                st.write(f"**Showing {len(filtered_df)} funds**")
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download option
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name=f"mutual_funds_data_{dt.date.today()}.csv",
                    mime="text/csv"
                )
        
        # Data summary
        st.markdown("---")
        st.write("### Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Funds", len(df))
        
        with col2:
            if 'type_of_fund' in df.columns:
                unique_categories = df['type_of_fund'].nunique()
                st.metric("Fund Categories", unique_categories)
            else:
                st.metric("Fund Categories", "N/A")
        
        with col3:
            if 'risk_of_the_fund' in df.columns:
                unique_risks = df['risk_of_the_fund'].nunique()
                st.metric("Risk Levels", unique_risks)
            else:
                st.metric("Risk Levels", "N/A")
                
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
        st.info("""
        **To fix this:**
        1. Make sure your Google Sheet is publicly accessible (Anyone with link can view)
        2. Check that your Google Sheet has data in the expected format
        3. Verify the column names match what the code expects
        """)
        
        # Show what we tried to load
        st.write("**Troubleshooting info:**")
        st.write(f"Error type: {type(e).__name__}")
        st.write("Full error:", str(e))
def description():
    st.subheader("About This Tool")
    
    st.markdown("""
    ### Mutual Fund Returns Predictor
    
    **Purpose:**
    This application helps investors make informed decisions by predicting potential returns from mutual funds based on key fund characteristics.
    
    **How It Works:**
    1. Enter Fund Details - Provide basic information about the mutual fund
    2. Select Prediction Period - Choose your investment horizon
    3. Get Prediction - Receive expected returns with investment guidance
    
    **Investment Philosophy:**
    - Past performance ≠ Future results
    - Diversification reduces risk
    - Long-term investing typically yields better returns
    - Understand your risk tolerance before investing
    
    **Disclaimer:** 
    Predictions are based on statistical models and historical patterns. 
    Actual returns may vary due to market conditions, economic factors, and other variables. 
    Always consult with a qualified financial advisor before making investment decisions.
    """)

# Enhanced sidebar with input explanations
st.sidebar.title("Navigation")
page_options = ["Returns Prediction", "About Tool", "Analytics"]
selected_page = st.sidebar.selectbox("Choose section:", page_options, index=0)  # Default to Returns Prediction

# Input explanations in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Input Guide")

with st.sidebar.expander("AUM (Assets Under Management)"):
    st.write("Total value of assets managed by the fund. Higher AUM often indicates investor confidence and fund stability.")

with st.sidebar.expander("NAV (Net Asset Value)"):
    st.write("Price per unit of the mutual fund. Calculated as total assets minus liabilities divided by number of units.")

with st.sidebar.expander("Fund Rating"):
    st.write("Quality rating from 1-5 stars. Higher ratings typically indicate better historical performance and management.")

with st.sidebar.expander("Equity Allocation"):
    st.write("Percentage invested in stocks. Higher equity = higher potential returns but also higher risk.")

with st.sidebar.expander("Risk Profile"):
    st.write("Fund's risk level. Very High = aggressive, Low = conservative. Match with your personal risk tolerance.")

with st.sidebar.expander("Fund Category"):
    st.write("Type of mutual fund. Equity = stocks, Debt = bonds, Hybrid = mix of both, Solution Oriented = goal-based.")

# Map selection to page function
page_functions = {
    "Returns Prediction": main_prediction_page,
    "About Tool": description,
    "Analytics": analytics
}

# Display the selected page
page_functions[selected_page]()
