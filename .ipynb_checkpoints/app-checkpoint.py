import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set up layout
st.set_page_config(page_title="AirMatters App", layout="wide")

###################################################################################################

# Set background and sidebar styles
def set_bg_and_sidebar(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            height: 100%;
            margin: 0;
            padding: 0;
            color: white;
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: top center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Sidebar background */
        section[data-testid="stSidebar"] {{
            background-color: rgba(247, 244, 246, 0.11);
            padding-top: 20px;
            padding-left: 15px;
            padding-right: 15px;
        }}

        /* Sidebar title */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: white !important;
            text-shadow: 2px 2px 4px #000000;
        }}

        /* Sidebar radio labels (navigation items) */
        .st-emotion-cache-1v3fvcr label, .st-emotion-cache-16txtl3 label {{
            font-size: 5em !important;
            color: white !important;
            text-shadow: 2px 2px 4px #000000;
            text-align: left;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply styling
set_bg_and_sidebar("AirPollution1.jpg")  # Replace with your background image file path

##############################################################################################################################################


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("final_air_quality.csv")
    
air_quality_df = load_data()
df = air_quality_df.copy()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Modeling and Prediction"])

# ----------------------------------------
# Home Page
# ----------------------------------------

if page == "Home":
    #st.title("Welcome to the **AirMatters App**")    
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 60px; color: #fcfbfa;'>
        Welcome to the <strong>AirMatters Application</strong>
    </h1>
    """,
    unsafe_allow_html=True
)
    st.markdown("This Streamlit-based web application provides a comprehensive analysis of air quality data helping users to explore pollution trends")
   # Display the homepage image properly
    #st.image("AirPollution2.jpg", use_container_width=True) ----> Removing these beacuse of App loading issues
    st.markdown("- Explore China air pollution data (Urban, Suburban, Rural, Industrial)")
    st.markdown("- Understand pollutant trends over time")
    st.markdown("- Build predictive models using machine learning")
###############################################################################################################
    total_records = len(air_quality_df)
    monitoring_sites = air_quality_df['station'].nunique()
    
    st.markdown("""
    <style>
    .kpi-card {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        font-family: Arial;
    }
    .kpi-card h2 {
        font-size: 36px;
        color: #000000;
        margin: 0;
    }
    .kpi-card p {
        font-size: 18px;
        color: #000000;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='kpi-card'><h2>{total_records}</h2><p>Total Records</p></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='kpi-card'><h2>{monitoring_sites}</h2><p>Monitoring Sites</p></div>", unsafe_allow_html=True)

##############################################################################################################################################
   
    # Define card styling
    st.markdown("""
    <style>
    .site-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 1px 1px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        font-family: Arial;
    }
    .site-title {
        font-size: 20px;
        font-weight: bold;
        color: #2A416D;
    }
    .site-desc {
        font-size: 16px;
        color: #444444;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='site-card'>
            <div class='site-title'>Urban Site: Wanshouxigong</div>
            <div class='site-desc'>
            Located in central Beijing, this site represents typical urban air quality. Surrounded by dense traffic and residential/commercial zones.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("""
        <div class='site-card'>
            <div class='site-title'>Rural Site: Huairou</div>
            <div class='site-desc'>
            Situated in the northern rural region of Beijing. Characterized by lower population density and more natural surroundings.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='site-card'>
            <div class='site-title'>Suburban Site: Changping</div>
            <div class='site-desc'>
            Located in the northern outskirts of Beijing. Considered a suburban area, less dense than the city center but still developed.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown("""
        <div class='site-card'>
            <div class='site-title'>Industrial Site: Aotizhongxin</div>
            <div class='site-desc'>
            Known as a hotspot due to its proximity to Olympic venues and high development zones. This site is often monitored closely.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------
# EDA Page
# ----------------------------------------
elif page == "Exploratory Data Analysis":
    st.title(" Exploratory Data Analysis")

    st.subheader("Dataset Description")
    st.write(df.describe())

##################################################################################################################################

    st.subheader("Missing Values")
    st.markdown("Missing values in a dataset represent data that was not collected, is unavailable, or could not be recorded for certain features. In this air quality dataset, missing values might occur in columns like 'PM2.5', 'SO2', 'TEMP', etc., due to faulty sensors, data transmission issues, or operational constraints. ")
    st.markdown("In the Dataset used here for this analysis, all Missing values were identified and handled to ensure that the dataset maintained integrity for further analysis and machine learning modeling.")
    # Create missing values DataFrame
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Column', 'Missing Values']
    
    
    # Build HTML table with colored rows for missing values > 0
    table_html = """
    <style>
        .missing-table {
            border-collapse: collapse;
            width: 50%;
        }
        .missing-table th, .missing-table td {
            border: 1px solid ##454545;
            text-align: left;
            padding: 8px;
        }
        .missing-table tr:nth-child(even) {
            background-color: ##999999;
        }
        .highlight {
            background-color: #0000FF;
            color: white;
            font-weight: bold;
        }
    </style>
    <table class="missing-table">
        <tr><th>Column</th><th>Missing Values</th></tr>
    """
    
    for index, row in missing_df.iterrows():
        highlight = "highlight" if row['Missing Values'] > 0 else ""
        table_html += f"<tr class='{highlight}'><td>{row['Column']}</td><td>{row['Missing Values']}</td></tr>"
    
    table_html += "</table>"
    
    # Render in Streamlit
    st.markdown(table_html, unsafe_allow_html=True)

######################################################################################################################################
    st.subheader("Distribution of Pollutants in a Bar Chart")
    st.markdown("This bar charts show the distribution of pollutants to explore how different pollutant levels dominate and how their levels differ. This helps to prioritize which pollutants to monitor or control and to identify how to locate resources for air quality improvement.")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs = axs.flatten()

    for i, pollutant in enumerate(pollutants):
        sns.histplot(df[pollutant], kde=True, bins=30, ax=axs[i])
        axs[i].set_title(f'Distribution of {pollutant}')
        axs[i].set_xlabel(pollutant)
        axs[i].set_ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)
    
####################################################################################################################################
    #st.subheader("Record Count by Time of Day")
    #fig, ax = plt.subplots(figsize=(10, 6))  # Adjust width and height here
    #sns.countplot(x='TimeOfDay', data=air_quality_df, order=['Morning', 'Afternoon', 'Evening', 'Night'], ax=ax)
    #ax.set_title('Record Count by Time of Day', fontsize=8)
    #ax.set_xlabel('Time of Day', fontsize=8)
    #ax.set_ylabel('Count')
    #st.pyplot(fig)   -------> Removing beacuse of App loading issues
##################################################################################################################################
    
    st.subheader("Average Pollutant Levels on Weekdays vs Weekends")
    st.markdown("The reason to analyze Weekdays vs Weekends is to identify Human Activity Impact on the air pollution. This helps discover recurring trends across weeks.Comparing both helps isolate the effect of human behavior on air quality.")  
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    # Group by 'Weekend' and calculate average
    avg_pollutants = air_quality_df.groupby('Weekend')[pollutants].mean().T
    avg_pollutants.index.name = 'Pollutant'
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_pollutants.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_title('Average Pollutant Levels: Weekdays vs Weekends', fontsize=8)
    ax.set_xlabel('Pollutant', fontsize=8)
    ax.set_ylabel('Average Concentration', fontsize=8)
    ax.legend(title='Weekend', labels=['Weekday (0)', 'Weekend (1)'])
    plt.xticks(rotation=0)
    st.pyplot(fig)
    
#######################################################################################################################################
    #st.subheader("Record Count: Weekends vs Weekdays")
    #st.markdown("Weekdays often see higher pollution due to more traffic, industrial activity, and commuting. ")
    #st.markdown("Weekends may have lower emissions due to reduced business operations and travel. ")
    #plt.figure(figsize=(6, 4))
    #sns.countplot(x='Weekend', hue='Weekend', data=air_quality_df, palette='viridis', legend=False)
    #sns.countplot(x='Weekend', data=air_quality_df, palette='viridis')
    #plt.title('Record Count by Weekend Indicator')
    #plt.xlabel('Weekend (0 = Weekday, 1 = Weekend)')
   # plt.ylabel('Count')
    #st.pyplot(plt)-------> Removing beacuse of App loading issues

#######################################################################################################################################
    
    st.subheader("Scatter Plot Grid of Key Air Pollutants")
    st.markdown("The below scatter plots helps visualize correlations (positive, negative, or no correlation) between pairs of pollutants.This allows side-by-side comparison of relationships between multiple pollutants in one view which helps spot patterns that may not be obvious from individual plots.")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    # Set up figure
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 25))
    axes = axes.flatten()
    plot_num = 0

    # Plot each unique pair
    for i in range(len(pollutants)):
        for j in range(i + 1, len(pollutants)):
            sns.scatterplot(
                x=air_quality_df[pollutants[i]],
                y=air_quality_df[pollutants[j]],
                ax=axes[plot_num],
                alpha=0.5
            )
            axes[plot_num].set_title(f'{pollutants[i]} vs {pollutants[j]}')
            axes[plot_num].set_xlabel(pollutants[i])
            axes[plot_num].set_ylabel(pollutants[j])
            plot_num += 1
    # Hide unused subplots
    for k in range(plot_num, len(axes)):
        fig.delaxes(axes[k])
    plt.tight_layout()
    st.pyplot(fig)
    
########################################################################################################################################

    #pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    #sns.set(style="whitegrid")
    #st.subheader("Distribution of Each Pollutant Across Areas")
    #st.markdown("""This section presents boxplots for each air pollutant to compare their concentration levels across different areas. This helps to understand **Median values** of pollutants. **Spread or variability** of data within each area.**Outliers** unusually high or low readings and **Comparative pollution levels** between different areas. """)

   # fig, axs = plt.subplots(2, 3, figsize=(18, 10))
   # for i, pollutant in enumerate(pollutants):
     #   row, col = divmod(i, 3)
        #sns.boxplot(x='Category', y=pollutant, data=air_quality_df, ax=axs[row][col], palette='Set2')
        #sns.boxplot(x='Category', y=pollutant, hue='Category', data=air_quality_df, ax=axs[row][col], palette='Set2', legend=False)
        #axs[row][col].set_title(f'{pollutant} by Area')
        #axs[row][col].set_xlabel('Area')
        #axs[row][col].set_ylabel(pollutant)
   # plt.tight_layout()
   # st.pyplot(fig) -------> Removing beacuse of App loading issues

########################################################################################################################################## 
    
    st.subheader("Correlation Heatmap of Pollutants")
    st.markdown("This heatmap shows how strongly each pair of pollutants is related to one another, based on their Pearson correlation coefficient (values between -1 and 1)")
    st.markdown ("A high positive correlation between PM2.5 and PM10 means they often rise and fall together")
    st.markdown("A low or negative correlation between O3 and NO2 might indicate different sources or behaviors in the environment")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    corr_matrix = air_quality_df[pollutants].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap of Pollutants')
    st.pyplot(fig)
  
####################################################################################################################################
    
    #st.subheader("Daily Average Pollutant Levels Over Time")
    #air_quality_df['Datetime'] = pd.to_datetime(air_quality_df['Datetime'])
    #air_quality_df.set_index('Datetime', inplace=True)
    #st.markdown("This graph will help to observe trends, seasonality, or spikes in pollutant levels over time.Especially useful for identifying long-term pollution patterns or days with unusual spikes.")

    #pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
  # Set up 2x3 grid of subplots
    #fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    #axs = axs.flatten()  # Flatten the 2D array of axes to iterate easily
    #for i, pollutant in enumerate(pollutants):
        #daily_avg = air_quality_df[pollutant].resample('D').mean()
        #axs[i].plot(daily_avg, color='darkgreen')
        #axs[i].set_title(f'{pollutant} Over Time', fontsize=12)
        #axs[i].set_xlabel('Date')
        #axs[i].set_ylabel(f'{pollutant} Level')
        #axs[i].tick_params(axis='x', labelrotation=45)
        #axs[i].grid(True)    
    #plt.tight_layout()
    #st.pyplot(fig) -------> Removing beacuse of App loading issues

#####################################################################################################################################
    
    st.subheader("Temperature vs Pollutant Levels")
    st.markdown("These plots help to visually assess Temperature changes over time (e.g., summer vs winter). Some pollutants behave differently in different weather.Unusually high pollutant values at certain temperatures may stand out as anomalies")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
   # Create 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()  
    for i, pollutant in enumerate(pollutants):
        sns.scatterplot(x='TEMP', y=pollutant, data=air_quality_df, alpha=0.3, ax=axs[i])
        axs[i].set_title(f'Temperature vs {pollutant}')
        axs[i].set_xlabel('Temperature (°C)')
        axs[i].set_ylabel(f'{pollutant} (µg/m³)')
        axs[i].grid(True)    
    plt.tight_layout()
    st.pyplot(fig)

##################################################################################################################################
    
    st.subheader("Scatter Plots: Wind Speed vs Pollutants")
    st.markdown("These plots helps to visually explore how wind speed influence the concentration levels of different pollutants. Correlation values gives a quantitative sense of how strongly wind speed is associated with changes in pollutant levels. A negative correlation may indicate that higher wind speed disperses pollutants.")
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
   # Create 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()  
    for i, pollutant in enumerate(pollutants):
        correlation = air_quality_df['WSPM'].corr(air_quality_df[pollutant])
        sns.scatterplot(x='WSPM', y=pollutant, data=air_quality_df, alpha=0.3, color='green', ax=axs[i])
        axs[i].set_title(f'WSPM vs {pollutant}\nCorrelation = {correlation:.2f}', fontsize=11)
        axs[i].set_xlabel('Wind Speed (m/s)')
        axs[i].set_ylabel(f'{pollutant} (µg/m³)')
        axs[i].grid(True)   
    plt.tight_layout()
    st.pyplot(fig)
# ----------------------------------------
# Modeling Page
# ----------------------------------------
elif page == "Modeling and Prediction":
    st.title("Modeling and Prediction")

# Feature selection
features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
X = air_quality_df[features]
y = air_quality_df['PM2.5']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Streamlit UI
st.subheader("Comparison of Machine Learning Models")

st.markdown("### Model Performance Metrics")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Linear Regression")
    st.write(f"**R² Score:** {r2_lr:.2f}")
    st.write(f"**RMSE:** {rmse_lr:.2f}")
    st.write(f"**MSE:** {mse_lr:.2f}")

with col2:
    st.markdown("#### Random Forest")
    st.write(f"**R² Score:** {r2_rf:.2f}")
    st.write(f"**RMSE:** {rmse_rf:.2f}")
    st.write(f"**MSE:** {mse_rf:.2f}")

# Best Model Determination and Explanation
st.markdown("### Best Model Analysis")

if r2_rf > r2_lr:
    st.success("**Random Forest Regressor** outperforms Linear Regression based on the evaluation metrics.")

    st.markdown("""
    **Why Random Forest performs better:**
    - **Non-linearity handling**: Random Forest can model complex, non-linear relationships between features and PM2.5 levels, which Linear Regression cannot.
    - **Feature interactions**: It captures interactions between features (e.g., how temperature and CO levels jointly influence air quality).
    - **Robust to noise and outliers**: Ensemble nature helps smooth out anomalies.
    - **High R² Score**: This means it explains more variance in the data than Linear Regression.

    > **R² Score (RF):** {:.2f}  
    > **RMSE (RF):** {:.2f}  
    > **Compared to Linear Regression R²:** {:.2f}
    """.format(r2_rf, rmse_rf, r2_lr))
else:
    st.success("**Linear Regression** is the better model based on R² Score.")

    st.markdown("""
    **Why Linear Regression might perform better here:**
    - **Simplicity**: If the relationships between variables are mostly linear, this model can generalize well.
    - **Interpretability**: Coefficients give a clear understanding of how each feature affects PM2.5.
    - **Lower complexity**: Ideal when overfitting is a concern and dataset is relatively clean.

    > **R² Score (LR):** {:.2f}  
    > **RMSE (LR):** {:.2f}  
    > **Compared to Random Forest R²:** {:.2f}
    """.format(r2_lr, rmse_lr, r2_rf))

st.markdown("**Note:** R² Score indicates how well the model explains variance in PM2.5. RMSE/MSE represent error magnitude.")

# Feature Importance Plot for Random Forest
st.markdown("### Feature Importance - Random Forest")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature', hue='Feature', palette='Blues_d', ax=ax, legend=False)
ax.set_title('Feature Importance - Random Forest')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.grid(True)
st.pyplot(fig)

# Feature Importance Plot for Linear Regression (using absolute coefficient values)
st.markdown("### Feature Importance - Linear Regression")
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_,
    'Importance': np.abs(lr_model.coef_)
}).sort_values(by='Importance', ascending=False)

fig_lr, ax_lr = plt.subplots(figsize=(8, 5))
sns.barplot(data=coefficients, x='Importance', y='Feature', hue='Feature', palette='Greens_d', ax=ax_lr, legend=False)
ax_lr.set_title('Feature Importance - Linear Regression')
ax_lr.set_xlabel('Absolute Coefficient Value')
ax_lr.set_ylabel('Features')
ax_lr.grid(True)
st.pyplot(fig_lr)

st.markdown("### Actual vs Predicted PM2.5")
st.markdown("""
    <style>
    .streamlit-expanderHeader {
        color: #FFFFFF 
    }
    .css-1l5x3x6 {
        color: #FFFFFF 
    }
    </style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Linear Regression", "Random Forest"])

with tab1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred_lr, alpha=0.5, color="green")
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    ax1.set_xlabel("Actual PM2.5")
    ax1.set_ylabel("Predicted PM2.5")
    ax1.set_title("Linear Regression")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred_rf, alpha=0.5, color="blue")
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    ax2.set_xlabel("Actual PM2.5")
    ax2.set_ylabel("Predicted PM2.5")
    ax2.set_title("Random Forest")
    st.pyplot(fig2)



       
   