import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Set the title of the Streamlit app
st.title("Dynamic Pricing Model")

# Function to preprocess data using Label Encoding for Vehicle_Type
def preprocess_data(data):
    # Handling missing values for numeric features
    numeric_features = data.select_dtypes(include=['float', 'int']).columns
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())
    
    # Handling missing values for categorical features
    categorical_features = data.select_dtypes(include=['object']).columns
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])
    
    # Label encode "Vehicle_Type"
    vehicle_type_mapping = {
        "Economy": 0,
        "Premium": 1
        # Add more mappings if there are additional vehicle types
    }
    if "Vehicle_Type" in data.columns:
        data['Vehicle_Type'] = data['Vehicle_Type'].map(vehicle_type_mapping)
    else:
        st.error("Column 'Vehicle_Type' not found in the dataset.")
    
    return data

# Load the main dataset
try:
    data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Dynamic\dynamic_pricing.csv")
    st.success("Main dataset loaded successfully.")
except FileNotFoundError:
    st.error("Main data file 'dynamic_pricing.csv' not found.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading main data: {e}")
    st.stop()

# Preprocess the data
data = preprocess_data(data)

# Display preprocessed data
st.subheader("Preprocessed Data")
st.dataframe(data)

# Calculate demand_multiplier based on percentile for high and low demand
high_demand_percentile = 75
low_demand_percentile = 25
data['demand_multiplier'] = np.where(
    data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
    data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
    data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile)
)

# Calculate supply_multiplier based on percentile for high and low supply
high_supply_percentile = 75
low_supply_percentile = 25
data['supply_multiplier'] = np.where(
    data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], high_supply_percentile),
    np.percentile(data['Number_of_Drivers'], high_supply_percentile) / data['Number_of_Drivers'],
    np.percentile(data['Number_of_Drivers'], low_supply_percentile) / data['Number_of_Drivers']
)

# Define price adjustment factors for high and low demand/supply
demand_threshold_low = 0.8
supply_threshold_high = 0.8

# Calculate adjusted_ride_cost for dynamic pricing
data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
    np.maximum(data['demand_multiplier'], demand_threshold_low) *
    np.maximum(data['supply_multiplier'], supply_threshold_high)
)

# Calculate the profit percentage for each ride
data['profit_percentage'] = ((data['adjusted_ride_cost'] - data['Historical_Cost_of_Ride']) / data['Historical_Cost_of_Ride']) * 100

# Identify profitable and loss rides
profitable_rides = data[data['profit_percentage'] > 0]
loss_rides = data[data['profit_percentage'] < 0]

# Calculate counts
profitable_count = len(profitable_rides)
loss_count = len(loss_rides)

# Create and display a donut chart
labels = ['Profitable Rides', 'Loss Rides']
values = [profitable_count, loss_count]
fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig_donut.update_layout(title_text='Distribution of Profitable and Loss Rides')
st.plotly_chart(fig_donut)

# Function to train the Random Forest model
def train_model(data):
    model = RandomForestRegressor(random_state=42)
    feature_columns = ["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]
    
    # Verify all feature columns exist
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        st.error(f"The following required columns are missing from the data: {missing_features}")
        return None
    
    # Check if target column exists
    if "adjusted_ride_cost" not in data.columns:
        st.error("Column 'adjusted_ride_cost' not found in the dataset.")
        return None
    
    x = data[feature_columns]
    y = data["adjusted_ride_cost"]
    
    model.fit(x, y)
    return model

# Function to predict price based on user input
def predict_price(model, number_of_riders, number_of_drivers, vehicle_type, expected_ride_duration):
    # Convert Vehicle_Type to numeric
    vehicle_type_numeric = 1 if vehicle_type == "Premium" else 0
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, expected_ride_duration]])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Streamlit UI for user inputs
st.sidebar.header("Input Parameters")

user_number_of_riders = st.sidebar.slider("Number of Riders", min_value=1, max_value=100, value=50)
user_number_of_drivers = st.sidebar.slider("Number of Drivers", min_value=1, max_value=100, value=25)
user_vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Economy", "Premium"])
expected_ride_duration = st.sidebar.slider("Expected Ride Duration (minutes)", min_value=5, max_value=60, value=30)

# Train the model
model = train_model(data)

if model:
    # Predict using user inputs
    predicted_price = predict_price(model, user_number_of_riders, user_number_of_drivers, user_vehicle_type, expected_ride_duration)
    st.sidebar.subheader("Predicted Price")
    st.sidebar.write(f"${predicted_price:.2f}")
else:
    st.sidebar.write("Model could not be trained.")

# Visualization: Actual vs Predicted Values
st.header("Actual vs Predicted Values")

# Load and preprocess test data
try:
    test_data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Dynamic\dynamic_pricing.csv")
    test_data = preprocess_data(test_data)
    st.success("Test dataset loaded and preprocessed successfully.")
except FileNotFoundError:
    st.error("Test data file 'test_data.csv' not found.")
    test_data = None
except Exception as e:
    st.error(f"An error occurred while loading test data: {e}")
    test_data = None

if model and test_data is not None:
    # Calculate demand_multiplier and supply_multiplier for test data
    test_data['demand_multiplier'] = np.where(
        test_data['Number_of_Riders'] > np.percentile(test_data['Number_of_Riders'], high_demand_percentile),
        test_data['Number_of_Riders'] / np.percentile(test_data['Number_of_Riders'], high_demand_percentile),
        test_data['Number_of_Riders'] / np.percentile(test_data['Number_of_Riders'], low_demand_percentile)
    )
    
    test_data['supply_multiplier'] = np.where(
        test_data['Number_of_Drivers'] > np.percentile(test_data['Number_of_Drivers'], high_supply_percentile),
        np.percentile(test_data['Number_of_Drivers'], high_supply_percentile) / test_data['Number_of_Drivers'],
        np.percentile(test_data['Number_of_Drivers'], low_supply_percentile) / test_data['Number_of_Drivers']
    )
    
    # Calculate adjusted_ride_cost for test data
    test_data['adjusted_ride_cost'] = test_data['Historical_Cost_of_Ride'] * (
        np.maximum(test_data['demand_multiplier'], demand_threshold_low) *
        np.maximum(test_data['supply_multiplier'], supply_threshold_high)
    )
    
    # Select features and target for test data
    feature_columns = ["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]
    missing_features_test = [col for col in feature_columns if col not in test_data.columns]
    if missing_features_test:
        st.error(f"The following required columns are missing from the test data: {missing_features_test}")
    else:
        x_test = test_data[feature_columns]
        y_test = test_data["adjusted_ride_cost"]
        
        # Predict on the test set
        y_pred = model.predict(x_test)
        
        # Create a scatter plot with actual vs predicted values
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Actual vs Predicted'
        ))
        
        # Add a line representing the ideal case
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Ideal',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            showlegend=True,
        )
        
        st.plotly_chart(fig)
