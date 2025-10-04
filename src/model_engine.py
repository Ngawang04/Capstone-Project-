import pandas as pd
import numpy as np
from prophet import Prophet
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb # The Weights & Biases SDK

# Define the path to the processed parquet file
PROCESSED_DATA_PATH = '/Users/ngachoe2002/Desktop/inventoryai/data/processed/df_final_full.parquet' 

def load_and_aggregate_data(state_id: str):
    """
    Loads the full processed M5 dataset and aggregates total sales by date 
    for a single specified STATE (e.g., 'CA') for fast model testing.
    """
    try:
        # 1. Load the massive, clean Parquet file 
        df_full = pd.read_parquet(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}. Check your path.")
        return pd.DataFrame()

    # 2. FILTER: Select ONLY the rows belonging to the input state_id
    df_filtered = df_full[df_full['state_id'] == state_id].copy()

    # 3. AGGREGATE: Group the data by date ('ds') and sum up the sales ('y')
    # This creates a single time series representing the TOTAL daily sales for all CA stores combined.
    df_aggregate = df_filtered.groupby('ds').agg(
        y=('y', 'sum'),
        is_event=('is_event', 'max') 
    ).reset_index()
    
    # return the data, along with the original DataFrame for a future optimization step
    return df_aggregate, df_full.head(10) # Return a small sample of the full data for context if needed


def train_prophet_model(df_train: pd.DataFrame):
    """
    Initializes, configures, and trains the Prophet model
    using the aggregated time series data and external regressors.
    """
    # 1. Initialize Prophet with optimized seasonal settings for retail
    m = Prophet(
        seasonality_mode='multiplicative', # Good for sales data where magnitude of seasonality grows with trend
        weekly_seasonality=True, 
        yearly_seasonality=True 
    )
    
    # 2. Add the external regressor (the feature engineered from calendar.csv)
    # This tells Prophet to look at the 'is_event' column for predictive power.
    m.add_regressor('is_event') 
    
    # 3. Fit the model to the training data
    m.fit(df_train)
    
    return m

# Performance metrics for Prophet model
def calculate_metrics(y_true, y_pred):
    """Calculates Mean Absolute Error (MAE), RMSE, and Mean Absolute Percentage Error (MAPE)."""
    
    # 1. Calculate MAE: Average magnitude of errors.
    mae = mean_absolute_error(y_true, y_pred)
    
    # 2. Calculate RMSE: Penalizes larger errors more heavily.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 3. Calculate MAPE: Business-friendly error percentage, handling zero values safely.
    # np.where ensures we don't divide by zero if actual sales (y_true) are 0.
    y_true_safe = np.where(y_true == 0, 1, y_true) 
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return {
        'MAE': round(mae, 2), 
        'MAPE': round(mape, 2), 
        'RMSE': round(rmse, 2)
    }

if __name__ == '__main__':
    TEST_STATE = 'CA'
    VAL_PERIOD = 28 # Days to hold out for testing/validation

    # --- 1. Load Data ---
    df_aggregate, _ = load_and_aggregate_data(TEST_STATE)
    
    # --- 2. Split Data for Validation ---
    # a. Training Data: Use all data except the last 28 days
    df_train_only = df_aggregate.iloc[:-VAL_PERIOD].copy()
    
    # b. Validation Data: The last 28 days (TRUE values)
    df_validation_true = df_aggregate.iloc[-VAL_PERIOD:].copy()
    
    # --- 3. Initialize W&B Run ---
    with wandb.init(project="InventoryAI-Capstone", name="Ngawang-Prophet-Baseline-CA") as run:
        
        # Save key settings to W&B (for comparison later)
        run.config.model_type = 'Prophet'
        run.config.validation_days = VAL_PERIOD
        run.config.regressor_count = 1 # We only added 'is_event'

        # --- 4. TRAIN the model on the smaller training set ---
        prophet_model = train_prophet_model(df_train_only) 

        # --- 5. GENERATE PREDICTION on the validation dates ---
        
        # Create the future dataframe (must use the dates from the holdout set)
        df_validation_future = prophet_model.make_future_dataframe(periods=VAL_PERIOD, include_history=False)
        
        # CRITICAL: Add the 'is_event' regressor for the future/validation period
        df_validation_future['is_event'] = df_validation_true['is_event'].reset_index(drop=True)

        # Generate prediction
        forecast_validation = prophet_model.predict(df_validation_future)
        
        # --- 6. CALCULATE AND LOG METRICS ---
        
        # True sales values
        y_true = df_validation_true['y'].values
        # Predicted sales values
        y_pred = forecast_validation['yhat'].values
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Log the final scores to the W&B dashboard!
        run.log(metrics) 
        
        print("\nValidation Complete. Metrics Logged to W&B:")
        print(metrics)