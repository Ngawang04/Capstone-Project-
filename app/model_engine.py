import pandas as pd
import numpy as np
from prophet import Prophet
import os
from sklearn.metrics import mean_absolute_error

# Define the path for the processed data file (Parquet is best)
PROCESSED_DATA_PATH = '/Users/ngachoe2002/Desktop/inventoryai/data/processed/df_final_full.parquet'

def generate_forecast(item_id: str, store_id: str, periods: int = 28):
    """
    Loads the full processed M5 data, filters it, trains a Prophet model,
    and generates a forecast for the specified period.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        # This will be useful when the app runs
        print(f"ERROR: Processed data not found at {PROCESSED_DATA_PATH}. Run data pipeline first.")
        return None

    # 1. Load the entire clean dataset efficiently
    df_full = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"Loaded full data. Total rows: {len(df_full)}")

    # 2. Filter to a single series for baseline training (Required for initial testing speed)
    df_series = df_full[
        (df_full['item_id'] == item_id) & 
        (df_full['store_id'] == store_id)
    ].copy()

    # Select only the required columns for Prophet (ds and y)
    df_train = df_series[['ds', 'y']]
    
    # 3. Initialize and Train the Model (Baseline settings)
    m = Prophet(
        seasonality_mode='multiplicative',
        weekly_seasonality=True, 
        daily_seasonality=False
    )
    
    m.fit(df_train)
    print(f"Model trained successfully on {store_id}/{item_id}.")
    
    # 4. Create Future Dataframe for Forecasting
    future = m.make_future_dataframe(periods=periods)
    
    # 5. Generate the Forecast
    forecast = m.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def evaluate_forecast(y_true, y_pred):
    """Calculates Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)."""
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE safely (avoiding division by zero)
    y_true_safe = np.where(y_true == 0, 1, y_true) 
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return {'MAE': round(mae, 2), 'MAPE': round(mape, 2)}

# --- Test Block (Run this when you execute the script) ---
if __name__ == '__main__':
    TEST_ITEM = 'HOBBIES_1_001'
    TEST_STORE = 'CA_1'
    
    # Generate forecast
    forecast_results = generate_forecast(TEST_ITEM, TEST_STORE, periods=28)
    
    if forecast_results is not None:
        print("\nSample 28-Day Forecast:")
        print(forecast_results.tail(28))
        
        # --- Dummy Evaluation for Demonstration ---
        # NOTE: For real evaluation, you need a test set, but this confirms the function works.
        # metrics = evaluate_forecast(actual_test_data['y'], forecast_results['yhat'])
        # print(f"\nSample Metrics: {metrics}")