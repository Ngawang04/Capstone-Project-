import json
import os

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb  # The Weights & Biases SDK

PROCESSED_DATA_PATH = 'data/processed/df_final_full.parquet'
SELL_PRICES_PATH = 'data/sell_prices.csv'
MODEL_SAVE_PATH = 'models/final_prophet_model.json'


def load_and_aggregate_data(state_id: str) -> pd.DataFrame:
    """
    Load the processed sales data, merge in sell prices, and aggregate daily totals
    for a target state (e.g., 'CA').
    """
    try:
        df_full = pd.read_parquet(PROCESSED_DATA_PATH)
        df_prices = pd.read_csv(SELL_PRICES_PATH)
    except FileNotFoundError:
        print("ERROR: Data files not found. Check paths.")
        return pd.DataFrame()

    df_filtered = df_full[df_full['state_id'] == state_id].copy()
    if df_filtered.empty:
        print(f"WARNING: No rows found for state {state_id}.")
        return pd.DataFrame()

    df_merged = pd.merge(
        df_filtered,
        df_prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )

    df_aggregate = (
        df_merged.groupby('ds')
        .agg(
            y=('y', 'sum'),
            is_event=('is_event', 'max'),
            avg_sell_price=('sell_price', 'mean'),
        )
        .reset_index()
        .sort_values('ds')
    )

    df_aggregate['avg_sell_price'] = (
        df_aggregate['avg_sell_price']
        .fillna(method='ffill')
        .fillna(method='bfill')
    )
    df_aggregate['ds'] = pd.to_datetime(df_aggregate['ds'])

    print(f"Data aggregation complete for State {state_id}. Rows: {len(df_aggregate)}.")
    return df_aggregate


def create_time_features_for_ml(df: pd.DataFrame):
    """Extract required numerical features from the 'ds' column for linear models."""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])

    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month

    X = df.drop(columns=['ds', 'y'])
    y = df['y']

    X = pd.get_dummies(X, columns=['day_of_week', 'month'], drop_first=True)
    return X, y


def train_prophet_model(df_train: pd.DataFrame):
    """Initialise and train the Prophet model with exogenous regressors."""
    model = Prophet(
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.add_regressor('is_event')
    model.add_regressor('avg_sell_price')
    model.fit(df_train)
    return model


def train_ridge_model(df_aggregate: pd.DataFrame, state_id: str):
    """Train the Ridge Regression benchmark (Dhruv's solution)."""
    if df_aggregate.empty:
        print(f"WARNING: Skipping Ridge benchmark; no data for {state_id}.")
        return {}

    val_period = 28

    with wandb.init(project="InventoryAI-Capstone", name=f"Dhruv-Ridge-Benchmark-{state_id}", reinit=True) as run:
        run.config.model_type = 'Ridge Regression (Scikit-learn)'
        run.config.alpha = 1.0

        X_all, y_all = create_time_features_for_ml(df_aggregate)

        X_train = X_all.iloc[:-val_period]
        y_train = y_all.iloc[:-val_period]
        X_val = X_all.iloc[-val_period:]
        y_val = y_all.iloc[-val_period:]

        model = Ridge(alpha=run.config.alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = calculate_metrics(y_val, y_pred)
        run.log(metrics)

        print(f"\nRidge benchmark complete for {state_id}. Metrics logged to W&B.")
        return metrics


def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_safe = np.where(y_true == 0, 1, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        'MAE': round(mae, 2),
        'MAPE': round(mape, 2),
        'RMSE': round(rmse, 2),
    }


def predict_naive_seasonal(df_aggregate: pd.DataFrame, state_id: str):
    """Compute the baseline error using a 7-day seasonal naive forecast."""
    if df_aggregate.empty:
        print(f"WARNING: Skipping naive benchmark; no data for {state_id}.")
        return {}

    val_period = 28
    df_baseline = df_aggregate.copy()
    df_baseline['y_lag_7'] = df_baseline['y'].shift(7)

    with wandb.init(project="InventoryAI-Capstone", name=f"Divij-Naive-Baseline-{state_id}", reinit=True) as run:
        run.config.model_type = 'Seasonal Naive (Lag 7)'
        run.config.state = state_id

        df_val = df_baseline.iloc[-val_period:].copy()
        y_true = df_val['y'].values
        y_pred = df_val['y_lag_7'].values

        metrics = calculate_metrics(y_true, y_pred)
        run.log(metrics)

        print(f"\nNaive baseline complete for {state_id}. Metrics logged to W&B.")
        return metrics


def save_prophet_model(model_object, path=MODEL_SAVE_PATH):
    """Persist the final Prophet model to JSON for deployment."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fout:
        json.dump(model_to_json(model_object), fout)
    print(f"\nSUCCESS: Final model saved for deployment at: {path}")


if __name__ == '__main__':
    TEST_STATE = 'CA'
    VAL_PERIOD = 28

    df_aggregate = load_and_aggregate_data(TEST_STATE)
    if df_aggregate.empty:
        raise SystemExit(f"No data available for state {TEST_STATE}.")

    df_train_only = df_aggregate.iloc[:-VAL_PERIOD].copy()
    df_validation_true = df_aggregate.iloc[-VAL_PERIOD:].copy()

    with wandb.init(project="InventoryAI-Capstone", name="Ngawang-Prophet-Baseline-CA") as run:
        run.config.model_type = 'Prophet'
        run.config.validation_days = VAL_PERIOD
        run.config.regressors = ['is_event', 'avg_sell_price']

        prophet_model = train_prophet_model(df_train_only)

        df_validation_future = prophet_model.make_future_dataframe(periods=VAL_PERIOD, include_history=False)
        df_validation_future['is_event'] = df_validation_true['is_event'].reset_index(drop=True).values
        df_validation_future['avg_sell_price'] = df_validation_true['avg_sell_price'].reset_index(drop=True).values

        forecast_validation = prophet_model.predict(df_validation_future)

        y_true = df_validation_true['y'].values
        y_pred = forecast_validation['yhat'].values

        metrics = calculate_metrics(y_true, y_pred)
        run.log(metrics)

        print("\nValidation complete. Metrics logged to W&B:")
        print(metrics)

    TEST_STATE_NAIVE = 'WI'
    df_divij = load_and_aggregate_data(TEST_STATE_NAIVE)
    naive_metrics = predict_naive_seasonal(df_divij, TEST_STATE_NAIVE)
    if naive_metrics:
        print(f"\nFinal naive baseline metrics ({TEST_STATE_NAIVE}): {naive_metrics}")

    TEST_STATE_DHRUV = 'CA'
    df_dhruv = load_and_aggregate_data(TEST_STATE_DHRUV)
    ridge_metrics = train_ridge_model(df_dhruv, TEST_STATE_DHRUV)
    if ridge_metrics:
        print(f"\nDhruv's Ridge metrics ({TEST_STATE_DHRUV}): {ridge_metrics}")

    df_train_full = df_aggregate.copy()
    with wandb.init(project="InventoryAI-Capstone", name="FINAL-PROPHET-MODEL-CA", reinit=True) as run:
        run.config.model_type = 'Prophet (Final Deployment Version)'
        run.config.regressors = ['is_event', 'avg_sell_price']

        final_prophet_model = train_prophet_model(df_train_full)
        save_prophet_model(final_prophet_model)

    print("\nModel finalisation complete. Model ready for UI integration.")
