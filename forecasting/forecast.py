"""
forecasting/forecast.py - Demand Forecasting with Leakage-Safe Train/Test Split

Methodology:
- Last 4 weeks = held-out test set (leakage prevention)
- Earlier weeks = training data ONLY
- Baselines: Last Value, Moving Average (window=4)
- Advanced: Holt-Winters Exponential Smoothing (trend + seasonality)
- Metrics: MAE, MAPE per SKU
- Output: forecasting/results.csv
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "sales_history.csv"
RESULTS_PATH = Path(__file__).parent / "results.csv"
FORECAST_HORIZON = 4   # Weeks to forecast
TEST_WEEKS = 4         # Last N weeks held out as test set
MA_WINDOW = 4          # Moving average window


def load_data() -> pd.DataFrame:
    """Load and validate sales history."""
    df = pd.read_csv(DATA_PATH, parse_dates=["week"])
    df = df.sort_values(["sku", "week"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows, {df['sku'].nunique()} SKUs.")
    return df


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Handles zeros in actuals by replacing with small epsilon.
    """
    actual = np.where(actual == 0, 1e-8, actual)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def last_value_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Baseline 1: Naive forecast - repeat the last observed value.
    Simple but surprisingly competitive for stable series.
    """
    return np.full(horizon, train.iloc[-1])


def moving_average_forecast(train: pd.Series, horizon: int, window: int = MA_WINDOW) -> np.ndarray:
    """
    Baseline 2: Moving average of last `window` observations.
    Smooths out noise; better than last-value for trending series.
    """
    w = min(window, len(train))
    avg = train.iloc[-w:].mean()
    return np.full(horizon, avg)


def holt_winters_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Advanced Model: Holt-Winters Exponential Smoothing.

    Configuration:
    - trend='add': additive trend component (linear trend)
    - seasonal=None: no seasonal component (weekly data, not enough for yearly seasonality)
    - damped_trend=True: dampens long-term trend to avoid over-extrapolation
    - initialization_method='estimated': let statsmodels estimate initial states

    For series too short for HW (<6 observations), fall back to moving average.
    """
    if len(train) < 6:
        logger.warning(f"Series too short ({len(train)} obs), using MA fallback.")
        return moving_average_forecast(train, horizon)

    try:
        model = ExponentialSmoothing(
            train.values.astype(float),
            trend="add",
            seasonal=None,
            damped_trend=True,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, remove_bias=True)
        forecast = fit.forecast(horizon)
        # Clip to non-negative (demand can't be negative)
        return np.maximum(forecast, 0)
    except Exception as e:
        logger.warning(f"HW failed: {e}. Falling back to MA.")
        return moving_average_forecast(train, horizon)


def forecast_sku(sku_data: pd.DataFrame, sku: str) -> List[Dict]:
    """
    Forecast demand for a single SKU.

    Leakage Prevention Strategy:
    - Sort chronologically
    - Split: train = all rows EXCEPT last TEST_WEEKS rows
    - Test = last TEST_WEEKS rows
    - Fit models on train ONLY
    - Evaluate on test
    - Never use test data to fit or tune

    Returns a list of dicts (one per test week) for results.csv.
    """
    sku_data = sku_data.sort_values("week").reset_index(drop=True)

    if len(sku_data) <= TEST_WEEKS:
        logger.warning(f"SKU {sku}: insufficient data ({len(sku_data)} rows). Skipping.")
        return []

    # Chronological split - no data leakage
    train = sku_data.iloc[:-TEST_WEEKS]["units_sold"]
    test = sku_data.iloc[-TEST_WEEKS:]["units_sold"]
    test_weeks = sku_data.iloc[-TEST_WEEKS:]["week"].dt.strftime("%Y-%m-%d").tolist()

    # Run all three models on train data
    lv_pred = last_value_forecast(train, TEST_WEEKS)
    ma_pred = moving_average_forecast(train, TEST_WEEKS)
    hw_pred = holt_winters_forecast(train, TEST_WEEKS)

    # Compute per-forecast metrics
    actuals = test.values
    rows = []
    for i in range(TEST_WEEKS):
        rows.append({
            "sku": sku,
            "week": test_weeks[i],
            "actual": float(actuals[i]),
            "hw_prediction": round(float(hw_pred[i]), 2),
            "lv_baseline": round(float(lv_pred[i]), 2),
            "ma_baseline": round(float(ma_pred[i]), 2),
            "hw_mae": round(abs(actuals[i] - hw_pred[i]), 2),
            "lv_mae": round(abs(actuals[i] - lv_pred[i]), 2),
            "ma_mae": round(abs(actuals[i] - ma_pred[i]), 2),
            "hw_mape": round(compute_mape(np.array([actuals[i]]), np.array([hw_pred[i]])), 2),
            "lv_mape": round(compute_mape(np.array([actuals[i]]), np.array([lv_pred[i]])), 2),
            "ma_mape": round(compute_mape(np.array([actuals[i]]), np.array([ma_pred[i]])), 2),
        })

    return rows


def run_forecast() -> pd.DataFrame:
    """
    Run forecasting pipeline for all SKUs.
    Returns results DataFrame and saves to results.csv.
    """
    df = load_data()
    all_rows = []

    for sku in df["sku"].unique():
        sku_data = df[df["sku"] == sku].copy()
        logger.info(f"Forecasting SKU: {sku} ({len(sku_data)} weeks of history)")
        rows = forecast_sku(sku_data, sku)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No forecast results generated.")
        return pd.DataFrame()

    results = pd.DataFrame(all_rows)

    # Summary statistics per SKU
    summary_cols = ["hw_mae", "lv_mae", "ma_mae", "hw_mape", "lv_mape", "ma_mape"]
    summary = results.groupby("sku")[summary_cols].mean().round(2)

    logger.info("\n" + "="*60)
    logger.info("FORECAST SUMMARY (mean over 4 test weeks)")
    logger.info("="*60)
    logger.info(summary.to_string())
    logger.info("="*60)

    # Overall model comparison
    logger.info(f"\nOverall Mean MAE  — HW: {results['hw_mae'].mean():.2f} | LV: {results['lv_mae'].mean():.2f} | MA: {results['ma_mae'].mean():.2f}")
    logger.info(f"Overall Mean MAPE — HW: {results['hw_mape'].mean():.2f}% | LV: {results['lv_mape'].mean():.2f}% | MA: {results['ma_mape'].mean():.2f}%")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    logger.info(f"\nResults saved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    results = run_forecast()
    print(f"\n✅ Forecasting complete. {len(results)} rows written to forecasting/results.csv")
