
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="",
    layout="wide"
)

st.title(" Sales Forecasting ML App")
st.write(
    "Upload monthly sales data, compare ML models, "
    "and forecast future sales."
)

st.sidebar.header("Settings")
forecast_months = st.sidebar.slider(
    "Forecast months",
    min_value=1,
    max_value=12,
    value=6
)

uploaded_file = st.file_uploader(
    "Upload your sales CSV file",
    type=["csv"]
)

# Load uploaded data or use the project's sales.csv
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    try:
        raw_df = pd.read_csv("sales.csv")
        st.info("Using the project's sample sales.csv file.")
    except FileNotFoundError:
        st.warning("Please upload a CSV with Month and Sales columns.")
        st.stop()

# Check required columns
if not {"Month", "Sales"}.issubset(raw_df.columns):
    st.error("CSV must contain columns named Month and Sales.")
    st.stop()

# Clean and prepare data
df = raw_df[["Month", "Sales"]].copy()
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df = df.dropna(subset=["Month", "Sales"])
df = df.groupby("Month", as_index=False)["Sales"].sum()
df = df.sort_values("Month").reset_index(drop=True)

if len(df) < 15:
    st.error("Please provide at least 15 monthly records.")
    st.stop()

st.subheader("1. Sales Data")
st.dataframe(df.tail(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total sales", f"{df['Sales'].sum():,.2f}")
col2.metric("Average monthly sales", f"{df['Sales'].mean():,.2f}")
col3.metric("Number of months", len(df))

# Plot historical sales
st.subheader("2. Historical Sales Trend")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Month"], df["Sales"], marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.grid(True)
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

# Feature engineering
model_df = df.copy()
model_df["Year"] = model_df["Month"].dt.year
model_df["Month_Number"] = model_df["Month"].dt.month
model_df["Lag_1"] = model_df["Sales"].shift(1)
model_df["Rolling_Mean_3"] = (
    model_df["Sales"].shift(1).rolling(3).mean()
)
model_df = model_df.dropna().reset_index(drop=True)

feature_cols = [
    "Year",
    "Month_Number",
    "Lag_1",
    "Rolling_Mean_3"
]

if len(model_df) < 10:
    st.error("Not enough usable records after feature engineering.")
    st.stop()

# Chronological train-test split
split_idx = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_idx]
test_df = model_df.iloc[split_idx:]

X_train = train_df[feature_cols]
y_train = train_df["Sales"]
X_test = test_df[feature_cols]
y_test = test_df["Sales"]

# Train and compare models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

results_df = pd.DataFrame(results).sort_values("MAE")

st.subheader("3. Model Performance")
st.dataframe(results_df, use_container_width=True)

best_name = results_df.iloc[0]["Model"]
st.success(f"Best model on the test period by MAE: {best_name}")

# Actual vs predicted
st.subheader("4. Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(test_df["Month"], y_test.values, marker="o", label="Actual")

for name, pred in predictions.items():
    ax.plot(test_df["Month"], pred, marker=".", label=name)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True)
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

# Refit the selected model on all available feature rows
best_model = models[best_name]
best_model.fit(model_df[feature_cols], model_df["Sales"])

# Forecast future months recursively
history = df[["Month", "Sales"]].copy()
forecast_rows = []

for _ in range(forecast_months):
    next_month = history["Month"].max() + pd.offsets.MonthBegin(1)

    input_row = pd.DataFrame({
        "Year": [next_month.year],
        "Month_Number": [next_month.month],
        "Lag_1": [history["Sales"].iloc[-1]],
        "Rolling_Mean_3": [history["Sales"].tail(3).mean()]
    })

    predicted = float(best_model.predict(input_row[feature_cols])[0])
    predicted = max(0, predicted)

    forecast_rows.append({
        "Month": next_month,
        "Predicted_Sales": round(predicted, 2)
    })

    history = pd.concat([
        history,
        pd.DataFrame({
            "Month": [next_month],
            "Sales": [predicted]
        })
    ], ignore_index=True)

forecast_df = pd.DataFrame(forecast_rows)

st.subheader("5. Future Sales Forecast")
st.dataframe(forecast_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Month"], df["Sales"], marker="o", label="Historical sales")
ax.plot(
    forecast_df["Month"],
    forecast_df["Predicted_Sales"],
    marker="s",
    linestyle="--",
    label="Forecast sales"
)
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True)
fig.autofmt_xdate()
st.pyplot(fig)
plt.close(fig)

csv_data = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name="sales_forecast.csv",
    mime="text/csv"
)

st.caption(
    "Note: This is an educational forecasting project. "
    "Predictions are estimates, not guaranteed future sales."
)