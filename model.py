import requests
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from datetime import timedelta


def predict_stock_price(symbol, days):
    symbol = symbol.strip().upper()

    # Convert AAPL.US to AAPL for Yahoo Finance
    ticker = symbol

    if ticker.endswith(".US"):
        ticker = ticker[:-3]

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}?range=1y&interval=1d"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    if response.status_code != 200:
        raise ValueError(
            "An kasa samun bayanan stock. "
            "Duba internet ko stock symbol."
        )

    result = response.json()

    chart = result.get("chart", {})
    results = chart.get("result")

    if not results:
        raise ValueError(
            "Ba a samu bayanan wannan stock ba. "
            "Gwada AAPL.US ko MSFT.US."
        )

    stock = results[0]

    timestamps = stock.get("timestamp", [])
    quotes = stock.get("indicators", {}).get("quote", [])

    if not timestamps or not quotes:
        raise ValueError(
            "Bayanan stock ba su wadatar ba."
        )

    closes = quotes[0].get("close", [])

    data = pd.DataFrame({
        "Date": pd.to_datetime(
            timestamps,
            unit="s",
            utc=True
        ).date,
        "Close": closes
    })

    data = data.dropna(subset=["Close"])
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    if len(data) < 30:
        raise ValueError(
            "Bayanan stock sun yi kadan. "
            "Gwada wani stock symbol."
        )

    prices = data["Close"].astype(float).values

    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(
        len(prices),
        len(prices) + days
    ).reshape(-1, 1)

    predictions = model.predict(future_X)

    last_price = float(prices[-1])
    predicted_price = float(predictions[-1])

    history_dates = (
        data["Date"]
        .dt.strftime("%Y-%m-%d")
        .tail(30)
        .tolist()
    )

    history_prices = prices[-30:].round(2).tolist()

    last_date = data["Date"].iloc[-1]

    prediction_dates = [
        (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, days + 1)
    ]

    prediction_prices = predictions.round(2).tolist()

    return {
        "symbol": symbol,
        "last_price": round(last_price, 2),
        "predicted_price": round(predicted_price, 2),
        "days": days,
        "history_dates": history_dates,
        "history_prices": history_prices,
        "prediction_dates": prediction_dates,
        "prediction_prices": prediction_prices,
  }
