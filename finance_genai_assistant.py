# ============================================
# Finance GenAI Assistant – India VIX Analysis
# ============================================

import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import uvicorn 

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# ============================================
# Load Environment Variables
# ============================================

# Load .env file (contains GOOGLE_API_KEY and MODEL_NAME)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)


# ============================================
# VIX Forecasting and Interpretation Functions
# ============================================

def predict_next_day_close(data):
    """
    Predicts the next day's India VIX value based on the provided inputs:
    open_price, high_price, low_price, and prev_close.
    """
    open_price, high_price, low_price, prev_close = (
        data["open_price"],
        data["high_price"],
        data["low_price"],
        data["prev_close"],
    )

    model = joblib.load("india_vix_forecast_model.pkl")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    overnight_change = (open_price - prev_close) / prev_close * 100

    new_df = pd.DataFrame(
        [
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Prev. Close": prev_close,
                "Overnight_NIFTY_Change": overnight_change,
            }
        ]
    )

    new_scaled = scaler_X.transform(new_df)
    pred_scaled = model.predict(new_scaled)
    predicted_close = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

    return float(predicted_close[0][0])


def interpret_vix_change(current_vix: float, predicted_vix: float):
    """
    Interprets the predicted change in VIX and returns a short summary.
    """
    change_pct = ((predicted_vix - current_vix) / current_vix) * 100
    if change_pct > 2:
        interpretation = "Higher volatility expected."
    elif change_pct < -2:
        interpretation = "Lower volatility expected."
    else:
        interpretation = "Stable volatility outlook."
    return interpretation, round(change_pct, 2)


def forecast_and_interpret_vix(data):
    """
    Forecasts the next day's India VIX and provides a brief interpretation.
    Which Takes Input dict as data which consist of
    data["open_price"]
    data["high_price"]
    data["low_price"]
    data["prev_close"]

    Steps:
    1. Uses `predict_next_day_close()` to get the forecast.
    2. Interprets it using `interpret_vix_change()`.
    3. Returns a clear financial summary.
    """
    predicted_vix = predict_next_day_close(data)
    interpretation, change_pct = interpret_vix_change(data["prev_close"], predicted_vix)

    response = (
        f"Based on today's India VIX data (Open: {data['open_price']}, High: {data['high_price']}, "
        f"Low: {data['low_price']}, Prev Close: {data['prev_close']}), "
        f"the forecast for tomorrow's VIX is {predicted_vix:.2f} ({change_pct:+.2f}%). "
        f"Interpretation: {interpretation}"
    )
    return response


# ============================================
# Historical Data Retrieval Tool
# ============================================
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def get_current_vix(date_range: str = "latest"):
    """
    Fetches India VIX data from CSV for the desired timeframe.

    Supported patterns:
    - "latest" / "current" / "today"
    - "7d"  -> last 7 days
    - "3w"  -> last 3 weeks
    - "2m"  -> last 2 months
    - "YYYY-MM-DD to YYYY-MM-DD" -> custom date range

    Returns a dict safe for tool usage.
    """

    # Load dataset
    df = pd.read_csv("sample_dataset_23-25.csv")   # <-- Update path
    df["Date "] = pd.to_datetime(df["Date "])
    df = df.sort_values("Date ")

    date_range = date_range.lower().strip()

    # 1️⃣ LATEST
    if date_range in ["latest", "current", "today"]:
        row = df.iloc[-1]
        return {
            "date": str(row["Date "].date()),
            "open_price": float(row["Open "]),
            "high_price": float(row["High "]),
            "low_price": float(row["Low "]),
            "prev_close": float(row["Prev. Close "]),
        }

    # 2️⃣ N DAYS (e.g., "7d")
    if date_range.endswith("d") and date_range[:-1].isdigit():
        n_days = int(date_range[:-1])
        end_date = df["Date "].max()
        start_date = end_date - timedelta(days=n_days)

        filtered = df[(df["Date "] >= start_date) & (df["Date "] <= end_date)]
        return {"from": str(start_date.date()), "to": str(end_date.date()), "data": filtered.to_dict("records")}

    # 3️⃣ N WEEKS (e.g., "3w")
    if date_range.endswith("w") and date_range[:-1].isdigit():
        n_weeks = int(date_range[:-1])
        end_date = df["Date "].max()
        start_date = end_date - timedelta(weeks=n_weeks)

        filtered = df[(df["Date "] >= start_date) & (df["Date "] <= end_date)]
        return {"from": str(start_date.date()), "to": str(end_date.date()), "data": filtered.to_dict("records")}

    # 4️⃣ N MONTHS (e.g., "2m")
    if date_range.endswith("m") and date_range[:-1].isdigit():
        n_months = int(date_range[:-1])
        end_date = df["Date "].max()
        start_date = end_date - relativedelta(months=n_months)

        filtered = df[(df["Date "] >= start_date) & (df["Date "] <= end_date)]
        return {"from": str(start_date.date()), "to": str(end_date.date()), "data": filtered.to_dict("records")}

    # 5️⃣ CUSTOM RANGE ("YYYY-MM-DD to YYYY-MM-DD")
    if "to" in date_range:
        try:
            start_str, end_str = [x.strip() for x in date_range.split("to")]
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")

            filtered = df[(df["Date "] >= start_date) & (df["Date "] <= end_date)]
            return {"from": start_str, "to": end_str, "data": filtered.to_dict("records")}

        except Exception:
            raise ValueError(
                "Invalid range format. Expected: 'YYYY-MM-DD to YYYY-MM-DD'"
            )

    # ❌ Unsupported format
    raise ValueError(
        "Invalid date_range format.\n"
        "Allowed formats: 'latest', 'today', '7d', '3w', '2m', "
        "'YYYY-MM-DD to YYYY-MM-DD'."
    )



# ============================================
# Agent Setup
# ============================================

system_prompt = """
You are a financial market assistant providing analysis and insights on India VIX and market volatility.

Important Prediction Rules:
1. If the user asks for any kind of VIX prediction, forecasting, expected VIX, or “what will VIX be”, 
   you MUST FIRST call the `get_current_vix` tool to retrieve the latest VIX values.

2. If the query is specifically about predicting “next day”, “tomorrow”, “upcoming session”, or similar:
   - ALWAYS call `get_current_vix` with input = "latest".
   - The next-day forecast MUST be based on the VIX values fetched for today's date.
   - After retrieving today’s data, call the `forecast_and_interpret_vix` tool to generate the prediction.

3. If the prediction refers to a future date but the user also specifies a timeframe (e.g., “prediction based on this week”):
   - First fetch the correct range using `get_current_vix` (e.g., “1w”, “7d”, “1m”).
   - Then use that data for forecasting.

Historical / Range Understanding:
- “today” → "latest"
- “this week” → "1w"
- “last 7 days” → "7d"
- “one month” → "1m"
- Custom ranges → "YYYY-MM-DD to YYYY-MM-DD"

Capabilities:
1. Fetch historical or recent VIX data using the `get_current_vix` tool.
2. Forecast next-day VIX values using the trained ML model (`forecast_and_interpret_vix` tool).
3. Analyze changes in VIX across date ranges and summarize trends.

Trend Interpretation Guidelines:
- If VIX increases → say “Higher volatility expected.”
- If VIX decreases → say “Lower volatility expected.”
- If change is small → say “Stable volatility outlook.”
- If asked "why VIX is rising/falling", use historical data to explain direction + % change.

General Knowledge:
You can answer broader financial questions like:
- What does a rising VIX indicate?
- How does NIFTY affect volatility?
- Why volatility matters for traders and investors.

Keep responses concise, data-backed, and easy to understand.

"""

tools = [forecast_and_interpret_vix, get_current_vix]

agent = create_react_agent(
    llm,
    tools
)

# ============================================
# Example Query
# ============================================
@app.post("/chat")
def main(query:str = Form(...)):
    response = agent.invoke(
        {
            "messages": [
                {"role":"system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }
    )
    print(response["messages"])
    final_res = response["messages"][-1].content
    if type(final_res) == list:
        return final_res[-1]["text"]
    else:
        return final_res
    
# --- UI Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def serve_chatbot_ui(request: Request):
    """
    Serves the HTML template for the chatbot UI.
    """
    return templates.TemplateResponse("chatbot.html", {"request": request})
    

if __name__ == "__main__":
    uvicorn.run("finance_genai_assistant:app",host="0.0.0.0",port=1525,reload=True,log_level="debug")
