import pandas as pd
import yfinance as yf
from tqdm import tqdm

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}

sp500 = pd.read_html(url, storage_options=headers)[0]
tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.B -> BRK-B

# tickers = tickers[:10]  # Limit to first 10 tickers for testing

output_path = "data/financial_statements.csv"

records = []

for symbol in tqdm(tickers, desc="Fetching financial statements"):
    ticker = yf.Ticker(symbol)
    try:
        statements = {
            "balance_sheet": ticker.balance_sheet,
            "cash_flow": ticker.cash_flow,
            "income_statement": ticker.income_stmt,
        }

        for stmt_name, df in statements.items():
            if df is not None and not df.empty:
                stacked = df.stack(future_stack=True).dropna()

                for (metric, date), value in stacked.items():
                    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                    records.append(
                        {
                            "symbol": symbol,
                            "statement": stmt_name,
                            "metric": metric,
                            "date": date_str,
                            "value": value,
                        }
                    )
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

df_all = pd.DataFrame(records)
df_all.to_csv(output_path, index=False, header=False)
