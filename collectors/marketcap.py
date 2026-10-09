import pandas as pd
import yfinance as yf
from tqdm import tqdm

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}

sp500 = pd.read_html(url, storage_options=headers)[0]
tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.B -> BRK-B

# tickers = tickers[:10]  # Limit to first 10 tickers for testing

market_caps = []
for t in tqdm(tickers, desc="Fetching market caps"):
    try:
        info = yf.Ticker(t).fast_info
        market_caps.append((t, info.get("marketCap", None)))
    except Exception as e:
        print(f"Error fetching data for {t}: {e}")
        market_caps.append((t, None))

df_market_caps = pd.DataFrame(market_caps, columns=["Ticker", "Market Cap"])
df_market_caps.dropna(inplace=True)
df_market_caps = df_market_caps.sort_values(
    by="Market Cap", ascending=False
).reset_index(drop=True)
print(df_market_caps.head())

df_market_caps.to_csv("data/market_caps.csv", index=False)
