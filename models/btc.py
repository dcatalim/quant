import json
import collections
import numpy as np
import requests
import websocket
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ------------------------------------------------------------------------------
# 1. Paper Trading Broker Simulation (Handles Position & Real-Time P&L)
# ------------------------------------------------------------------------------
class PaperBroker:
    def __init__(self, initial_balance=10000.0, fee_rate=0.001):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.position = 0.0          # > 0: Long, < 0: Short, == 0: Flat
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.fee_rate = fee_rate

    def update_pnl(self, current_price):
        """
        Calculates unrealized P&L, realized P&L, and total portfolio equity.
        For Shorts: P&L rises when price drops.
        """
        unrealized_pnl = 0.0
        if self.position > 0:
            # Long: profit when price > entry_price
            unrealized_pnl = (current_price - self.entry_price) * self.position
        elif self.position < 0:
            # Short: profit when price < entry_price
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)

        total_equity = self.cash + unrealized_pnl
        net_return = ((total_equity - self.initial_balance) / self.initial_balance) * 100
        return unrealized_pnl, self.realized_pnl, total_equity, net_return

    def execute_signal(self, signal, price):
        """
        signal:
          +1: Go / Stay Long
          -1: Go / Stay Short
           0: Go Flat (Close position)
        """
        # If signal matches existing position direction, do nothing
        if (signal == 1 and self.position > 0) or (signal == -1 and self.position < 0):
            return

        # Step 1: Close out existing position if changing direction or going flat
        if self.position != 0.0:
            qty = abs(self.position)
            fee = qty * price * self.fee_rate
            
            if self.position > 0:
                trade_pnl = (price - self.entry_price) * qty - fee
                print(f" [CLOSE LONG] Sold {qty:.5f} BTC @ ${price:,.2f} | P&L: ${trade_pnl:+.2f}")
            else:
                trade_pnl = (self.entry_price - price) * qty - fee
                print(f" [CLOSE SHORT] Covered {qty:.5f} BTC @ ${price:,.2f} | P&L: ${trade_pnl:+.2f}")

            self.realized_pnl += trade_pnl
            self.cash += trade_pnl  # Cash adjusted by net outcome
            self.position = 0.0
            self.entry_price = 0.0

        # Step 2: Open new position
        equity = self.cash
        if signal == 1:
            # Enter Long
            qty = (equity * 0.95) / price
            fee = qty * price * self.fee_rate
            self.cash -= fee
            self.position = qty
            self.entry_price = price
            print(f" [OPEN LONG] Bought {qty:.5f} BTC @ ${price:,.2f} | Fee: ${fee:.2f}")

        elif signal == -1:
            # Enter Short
            qty = (equity * 0.95) / price
            fee = qty * price * self.fee_rate
            self.cash -= fee
            self.position = -qty
            self.entry_price = price
            print(f" [OPEN SHORT] Sold short {qty:.5f} BTC @ ${price:,.2f} | Fee: ${fee:.2f}")

            
# ------------------------------------------------------------------------------
# 2. Live WebSocket Trading Client
# ------------------------------------------------------------------------------
class LiveTrader:
    def __init__(self, model, window_size=150, entry_threshold = 0.0008, exit_threshold = 0.0005):
        self.model = model
        self.window_size = window_size
        self.entry_threshold = entry_threshold 
        self.exit_threshold = exit_threshold   
        self.price_history = collections.deque(maxlen=window_size)
        self.scaler = MinMaxScaler()
        self.broker = PaperBroker(initial_balance=10000.0)

    def bootstrap_history(self, symbol="BTCUSDT"):
        """Pre-fills the rolling lookback window using the Binance REST API."""
        print(f"Fetching recent {self.window_size} candles to initialize buffer...")
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={self.window_size}"
        resp = requests.get(url).json()
        for kline in resp:
            self.price_history.append(float(kline[4]))  # close price
        print("Historical buffer initialized. Connecting to live stream...")

    def on_message(self, ws, message):
        event = json.loads(message)
        kline = event.get("k", {})
        
        current_price = float(kline.get("c", 0.0))
        is_candle_closed = kline.get("x", False)

        # 1. Update live ticker P&L on every tick
        unrealized, realized, equity, net_return = self.broker.update_pnl(current_price)
        print(f"\r[TICK] BTC: ${current_price:,.2f} | Unrealized: ${unrealized:+.2f} | Realized: ${realized:+.2f} | Equity: ${equity:,.2f} ({net_return:+.2f}%)", end="")

        # 2. Run model predictions only when a 1-minute candle formally closes
        if is_candle_closed:
            print("\n" + "=" * 80)
            print(f"1-Minute Candle Closed at ${current_price:,.2f}")
            self.price_history.append(current_price)

            # Format and normalize input sequence
            history_arr = np.array(self.price_history).reshape(-1, 1)
            scaled_window = self.scaler.fit_transform(history_arr)
            input_tensor = scaled_window.reshape(1, self.window_size, 1)

            # Predict next candle's close
            scaled_pred = self.model.predict(input_tensor, verbose=0)[0, 0]
            pred_price = self.scaler.inverse_transform([[scaled_pred]])[0, 0]
            expected_change = (pred_price - current_price) / current_price

            print(f"Model Predicted Next Close: ${pred_price:,.2f} (Expected: {expected_change:+.3%})")

            # -------------------------------------------------------------
            # Position-Aware Logic: Don't exit winning trades on minor noise
            # -------------------------------------------------------------
            current_pos = self.broker.position

            if current_pos == 0:
                if expected_change > self.entry_threshold:
                    print("Signal: [STRONG BULLISH] -> Open Long")
                    self.broker.execute_signal(1, current_price)
                elif expected_change < -self.entry_threshold:
                    print("Signal: [STRONG BEARISH] -> Open Short")
                    self.broker.execute_signal(-1, current_price)
                else:
                    print("Signal: [NEUTRAL] -> Stay flat (waiting for setup)")

            elif current_pos > 0:
                # Currently Long: Only exit/flip if model clearly flips bearish
                if expected_change < -self.exit_threshold:
                    print("Signal: [REVERSAL TO BEARISH] -> Close Long / Flip Short")
                    self.broker.execute_signal(-1, current_price)
                else:
                    print("Signal: [HOLD LONG] -> Trend intact or noise, riding position")

            elif current_pos < 0:
                # Currently Short: Only exit/flip if model clearly flips bullish
                if expected_change > self.exit_threshold:
                    print("Signal: [REVERSAL TO BULLISH] -> Close Short / Flip Long")
                    self.broker.execute_signal(1, current_price)
                else:
                    print("Signal: [HOLD SHORT] -> Downside continues, riding position")

            # Final P&L report for this closed candle
            unrealized, realized, equity, net_return = self.broker.update_pnl(current_price)
            print(f"Status -> Cash: ${self.broker.cash:,.2f} | Open BTC: {self.broker.position:.5f} | Total Equity: ${equity:,.2f}")
            print("=" * 80)

    def start(self):
        self.bootstrap_history()
        stream_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        ws = websocket.WebSocketApp(
            stream_url,
            on_message=self.on_message,
            on_error=lambda ws, err: print(f"\nWebSocket Error: {err}"),
            on_close=lambda ws, status, msg: print(f"\nConnection Closed ({status}: {msg})")
        )
        ws.run_forever()

# ------------------------------------------------------------------------------
# 3. Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load previously trained CAT model (or pass instantiated model)
    try:
        model = tf.keras.models.load_model("cat_model.h5")
        print("Loaded trained model 'cat_model.h5'.")
    except Exception:
        print("No saved 'cat_model.h5' found. Initializing a mock 200k-weight architecture...")
        from tensorflow.keras import layers, models
        model = models.Sequential([
            layers.Input(shape=(150, 1)),
            layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(96, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)
        ])

    trader = LiveTrader(model=model, window_size=150, entry_threshold=0.0008, exit_threshold=0.0005)
    trader.start()