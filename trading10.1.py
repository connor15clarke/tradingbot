import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from enum import Enum
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import websocket
import json
import threading


class BotLogger:
    def __init__(self):
        self.log = []

    def record(self, event, msg):
        self.log.append((event, msg))
    
    def output_log(self, print_log=True, save_to_file=None):
        output = '\n'.join([f"{event}: {msg}" for event, msg in self.log])
        if print_log:
            print(output)
        if save_to_file:
            with open(save_to_file, 'w') as f:
                f.write(output)

class Decision(Enum):
        BUY = 1
        SELL = 2
        HOLD = 3

class TradingBot:
    def __init__(self, tickers, alpaca_api_key, alpaca_api_secret, base_url='https://paper-api.alpaca.markets'):
        self.api = tradeapi.REST(alpaca_api_key, alpaca_api_secret, base_url, api_version='v2')
        self.api_key = 'PKJACFWB5U4H52OAS0SY'
        self.api_secret = 'cYQf2wmUAImmF39niytdj0Wfvr3vkBhXmbrwImO4'
        self.stock_data = {}  # Initialize the stock_data attribute as an empty dictionary
        self.tickers = tickers  # Initialize the tickers list
        self.portfolio = {'cash': 100000}
        self.initial_balance = 10000
        self.stop_loss_percentage = 0.03
        self.trailing_stop_loss_percentage = 0.08
        self.position_size_percentage = 0.02
        self.max_loss_percentage = 0.10
        self.trade_limit = 10000  # Limit trades per week
        self.trade_count = 0  # Weekly trade counter
        self.cooldown_period = 0  # Days to wait after a trade before trading again
        self.last_trade_day = {}  # Dictionary to store the last trade day for each ticker
        self.avg_volume = {}  # Add an attribute to store average volume for each ticker
        self.logger = BotLogger()

    def _setup_websocket(self):
        # Your Alpaca WebSocket URL
        ws_url = 'wss://data.alpaca.markets/stream'
        
        def on_open(ws):
            self.logger.record("WS_OPEN", "WebSocket opened.")
            # Step 3: Subscribe to real-time data
            message = {
                "action": "authenticate",
                "data": {"key_id": self.api_key, "secret_key": self.api_secret}
            }
            ws.send(json.dumps(message))
            listen_message = {
                "action": "listen",
                "data": {"streams": [f"AM.{ticker}" for ticker in self.tickers]}
            }
            ws.send(json.dumps(listen_message))
        
        def on_message(ws, message):
            # Step 4: Handle incoming data
            self._handle_realtime_data(json.loads(message))
        
        def on_close(ws):
            self.logger.record("WS_CLOSE", "WebSocket closed.")
        
        def on_error(ws, error):
            self.logger.record("WS_ERROR", f"WebSocket error: {error}")
        
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_open=on_open,
                                         on_message=on_message,
                                         on_close=on_close,
                                         on_error=on_error)

    def _handle_realtime_data(self, data):
        # Process the incoming WebSocket messages
        if 'stream' in data:
            if 'data' in data:
                # Extract the relevant data
                symbol = data['data']['S']
                close = data['data']['c']
                high = data['data']['h']
                low = data['data']['l']
                volume = data['data']['v']
                # Store the real-time data in a suitable format
                self.stock_data[symbol] = {
                    'Close': close,
                    'High': high,
                    'Low': low,
                    'Volume': volume,
                    'OBV': self._compute_obv(pd.Series([close]), pd.Series([volume]))
                }
                # You may want to trigger trading logic here

    def start(self):
        # Step 5: Start the WebSocket
        self._setup_websocket()
        # Run the websocket in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.start()


    def _compute_macd(self, prices, short_window=8, long_window=17, signal_window=9):
        short_ema = prices.ewm(span=short_window).mean()
        long_ema = prices.ewm(span=long_window).mean()
        macd = short_ema - long_ema
        macd_signal = macd.rolling(window=signal_window).mean()
        return macd, macd_signal

    def _compute_bollinger_bands(self, prices, window=10):
        sma = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = sma + (rolling_std * 1.5)
        lower_band = sma - (rolling_std * 1.5)
        return upper_band, lower_band
    
    def _compute_rsi(self, prices, window=8):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_stochastic_oscillator(self, prices, window=6):
        low_min = prices.rolling(window=window).min()
        high_max = prices.rolling(window=window).max()
        k = 100 * ((prices - low_min) / (high_max - low_min))
        return k

    def _compute_obv(self, prices, volumes):
        obv = volumes.copy()
        obv[prices < prices.shift(1)] *= -1
        return obv.cumsum()

    def _compute_support_resistance(self, prices):
        # Determine local minima for support, and local maxima for resistance
        local_min = argrelextrema(prices.values, np.less_equal, order=5)[0]
        local_max = argrelextrema(prices.values, np.greater_equal, order=5)[0]

        support_levels = prices.iloc[local_min]
        resistance_levels = prices.iloc[local_max]
        
        return support_levels, resistance_levels
    
    def _fetch_historical_volume(self, ticker, days=30):
        """Fetch the historical volume data and calculate the average volume."""
        # Fetch historical data from Alpaca
        barset = self.api.get_barset(ticker, 'day', limit=days)
        volumes = [bar.v for bar in barset[ticker]]
        
        # Calculate and return the average volume
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        self.avg_volume[ticker] = avg_volume  # Store the average volume
        return avg_volume

    def _analyze_stock(self, ticker):
        # Fetch historical data for the indicators
        historical_data = self._fetch_historical_data(ticker)
        prices = historical_data['Close']
        volumes = historical_data['Volume']

        # Compute indicators with historical data
        macd, macd_signal = self._compute_macd(prices)
        upper_band, lower_band = self._compute_bollinger_bands(prices)
        rsi = self._compute_rsi(prices)
        k = self._compute_stochastic_oscillator(prices)
        obv = self._compute_obv(prices, volumes)

        # Fetch real-time data
        real_time_data = self._fetch_realtime_data(ticker)  # This should be already implemented in your code
        current_close = real_time_data['Close']
        current_volume = real_time_data['Volume']

        # Append the real-time data point to the historical data
        prices = prices.append(pd.Series(current_close, index=[real_time_data.index[-1]]))
        volumes = volumes.append(pd.Series(current_volume, index=[real_time_data.index[-1]]))

        # Recompute the indicators now including the real-time data
        macd, macd_signal = self._compute_macd(prices)
        upper_band, lower_band = self._compute_bollinger_bands(prices)
        rsi = self._compute_rsi(prices)
        k = self._compute_stochastic_oscillator(prices)
        obv = self._compute_obv(prices, volumes)

        # Determine signals
        buy_signals = {
            'price_above_lower_band': current_close > lower_band.iloc[-1],
            'macd_above_signal': macd.iloc[-1] > macd_signal.iloc[-1],
            'rsi_below_30': rsi.iloc[-1] < 30,
            'k_below_20': k.iloc[-1] < 20,
            'volume_increase': current_volume > 1.5 * volumes.mean()
        }

        sell_signals = {
            'price_below_upper_band': current_close < upper_band.iloc[-1],
            'macd_below_signal': macd.iloc[-1] < macd_signal.iloc[-1],
            'rsi_above_70': rsi.iloc[-1] > 70,
            'k_above_80': k.iloc[-1] > 80,
            'volume_decrease': current_volume < 0.5 * volumes.mean()
        }

        # Compute the score for buy and sell signals
        buy_score = sum(buy_signals.values())
        sell_score = sum(sell_signals.values())

        # Determine the decision
        if buy_score >= 3:
            self.logger.record(f"{ticker}: BUY SIGNAL - {buy_score}")
            return Decision.BUY
        elif sell_score >= 3:
            self.logger.record(f"{ticker}: SELL SIGNAL - {sell_score}")
            return Decision.SELL
        else:
            self.logger.record(f"{ticker}: HOLD SIGNAL - No clear signal")
            return Decision.HOLD

    
    def reset_trade_count(self, date):
        if date.weekday() == 0:  
            self.trade_count = 0
    
    def execute_trades(self):
        for ticker in self.tickers: 
            if not self._can_trade(ticker):
                continue  
            decision = self._analyze_stock(ticker)
            if decision == Decision.BUY:
                quantity = self._calculate_position_size(self._fetch_realtime_data(ticker)["Close"])
                self._place_order(ticker, "buy", quantity)
            elif decision == Decision.SELL:
                quantity = self.portfolio.get(ticker, 0)
                self._place_order(ticker, "sell", quantity)
                
    def _compute_atr(self, data, window=14):
        """Compute the Average True Range (ATR) for given data."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean().fillna(tr)
        return atr
    
    def _can_trade(self, ticker, date_index):
        
        last_trade_day = self.last_trade_day.get(ticker, None)
        if last_trade_day is not None:
            if date_index - last_trade_day < self.cooldown_period:
                if self.last_trade_day.get(f"{ticker}_stop_loss_triggered", False):   
                        return True 
                return False
        return True


    def _check_trailing_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
    
   
        highest_price_key = f"{ticker}_highest_price"
        if highest_price_key not in self.portfolio:
            self.portfolio[highest_price_key] = current_price
        else:
            self.portfolio[highest_price_key] = max(self.portfolio[highest_price_key], current_price)
        
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
        
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[f"{ticker}_stop_loss_triggered"] = date_index 
            else:
                updated_trailing_stop = self.portfolio[highest_price_key] - (self.trailing_stop_loss_percentage * self.portfolio[highest_price_key])
                self.portfolio[trailing_key] = max(self.portfolio[trailing_key], updated_trailing_stop)

    
    def _calculate_position_size(self, price):
        balance = self.portfolio['cash']
        position_size = balance * self.position_size_percentage
        quantity = position_size // price 
        return int(quantity)  


    def _place_order(self, ticker, action, quantity, date_index=None):
        if self.trade_count >= self.trade_limit:
            self.logger.record("TRADE_LIMIT", f"Reached trade limit, cannot {action} {ticker}")
            return

        if quantity <= 0:
            self.logger.record("ORDER_QUANTITY", f"Cannot {action} non-positive quantity for {ticker}")
            return

        try:
            if action == "buy":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                self.portfolio[ticker] = self.portfolio.get(ticker, 0) + quantity
                self.portfolio['cash'] -= quantity * self.stock_data[ticker]["Close"]
                self.logger.record("BUY_ORDER", f"Submitted buy order for {quantity} shares of {ticker}")
            elif action == "sell":
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                self.portfolio[ticker] = self.portfolio.get(ticker, 0) - quantity
                self.portfolio['cash'] += quantity * self.stock_data[ticker]["Close"]
                self.logger.record("SELL_ORDER", f"Submitted sell order for {quantity} shares of {ticker}")
            # Increment the trade count after a successful order submission
            self.trade_count += 1
            # Update the last trade day for the ticker
            if date_index is not None:
                self.last_trade_day[ticker] = date_index
        except Exception as e:
            self.logger.record("ERROR", f"Could not place the {action} order for {ticker}: {e}")

if __name__ == "__main__":
    # Initialize your TradingBot with relevant parameters
    trading_bot = TradingBot(tickers=["TQQQ", "SQQQ"], alpaca_api_key="YOUR_API_KEY", alpaca_api_secret="YOUR_SECRET_KEY")
    trading_bot.start()  # This starts the WebSocket and your bot