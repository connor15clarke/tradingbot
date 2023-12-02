import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema
from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class BotLogger:
    def __init__(self):
        self.log = []

    def record(self, event, msg):
        self.log.append((event, msg))
    
    def output_log(self, print_log=True, save_to_file="trade_log.txt"):
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
    
class IndicatorWeightOptimizer:
    def __init__(self, historical_data, lookback_period=14):
        self.historical_data = historical_data
        self.lookback_period = lookback_period
        self.model = LinearRegression()

    def calculate_indicators(self, data):
        # Compute MACD and its signal line
        short_ema = data['Close'].ewm(span=8).mean()
        long_ema = data['Close'].ewm(span=17).mean()
        macd = short_ema - long_ema
        macd_signal = macd.rolling(window=9).mean()

        # Compute Bollinger Bands
        sma = data['Close'].rolling(window=10).mean()
        rolling_std = data['Close'].rolling(window=10).std()
        upper_band = sma + (rolling_std * 1.5)
        lower_band = sma - (rolling_std * 1.5)

        # Compute RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=8, min_periods=1).mean()
        avg_loss = loss.rolling(window=8, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Compute Stochastic Oscillator
        low_min = data['Close'].rolling(window=6).min()
        high_max = data['Close'].rolling(window=6).max()
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))

        # Compute OBV
        obv = data['Volume'].copy()
        obv[data['Close'] < data['Close'].shift(1)] *= -1
        obv = obv.cumsum()

        # Compute ATR
        tr1 = abs(data['High'] - data['Low'])
        tr2 = abs(data['High'] - data['Close'].shift(1))
        tr3 = abs(data['Low'] - data['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().fillna(tr)

        # Combine all indicators into a DataFrame
        indicators = pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'Upper_Band': upper_band,
            'Lower_Band': lower_band,
            'RSI': rsi,
            'Stochastic_Oscillator': k,
            'OBV': obv,
            'ATR': atr
        })

        return indicators

    def prepare_data(self):
        features, labels = [], []

        for ticker, data in self.historical_data.items():
            indicators = self.calculate_indicators(data)

            # Adjust the range to avoid out-of-bounds access
            for i in range(self.lookback_period, len(data) - 1):
                feature_vector = indicators.iloc[i].tolist()
                features.append(feature_vector)

                # Define labels based on whether the price increases or decreases the next day
                label = 1 if data['Close'].iloc[i + 1] > data['Close'].iloc[i] else 0
                labels.append(label)

        return np.array(features), np.array(labels)
    
    def train_model(self):
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model MSE: {mse}")

    def get_weights(self):
        # Return the model coefficients (weights)
        return self.model.coef_

    def predict(self, new_data):
        indicators = self.calculate_indicators(new_data)
        features = indicators.tail(self.lookback_period).to_numpy().reshape(1, -1)
        return self.model.predict(features)


class BacktestTradingBot:
    def __init__(self, tickers, start_date, end_date, lookback_period=14):
        self.lookback_period = lookback_period
        self.stock_data = {}  # Initialize the stock_data attribute as an empty dictionary
        self.historical_data = self._fetch_historical_data(tickers, start_date, end_date)
        self.portfolio = {'cash': 10000}
        self.initial_balance = 10000
        self.stop_loss_percentage = 0.05
        self.trailing_stop_loss_percentage = 0.05
        self.position_size_percentage = 0.02
        self.max_loss_percentage = 0.10
        self.trade_limit = 4 # Limit trades per week
        self.trade_count = 0  # Weekly trade counter
        self.cooldown_period = 0  # Days to wait after a trade before trading again
        self.last_trade_day = {}  # Dictionary to store the last trade day for each ticker
        self.optimizer = IndicatorWeightOptimizer(self.historical_data)
        self.optimizer.train_model()
        self.indicator_weights = self.optimizer.get_weights()
        self.logger = BotLogger()

    def _fetch_historical_data(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        processed_data = {}
        for ticker in tickers:
            df = data[ticker][['Close', 'High', 'Low', 'Volume']]
            df['OBV'] = self._compute_obv(df['Close'], df['Volume'])
            processed_data[ticker] = df
        return processed_data

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
    
    def _determine_market_condition(self, ticker):
        """
        Determine the market condition (bullish, bearish, neutral) based on indicators.
        """
        data = self.historical_data[ticker]
        short_ma = data['Close'].rolling(window=50).mean()  # Short-term moving average
        long_ma = data['Close'].rolling(window=200).mean()  # Long-term moving average
        macd, macd_signal = self._compute_macd(data['Close'])
        rsi = self._compute_rsi(data['Close'])

        # Check for bearish market condition
        if short_ma.iloc[-1] < long_ma.iloc[-1] and macd.iloc[-1] < macd_signal.iloc[-1] and rsi.iloc[-1] > 70:
            return 'bearish'
        # Check for bullish market condition
        elif short_ma.iloc[-1] > long_ma.iloc[-1] and macd.iloc[-1] > macd_signal.iloc[-1] and rsi.iloc[-1] < 30:
            return 'bullish'
        else:
            return 'neutral'

    def _init_optimizer(self):
        # Initialize and train the optimizer
        self.optimizer = IndicatorWeightOptimizer(self.historical_data)
        self.optimizer.train_model()
        self.indicator_weights = self.optimizer.get_weights()

    def _analyze_stock(self, ticker, date_index):
        data = self.historical_data[ticker]

        # Compute indicators
        macd, macd_signal = self._compute_macd(data['Close'])
        upper_band, lower_band = self._compute_bollinger_bands(data['Close'])
        rsi = self._compute_rsi(data['Close'])
        k = self._compute_stochastic_oscillator(data['Close'])
        obv = self._compute_obv(data['Close'], data['Volume'])
        atr = self._compute_atr(data)

        # Create a dictionary of the current values of indicators
        indicators = {
            'MACD': macd.iloc[date_index],
            'MACD_Signal': macd_signal.iloc[date_index],
            'Upper_Band': upper_band.iloc[date_index],
            'Lower_Band': lower_band.iloc[date_index],
            'RSI': rsi.iloc[date_index],
            'Stochastic_Oscillator': k.iloc[date_index],
            'OBV': obv.iloc[date_index],
            'ATR': atr.iloc[date_index]
        }

        # Calculate the weighted score
        score = sum(self.indicator_weights[i] * value for i, value in enumerate(indicators.values()))

        # Make the buy or sell decision based on the score
        # Note: The thresholds here are arbitrary and should be adjusted based on backtesting results
        buy_threshold = 0.5  # Example threshold
        sell_threshold = -0.5  # Example threshold
        if score > buy_threshold:
            decision = Decision.BUY
        elif score < sell_threshold:
            decision = Decision.SELL
        else:
            decision = Decision.HOLD

        return decision

    def reset_trade_count(self, date):
        # Check if it's the start of the week, and reset trade_count
        if date.weekday() == 0:  # Monday
            self.trade_count = 0      
    
    def _can_trade(self, ticker, date_index):
        # Check if a trade can be executed based on the cooldown period
        last_trade_day = self.last_trade_day.get(ticker, None)
        if last_trade_day is not None:
            if date_index - last_trade_day < self.cooldown_period:
                if self.last_trade_day.get(f"{ticker}_stop_loss_triggered", False):   
                        return True 
                return False
        return True
    
    def execute_trades(self):
        for date_index in range(self.lookback_period, len(self.historical_data[next(iter(self.historical_data))])):
            current_date = self.historical_data[next(iter(self.historical_data))].index[date_index].date()
            self.reset_trade_count(current_date)

            for ticker in self.historical_data:
                decision = self._analyze_stock(ticker, date_index)
                current_price = self.historical_data[ticker]['Close'].iloc[date_index]

            if decision == Decision.BUY:
                quantity = self._calculate_position_size(current_price)
                self._place_order(ticker, "buy", quantity, current_price)
            elif decision == Decision.SELL:
                quantity = self.portfolio.get(ticker, 0)
                self._place_order(ticker, "sell", quantity, current_price)

            self._check_trailing_stop_loss(ticker, date_index)


    def _check_trailing_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
    
    # Update the highest observed price
        highest_price_key = f"{ticker}_highest_price"
        if highest_price_key not in self.portfolio:
            self.portfolio[highest_price_key] = current_price
        else:
            self.portfolio[highest_price_key] = max(self.portfolio[highest_price_key], current_price)
        
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
        # If current price drops below the trailing stop, sell the asset
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[f"{ticker}_stop_loss_triggered"] = date_index  # Mark that stop loss was triggered for cooldown
        # Update the trailing stop based on the highest observed price
            else:
                updated_trailing_stop = self.portfolio[highest_price_key] - (self.trailing_stop_loss_percentage * self.portfolio[highest_price_key])
                self.portfolio[trailing_key] = max(self.portfolio[trailing_key], updated_trailing_stop)

    
    def _calculate_position_size(self, price):
    #Calculate the number of shares based on the position size percentage.
        amount_to_invest = self.portfolio['cash'] * self.position_size_percentage
        number_of_shares = amount_to_invest // price  # Use // for integer division
        return int(number_of_shares)

    def _place_order(self, ticker, action, quantity, price, datetime):
        if self.trade_count >= self.trade_limit or not self._can_trade(ticker):
            return

        if action == "buy":
            order_quantity = min(quantity, self._calculate_position_size(price))
            if order_quantity > 0:
                self.portfolio[ticker] = self.portfolio.get(ticker, 0) + order_quantity
                self.portfolio['cash'] -= price * order_quantity
                self.trade_count += 1
                self.last_trade_day[ticker] = datetime.now().date()

            # Setting up stop loss and trailing stop for the purchased stock
                atr = self._compute_atr(self.historical_data[ticker]).iloc[-1]
                stop_loss_multiplier = 2
                self.portfolio[f"{ticker}_stop_loss"] = price - (atr * stop_loss_multiplier)
                self.portfolio[f"{ticker}_trailing_stop"] = price * (1 - self.trailing_stop_loss_percentage)

        elif action == "sell":
            current_quantity = self.portfolio.get(ticker, 0)
            sell_quantity = min(quantity, current_quantity)
            if sell_quantity > 0:
                self.portfolio[ticker] = current_quantity - sell_quantity
                self.portfolio['cash'] += price * sell_quantity
                self.trade_count += 1
                self.last_trade_day[ticker] = datetime.now().date()

            # Removing the stop loss and trailing stop after selling
                if f"{ticker}_stop_loss" in self.portfolio:
                    del self.portfolio[f"{ticker}_stop_loss"]
                if f"{ticker}_trailing_stop" in self.portfolio:
                    del self.portfolio[f"{ticker}_trailing_stop"]
            
    def _check_max_loss(self, initial_balance, date_index):
        if self.portfolio['cash'] < initial_balance * (1 - self.max_loss_percentage):
            for ticker in self.historical_data:
                if ticker in self.portfolio and self.portfolio[ticker] > 0:
                    self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
    
    def _check_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
        
        # Check for traditional stop-loss
        stop_loss_key = f"{ticker}_stop_loss"
        if stop_loss_key in self.portfolio:
            if current_price < self.portfolio[stop_loss_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[ticker] = date_index  # Updating the last trade day to start the cooldown
                return  # Exit the function after selling
        
        # Check for trailing stop-loss
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
            # If current price drops below the trailing stop, sell the asset
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[ticker] = date_index  # Updating the last trade day to start the cooldown
            # Update the trailing stop if the current price is higher than the previous trailing stop + a certain percentage
            elif current_price > self.portfolio[trailing_key] * (1 + self.trailing_stop_loss_percentage):
                self.portfolio[trailing_key] = current_price * (1 - self.trailing_stop_loss_percentage)

        
    def run(self):
        closing_prices = {}

        for date_index in range(self.lookback_period, len(self.historical_data[next(iter(self.historical_data))])):
            date = self.historical_data[next(iter(self.historical_data))].index[date_index]
            self.reset_trade_count(date)
            self._check_max_loss(self.initial_balance, date_index)

            decisions = {}
            for ticker in self.historical_data:
                decision = self._analyze_stock(ticker, date_index)
                decisions[ticker] = decision
                self._check_stop_loss(ticker, date_index)

            if ('TQQQ' in decisions and 'SQQQ' in decisions) and (decisions['TQQQ'] == decisions['SQQQ']):
                self.logger.record(f"Date:{date}", f"Simultaneous signal detected for TQQQ and SQQQ. Skipping trades.")
                continue

            for ticker, decision in decisions.items():
                if decision == Decision.BUY or decision == Decision.SELL:
                    self.execute_trades()

            last_day_index = len(self.historical_data[next(iter(self.historical_data))]) - 1
            if date_index == last_day_index:
                for ticker in self.historical_data:
                    closing_prices[ticker] = self.historical_data[ticker]['Close'].iloc[date_index]

        total_shares_value = sum(self.portfolio.get(ticker, 0) * price for ticker, price in closing_prices.items())
        total_portfolio_value = self.portfolio['cash'] + total_shares_value

        print(f"Closing Prices: {closing_prices}")
        print(f"Total Portfolio Value: {total_portfolio_value}")
        print(self.portfolio)
        

# Tickers and date range
tickers = ["TQQQ", "SQQQ"]
start_date = "2019-01-01"
end_date = "2022-01-01"

bot = BacktestTradingBot(tickers, start_date, end_date)
bot.execute_trades()
bot.logger.output_log(print_log=True, save_to_file='trading_bot_log.txt')
bot.run()