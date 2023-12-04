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

    def calculate_indicators(self, data, date_index=None):
    # Initialize an empty DataFrame
        indicators = pd.DataFrame()

    # Ensure data contains necessary columns
        if not all(column in data.columns for column in ['Close', 'High', 'Low', 'Volume']):
            print("Data does not contain all required columns: 'Close', 'High', 'Low', 'Volume'")
            return indicators  # Return the empty DataFrame

    # Convert data to float and ensure the index is datetime
        try:
            data = data.copy().astype(float)
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            print(f"Error in data type conversion: {e}")
            return indicators  # Return the empty DataFrame

    # If date_index is provided, slice the data up to that index
        if date_index is not None:
            print(f"Calculating indicators for data up to index {date_index}, date: {data.index[date_index]}")
            if date_index >= len(data):
                print(f"Warning: date_index {date_index} is out of range for the data.")
                return indicators  # Return the empty DataFrame if out of range
            data = data.iloc[:date_index + 1]

        try:
        # MACD Calculation
            print("Calculating MACD...")
            short_ema = data['Close'].ewm(span=8).mean()
            long_ema = data['Close'].ewm(span=17).mean()
            macd = short_ema - long_ema
            macd_signal = macd.ewm(span=9).mean()
            print("MACD calculated successfully.")

        # RSI Calculation
            print("Calculating RSI...")
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=8, min_periods=1).mean()
            avg_loss = loss.rolling(window=8, min_periods=1).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            print("RSI calculated successfully.")

        # Bollinger Bands Calculation
            print("Calculating Bollinger Bands...")
            sma = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            upper_band = sma + (2 * rolling_std)
            lower_band = sma - (2 * rolling_std)
            print("Bollinger Bands calculated successfully.")

        # Stochastic Oscillator Calculation
            print("Calculating Stochastic Oscillator...")
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            print("Stochastic Oscillator calculated successfully.")

        # OBV Calculation
            print("Calculating OBV...")
            obv = data['Volume'].copy()
            obv[data['Close'] < data['Close'].shift(1)] *= -1
            bv = obv.cumsum()
            print("OBV calculated successfully.")

        # ATR Calculation
            print("Calculating ATR...")
            tr1 = abs(data['High'] - data['Low'])
            tr2 = abs(data['High'] - data['Close'].shift(1)).fillna(0)
            tr3 = abs(data['Low'] - data['Close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().fillna(tr)                
            print("ATR calculated successfully.")

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
            }, index=data.index)

        # Fill NaN values
            indicators.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace any inf values with NaN
            indicators.ffill(inplace=True)  # Forward fill to propagate last valid value forward
            indicators.dropna(inplace=True)  # Drop any remaining NaN values
            indicators.reset_index(drop=True, inplace=True)

            print(f"Length of indicators DataFrame: {len(indicators)}")
            return indicators

        except Exception as e:
            print(f"Error in indicator calculation: {e}")
            return indicators 

    def prepare_data(self, date_index=None):
        features, labels = [], []

        for ticker, data in self.historical_data.items():
            indicators = self.calculate_indicators(data, date_index)

            if indicators.empty:
                continue  # Skip processing if indicators are empty

            for i in range(self.lookback_period, len(data) - self.lookback_period):
                feature_vector = indicators.iloc[i - self.lookback_period:i].values.flatten()

            # Debug: Check if all feature vectors have the same length
                if features and len(feature_vector) != len(features[0]):
                    print(f"Inconsistent feature vector length for {ticker} at index {i}")
                    continue  # Skip this feature vector

                features.append(feature_vector)
                label = 1 if data['Close'].iloc[i] > data['Close'].iloc[i - 1] else 0
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

    def predict(self, new_data, date_index):
        # Ensure new_data is sliced correctly
        if date_index >= len(new_data):
            raise IndexError("date_index is out of range for new_data.")
        sliced_data = new_data.iloc[:date_index + 1]

        indicators = self.calculate_indicators(sliced_data)
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
        for ticker in tickers:
            print(f"Length of data for {ticker}: {len(self.historical_data[ticker])}")

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
        print(f"Analyzing {ticker} for date index: {date_index}, corresponding to date: {self.historical_data[ticker].index[date_index]}")
        data = self.historical_data[ticker]

        # Compute indicators
        indicators = self.optimizer.calculate_indicators(self.historical_data[ticker], date_index)

        if date_index < 0 or date_index >= len(indicators):
            print(f"Warning: date_index {date_index} is out of range for {ticker}.")
            return Decision.HOLD

        # Create a dictionary of the current values of indicators
        indicators_dict = {
            'MACD': indicators['MACD'].iloc[date_index],
            'MACD_Signal': indicators['MACD_Signal'].iloc[date_index],
            'Upper_Band': indicators['Upper_Band'].iloc[date_index],
            'Lower_Band': indicators['Lower_Band'].iloc[date_index],
            'RSI': indicators['RSI'].iloc[date_index],
            'Stochastic_Oscillator': indicators['Stochastic_Oscillator'].iloc[date_index],
            'OBV': indicators['OBV'].iloc[date_index],
            'ATR': indicators['ATR'].iloc[date_index]
        }

        if 'MACD' in indicators.columns:
            indicators_dict['MACD'] = indicators['MACD'].iloc[date_index]
        else:
            print(f"Warning: 'MACD' column not found in indicators for {ticker}.")
            return Decision.HOLD

        if 'MACD_Signal' in indicators.columns:
            indicators_dict['MACD_Signal'] = indicators['MACD_Signal'].iloc[date_index]
        else:
            print(f"Warning: 'MACD_Signal' column not found in indicators for {ticker}.")
            return Decision.HOLD

        if 'Upper_Band' in indicators.columns:
            indicators_dict['Upper_Band'] = indicators['Upper_Band'].iloc[date_index]
        else:
            print(f"Warning: 'Upper_Band' column not found in indicators for {ticker}.")
            return Decision.HOLD

        if 'Lower_Band' in indicators.columns:
            indicators_dict['Lower_Band'] = indicators['Lower_Band'].iloc[date_index]
        else:
            print(f"Warning: 'Lower_Band' column not found in indicators for {ticker}.")
            return Decision.HOLD

        if 'RSI' in indicators.columns:
            indicators_dict['RSI'] = indicators['RSI'].iloc[date_index]
        else:
            print(f"Warning: 'RSI' column not found in indicators for {ticker}.")
            return Decision.HOLD

        if 'Stochastic_Oscillator' in indicators.columns:
            indicators_dict['Stochastic_Oscillator'] = indicators['Stochastic_Oscillator'].iloc[date_index]
        else:
            print(f"Warning: 'Stochastic_Oscillator' column not found in indicators for {ticker}.")
            return Decision.HOLD

        if 'OBV' in indicators.columns:
            indicators_dict['OBV'] = indicators['OBV'].iloc[date_index]
        else:
            print(f"Warning: 'OBV' column not found in indicators for {ticker}.")
            return Decision.HOLD

        if 'ATR' in indicators.columns:
            indicators_dict['ATR'] = indicators['ATR'].iloc[date_index]
        else:
            print(f"Warning: 'ATR' column not found in indicators for {ticker}.")
        return Decision.HOLD

        # Calculate the weighted score
        weighted_score = sum(
            self.indicator_weights[i] * value for i, value in enumerate(indicators_dict.values())
        )
        # Make the buy or sell decision based on the score
        # Note: The thresholds here are arbitrary and should be adjusted based on backtesting results
        buy_threshold = 0.5  # Example threshold for buying
        sell_threshold = -0.0  # Example threshold for selling

        if weighted_score > buy_threshold:
            decision = Decision.BUY
        elif weighted_score < sell_threshold:
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

    def _place_order(self, ticker, action, quantity, price, date_index):
        print(f"Placing order: {action} {quantity} of {ticker} at {price} on index {date_index}")
        if self.trade_count >= self.trade_limit or not self._can_trade(ticker, date_index):
            print(f"Order for {ticker} not placed. Trade count or cooldown period restriction.")
            return

        if action == "buy":
            # Buying logic
            self._execute_buy(ticker, quantity, price, date_index)
        elif action == "sell":
            # Selling logic
            self._execute_sell(ticker, quantity, price, date_index)

    def _execute_buy(self, ticker, quantity, price, date_index):
        order_quantity = min(quantity, self._calculate_position_size(price))
        if order_quantity > 0:
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + order_quantity
            self.portfolio['cash'] -= price * order_quantity
            print(f"Executed buy: {order_quantity} of {ticker} at {price} on index {date_index}")
            # Setting up stop loss and trailing stop for the purchased stock
            self._setup_stop_loss(ticker, price, order_quantity)
        else:
            print(f"Buy order not executed for {ticker}. Insufficient quantity or cash.")

    def _execute_sell(self, ticker, quantity, price, date_index):
        current_quantity = self.portfolio.get(ticker, 0)
        sell_quantity = min(quantity, current_quantity)
        if sell_quantity > 0:
            self.portfolio[ticker] -= sell_quantity
            self.portfolio['cash'] += price * sell_quantity
            print(f"Executed sell: {sell_quantity} of {ticker} at {price} on index {date_index}")
            # Removing the stop loss and trailing stop after selling
            self._remove_stop_loss(ticker)
        else:
            print(f"Sell order not executed for {ticker}. Insufficient quantity.")

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
        self._init_optimizer()

        min_length = min(len(self.historical_data[ticker]) for ticker in self.historical_data)

    # Debug: Print the total number of days in the iteration
        total_days = len(self.historical_data[next(iter(self.historical_data))]) - self.lookback_period
        print(f"Total days to process: {total_days}")

    # Iterate over each trading day
        for date_index in range(self.lookback_period, len(self.historical_data[next(iter(self.historical_data))])):
            date = self.historical_data[next(iter(self.historical_data))].index[date_index]
            self.reset_trade_count(date)

        # Debug: Print the current date being processed
            print(f"Processing date: {date}")

        # Check portfolio for max loss
            self._check_max_loss(self.initial_balance, date_index)

        # Iterate over each ticker and make trading decisions
            for ticker in self.historical_data:
                self._check_stop_loss(ticker, date_index)
                decision = self._analyze_stock(ticker, date_index)
                current_price = self.historical_data[ticker]['Close'].iloc[date_index]

            # Execute trades based on decisions
                if decision == Decision.BUY:
                    quantity = self._calculate_position_size(current_price)
                    self._place_order(ticker, "buy", quantity, current_price, date_index)
                elif decision == Decision.SELL:
                    quantity = self.portfolio.get(ticker, 0)
                    self._place_order(ticker, "sell", quantity, current_price, date_index)

            # Check for trailing stop loss
            self._check_trailing_stop_loss(ticker, date_index)

        # Calculate and display end-of-day portfolio value
        self.calculate_portfolio_value(date_index)

        # Debug: Print progress
        print(f"Completed processing day {date_index - self.lookback_period + 1} of {total_days}")

# Rest of your code for setting up and running the bot

        
    def calculate_portfolio_value(self, date_index):
            last_day_index = len(self.historical_data[next(iter(self.historical_data))]) - 1
            closing_prices = {ticker: self.historical_data[ticker]['Close'].iloc[last_day_index] for ticker in self.historical_data}

            total_shares_value = sum(self.portfolio.get(ticker, 0) * closing_prices[ticker] for ticker in self.historical_data)
            total_portfolio_value = self.portfolio['cash'] + total_shares_value

            print(f"Closing Prices: {closing_prices}")
            print(f"Total Portfolio Value: {total_portfolio_value}")
            print(self.portfolio)
        

# Tickers and date range
tickers = ["TQQQ", "SQQQ"]
start_date = "2020-01-01"
end_date = "2021-01-01"

bot = BacktestTradingBot(tickers, start_date, end_date)
bot.logger.output_log(print_log=True, save_to_file='trading_bot_log.txt')
bot.run()