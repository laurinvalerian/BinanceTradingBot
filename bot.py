from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from config import BinanceConfig

class BinanceBot:
    def __init__(self, config):
        self.config = config
        self.client = Client(config.API_KEY, config.API_SECRET)
        # Increased trailing stop to protect profits better
        self.trailing_stop_pct = np.random.uniform(0.025, 0.035)  # Was 0.015-0.025
        # Reduced position size for better risk management
        self.max_position_size = np.random.uniform(0.05, 0.08)    # Was 0.08-0.12
        # Increased volume requirements
        self.min_volume_multiplier = np.random.uniform(1.8, 2.2)  # Was 1.3-1.7
        self.fee_rate = 0.001
        # Adjusted technical indicators for stronger trends
        self.rsi_period = np.random.randint(14, 18)              # Was 12-16
        self.volume_ma_period = np.random.randint(20, 24)        # Was 18-22
        # Increased profit target
        self.profit_take_pct = np.random.uniform(0.035, 0.055)   # Was 0.02-0.04
        self.max_trades_per_day = np.random.randint(2, 3)        # Was 3-5
        self.zones = np.random.choice([11, 12])                  # Was 10,11,12
        # Adjusted MACD parameters for stronger trend confirmation
        self.macd_fast = np.random.choice([12, 13])
        self.macd_slow = np.random.choice([26, 27, 28])         # Was 25,26,27
        self.macd_signal = np.random.choice([9, 10])

    def get_historical_data(self, symbol, interval='1m', limit=1000):
        try:
            # Calculate start time to be 3 years ago
            start_time = datetime.now() - timedelta(days=1095)
            end_time = datetime.now()
            
            all_klines = []
            
            while start_time < end_time:
                chunk_end = min(start_time + timedelta(days=100), end_time)
                
                try:
                    klines = self.client.get_historical_klines(
                        symbol, 
                        interval,
                        str(int(start_time.timestamp() * 1000)),
                        str(int(chunk_end.timestamp() * 1000)),
                        limit=1000
                    )
                    
                    if klines:
                        all_klines.extend(klines)
                    
                    start_time = chunk_end
                    time.sleep(0.5)
                    
                except Exception as chunk_error:
                    print(f"Error fetching chunk: {chunk_error}")
                    time.sleep(2)
                    continue
            
            if not all_klines:
                return None
                
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_zones(self, df, num_zones=10):
        if df is None or len(df) < 2:
            return None
            
        df = df.copy()
        
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_std'] = df['volume'].rolling(window=self.volume_ma_period).std()
        df['volume_z_score'] = (df['volume'] - df['volume_ma']) / df['volume_std']
        
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        try:
            # Enhanced volume profile calculation
            df['price_volatility'] = df['high'] - df['low']
            df['volume_profile'] = df['volume'] * df['volume_z_score'].abs() * df['price_volatility']
            
            bins = pd.qcut(df['volume_profile'], q=num_zones, duplicates='drop')
            df['zone'] = bins.cat.codes
            
            zone_volumes = df.groupby('zone')['volume'].sum()
            zone_price_range = df.groupby('zone')['price_volatility'].mean()
            total_volume = zone_volumes.sum()
            
            # Enhanced zone strength calculation
            zone_strengths = (zone_volumes / total_volume) * (1 + zone_price_range) * (1 + df.groupby('zone')['volume_z_score'].mean().abs())
            df['zone_strength'] = df['zone'].map(zone_strengths)
            
            # More stringent zone type classification
            df['zone_type'] = np.where(
                (df['close'] > df['open']) & 
                (df['upper_wick'] < df['body_size'] * 0.3) &  # Stricter wick requirements
                (df['volume'] > df['volume_ma'] * self.min_volume_multiplier),
                'strong_bullish',
                np.where(
                    (df['close'] < df['open']) & 
                    (df['lower_wick'] < df['body_size'] * 0.3) &
                    (df['volume'] > df['volume_ma'] * self.min_volume_multiplier),
                    'strong_bearish',
                    np.where(df['close'] > df['open'], 'weak_bullish', 'weak_bearish')
                )
            )
            
            return df
            
        except Exception as e:
            print(f"Error in zone calculation: {e}")
            return None

    def identify_trend(self, df, fast=None, slow=None, signal=None):
        df = df.copy()
        try:
            df['macd'] = df['close'].ewm(span=fast or self.macd_fast, adjust=False).mean() - \
                         df['close'].ewm(span=slow or self.macd_slow, adjust=False).mean()
            df['macd_signal'] = df['macd'].ewm(span=signal or self.macd_signal, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            df['macd_slope'] = df['macd'].diff()
            
            # Enhanced RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(span=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(span=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Additional trend indicators
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
            
            # Price momentum
            df['momentum'] = df['close'].pct_change(3)
            
            df['trend'] = 'neutral'
            
            # Stricter strong trend conditions
            strong_up = (
                (df['macd'] > df['macd_signal']) &
                (df['macd_slope'] > 0) &
                (df['rsi'] > 45) &  # Was 50
                (df['rsi'] < 65) &  # Was 70
                (df['close'] > df['ema_20']) &
                (df['ema_20'] > df['ema_50']) &
                (df['momentum'] > 0) &  # Added momentum requirement
                (df['volume'] > df['volume_ma'] * self.min_volume_multiplier)  # Added volume requirement
            )
            df.loc[strong_up, 'trend'] = 'strong_up'
            
            weak_up = (
                (df['macd'] > df['macd_signal']) &
                (df['rsi'] > 45) &  # Was 50
                (df['close'] > df['ema_20']) &
                (df['momentum'] > 0)  # Added momentum requirement
            )
            df.loc[weak_up & ~strong_up, 'trend'] = 'weak_up'
            
            strong_down = (
                (df['macd'] < df['macd_signal']) &
                (df['macd_slope'] < 0) &
                (df['rsi'] < 55) &  # Was 50
                (df['rsi'] > 35) &  # Was 30
                (df['close'] < df['ema_20']) &
                (df['ema_20'] < df['ema_50']) &
                (df['momentum'] < 0)  # Added momentum requirement
            )
            df.loc[strong_down, 'trend'] = 'strong_down'
            
            weak_down = (
                (df['macd'] < df['macd_signal']) &
                (df['rsi'] < 55) &  # Was 50
                (df['close'] < df['ema_20']) &
                (df['momentum'] < 0)  # Added momentum requirement
            )
            df.loc[weak_down & ~strong_down, 'trend'] = 'weak_down'
            
            return df.bfill()
            
        except Exception as e:
            print(f"Error in trend identification: {e}")
            return None

    def should_exit_trade(self, current_price, entry_price, high_since_entry, row):
        trailing_stop = high_since_entry * (1 - self.trailing_stop_pct)
        profit_target = entry_price * (1 + self.profit_take_pct)
        
        # Enhanced exit conditions
        exit_conditions = {
            'trailing_stop': current_price < trailing_stop,
            'profit_target': current_price >= profit_target,
            'trend_reversal': row['trend'] in ['strong_down', 'weak_down'],
            'volume_dropout': row['volume'] < row['volume_ma'] * 0.5,  # Stricter volume requirement
            'rsi_overbought': row['rsi'] > 65,  # Was 70
            'zone_resistance': row['zone_type'] in ['strong_bearish', 'weak_bearish'],
            'momentum_shift': row['momentum'] < -0.01  # Added momentum exit condition
        }
        
        # More conservative exit logic
        return (
            exit_conditions['trailing_stop'] or
            exit_conditions['profit_target'] or
            sum(exit_conditions.values()) >= 2  # Was 3
        )

    def calculate_position_size(self, balance, zone_strength, trend_strength):
        max_position = balance * self.max_position_size
        
        # Enhanced position sizing factors
        zone_factor = min(max(zone_strength * 1.2, 0.3), 1.0)  # Increased zone influence
        rsi_factor = min(abs(trend_strength['rsi'] - 50) / 40, 1.0)  # Was 50
        macd_factor = min(abs(trend_strength['macd_hist']) / 1.5, 1.0)  # Was 2.0
        volume_factor = min(trend_strength.get('volume_confidence', 1.0) * 1.2, 1.0)  # Increased volume influence
        
        # More conservative position sizing
        trend_factor = (
            rsi_factor * 0.35 +
            macd_factor * 0.35 +
            volume_factor * 0.2 +
            np.random.uniform(0.7, 1.0) * 0.1  # Reduced randomness
        )
        
        position = max_position * zone_factor * trend_factor
        return min(position, max_position * 0.8, balance * 0.90)  # More conservative limits

    def get_trend_strength(self, row):
        return {
            'rsi': row['rsi'],
            'macd_hist': row['macd_hist'],
            'macd_slope': row['macd_slope'],
            'ema_diff': (row['ema_20'] - row['ema_50']) / row['ema_50'] * 100,
            'volume_confidence': row['volume_z_score'] if row['volume'] > row['volume_ma'] * self.min_volume_multiplier else 0,
            'momentum': row['momentum']
        }