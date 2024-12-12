from bot import BinanceBot
from config import BinanceConfig
import time
import itertools
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def backtest_period(df, initial_balance=1000, bot=None):
    if bot is None:
        config = BinanceConfig()
        bot = BinanceBot(config)
        
    balance = initial_balance
    position = {'amount': 0, 'entry_price': 0}
    trades = []
    high_since_entry = 0
    trade_start_balance = initial_balance
    peak_balance = initial_balance
    trade_durations = []
    profits = []
    
    df = df.copy()
    
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        current_price = current_row['close']
        
        balance_with_position = balance
        if position['amount'] > 0:
            balance_with_position = balance + position['amount'] * current_price
        peak_balance = max(peak_balance, balance_with_position)
        
        if position['amount'] > 0:
            high_since_entry = max(high_since_entry, current_price)
            
            if bot.should_exit_trade(current_price, position['entry_price'], high_since_entry, current_row):
                trade_end = current_row['timestamp']
                trade_duration = (trade_end - trade_start).total_seconds() / 60
                trade_durations.append(trade_duration)
                
                gross_exit = position['amount'] * current_price
                fees = gross_exit * bot.fee_rate
                balance = balance + gross_exit - fees
                profit = balance - trade_start_balance
                profits.append(profit)
                
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'time': current_row['timestamp'],
                    'profit': profit,
                    'duration': trade_duration,
                    'fees': fees,
                    'exit_reason': 'trailing_stop' if current_price < high_since_entry * (1 - bot.trailing_stop_pct) else 'technical'
                })
                
                position = {'amount': 0, 'entry_price': 0}
        
        else:
            strong_bullish = (
                current_row['trend'] in ['strong_up', 'weak_up'] and
                current_row['zone_type'] in ['strong_bullish', 'weak_bullish'] and
                current_row['volume'] > current_row['volume_ma'] and
                current_row['rsi'] > 40 and current_row['rsi'] < 70 and
                current_row['macd'] > current_row['macd_signal']
            )
            
            if strong_bullish:
                trade_start_balance = balance
                trade_start = current_row['timestamp']
                
                position_size = bot.calculate_position_size(
                    balance,
                    current_row['zone_strength'],
                    bot.get_trend_strength(current_row)
                )
                
                gross_entry = min(position_size, balance * 0.95)
                fees = gross_entry * bot.fee_rate
                position['amount'] = (gross_entry - fees) / current_price
                position['entry_price'] = current_price
                balance -= (gross_entry + fees)
                high_since_entry = current_price
                
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'time': current_row['timestamp'],
                    'amount': position['amount'],
                    'position_size': gross_entry,
                    'zone_strength': current_row['zone_strength'],
                    'fees': fees
                })
    
    if position['amount'] > 0:
        gross_exit = position['amount'] * df.iloc[-1]['close']
        fees = gross_exit * bot.fee_rate
        balance += gross_exit - fees
    
    total_trades = len([t for t in trades if t['type'] == 'SELL'])
    profit_trades = len([t for t in trades if t['type'] == 'SELL' and t['profit'] > 0])
    win_rate = (profit_trades / total_trades * 100) if total_trades > 0 else 0
    total_profit = balance - initial_balance
    roi = (total_profit / initial_balance * 100) if initial_balance > 0 else 0
    max_drawdown = ((peak_balance - balance_with_position) / peak_balance * 100) if peak_balance > 0 else 0
    profit_factor = abs(sum(p for p in profits if p > 0) / sum(p for p in profits if p < 0)) if any(p < 0 for p in profits) else 0
    sharpe = np.sqrt(365) * np.mean(profits) / np.std(profits) if profits and len(profits) > 1 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'roi': roi,
        'total_profit': total_profit,
        'final_balance': balance,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'trades': trades,
        'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
        'avg_profit_per_trade': np.mean(profits) if profits else 0,
        'largest_win': max(profits) if profits else 0,
        'largest_loss': min(profits) if profits else 0,
        'avg_win': np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0,
        'avg_loss': np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
    }

class GridSearch:
    def __init__(self):
        self.param_grid = {
            'trailing_stop_pct': np.linspace(0.015, 0.035, 5),
            'max_position_size': np.linspace(0.05, 0.12, 4),
            'min_volume_multiplier': np.linspace(1.3, 2.2, 4),
            'profit_take_pct': np.linspace(0.02, 0.055, 4),
            'zones': [10, 11, 12],
            'macd_fast': [12, 13],
            'macd_slow': [25, 26, 27, 28],
            'macd_signal': [9, 10]
        }
        
    def generate_params(self, n_samples=100):
        params_list = []
        for _ in range(n_samples):
            params = {
                'trailing_stop_pct': np.random.choice(self.param_grid['trailing_stop_pct']),
                'max_position_size': np.random.choice(self.param_grid['max_position_size']),
                'min_volume_multiplier': np.random.choice(self.param_grid['min_volume_multiplier']),
                'profit_take_pct': np.random.choice(self.param_grid['profit_take_pct']),
                'zones': np.random.choice(self.param_grid['zones']),
                'macd_fast': np.random.choice(self.param_grid['macd_fast']),
                'macd_slow': np.random.choice(self.param_grid['macd_slow']),
                'macd_signal': np.random.choice(self.param_grid['macd_signal'])
            }
            params_list.append(params)
        return params_list

def run_grid_search(pairs=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], 
                   timeframes=['15m', '1h', '4h'], 
                   n_param_combinations=100,
                   iterations_per_combo=3):
    
    grid_search = GridSearch()
    param_combinations = grid_search.generate_params(n_param_combinations)
    
    config = BinanceConfig()
    all_results = []
    
    total_iterations = len(param_combinations) * len(pairs) * len(timeframes) * iterations_per_combo
    
    with tqdm(total=total_iterations, desc="Grid Search Progress") as pbar:
        for params in param_combinations:
            bot = BinanceBot(config)
            for param, value in params.items():
                setattr(bot, param, value)
            
            for pair in pairs:
                for timeframe in timeframes:
                    for _ in range(iterations_per_combo):
                        try:
                            df = bot.get_historical_data(symbol=pair, interval=timeframe)
                            if df is None or df.empty:
                                continue
                                
                            df = bot.calculate_zones(df, num_zones=params['zones'])
                            df = bot.identify_trend(df)
                            
                            if df is None or df.empty:
                                continue
                            
                            result = backtest_period(df, bot=bot)
                            result.update({
                                'pair': pair,
                                'timeframe': timeframe,
                                'parameters': params
                            })
                            
                            all_results.append(result)
                            
                            time.sleep(0.5)
                            
                        except Exception as e:
                            print(f"Error: {str(e)}")
                            time.sleep(2)
                        
                        pbar.update(1)
    
    # Analyze results
    df_results = pd.DataFrame(all_results)
    
    # Group by parameters and calculate mean metrics
    param_cols = list(param_combinations[0].keys())
    metrics = ['roi', 'win_rate', 'sharpe_ratio', 'profit_factor']
    
    best_params = (df_results.groupby(['pair', 'timeframe'] + 
                  [f'parameters.{p}' for p in param_cols])[metrics]
                  .mean()
                  .sort_values('roi', ascending=False)
                  .head(20))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_file = f'grid_search_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': df_results.to_dict('records'),
            'best_parameters': best_params.to_dict('index'),
            'test_info': {
                'pairs': pairs,
                'timeframes': timeframes,
                'n_param_combinations': n_param_combinations,
                'iterations_per_combo': iterations_per_combo
            }
        }, f, indent=2)
    
    print("\nTop 20 Parameter Combinations:")
    print(best_params)
    print(f"\nFull results saved to {results_file}")

if __name__ == "__main__":
    run_grid_search(
        pairs=[
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 
            'LINKUSDT', 'MATICUSDT', 'SOLUSDT', 'AVAXUSDT', 'NEARUSDT'
        ],
        timeframes=[
            '1m', '3m', '5m', '15m', '30m', 
            '1h', '2h', '4h', '6h', '8h', 
            '12h', '1d'
        ],
        n_param_combinations=100,
        iterations_per_combo=3
    )