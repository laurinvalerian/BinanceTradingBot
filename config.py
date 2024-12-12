# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class BinanceConfig:
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_API_SECRET')
    IS_TESTNET = True
    SYMBOL = 'BTCUSDT'
    RISK_PER_TRADE = 0.02  # 2% risk per trade