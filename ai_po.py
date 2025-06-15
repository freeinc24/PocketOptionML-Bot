import time
import json
import asyncio
import csv
import os
import random
from datetime import datetime, timedelta
from collections import deque
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from BinaryOptionsToolsV2.tracing import start_logs, Logger
import pandas as pd
import talib.abstract as ta
from sklearn.linear_model import LogisticRegression
import joblib
from config import SSID, DEMO, MIN_PAYOUT, PERIOD, EXPIRATION

# === GLOBAL CONFIG ===
start_logs(".", "WARNING", terminal=True)  # Initialize logging
logger = Logger()
ssid = SSID
demo = DEMO
min_payout = MIN_PAYOUT
period = PERIOD
expiration = EXPIRATION
TRADE_LOG = "trades_log.csv"
SMA_LOG = "sma_values.csv"
MODEL_FILE = "ml_model.pkl"

# === GLOBAL STATE ===
pairs = {}  # Replace global_value.pairs
websocket_is_connected = True  # Track connection status

# === INIT API ===
async def init_api():
    global api
    api = PocketOptionAsync(ssid)
    await asyncio.sleep(5)  # Wait for WebSocket connection
    return api

# === Load or Init Model ===
try:
    clf = joblib.load(MODEL_FILE)
    logger.info("Loaded saved ML model.")
except:
    clf = LogisticRegression()
    logger.warning("New model will be trained.")

# === LOG TRADE ===
def log_trade(pair, direction, amount, expiration, last, result=None):
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "pair": pair,
        "direction": direction,
        "amount": amount,
        "expiration": expiration,
        "signal": "SMA+MACD+RSI+ML",
        "price": round(last['close'], 5),
        "sma12": round(last['sma12'], 5),
        "sma26": round(last['sma26'], 5),
        "macd": round(last['macd'], 5),
        "rsi": round(last['rsi'], 2),
        "result": result if result else "pending"
    }
    file_exists = os.path.isfile(TRADE_LOG)
    with open(TRADE_LOG, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    logger.info(f"Trade logged: {pair}, {direction}, Result: {result}")

# === LOG INDICATORS ===
def log_sma(pair, last):
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "pair": pair,
        "sma12": round(last['sma12'], 5),
        "sma26": round(last['sma26'], 5),
        "macd": round(last['macd'], 5),
        "rsi": round(last['rsi'], 2),
        "price": round(last['close'], 5)
    }
    file_exists = os.path.isfile(SMA_LOG)
    with open(SMA_LOG, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    logger.debug(f"SMA logged: {pair}, SMA12: {last['sma12']}")

# === PAYOUT FILTER ===
async def get_payout():
    global pairs
    try:
        payout_data = await api.payout()  # Returns dict of asset:payout
        pairs = {}
        for asset, payout in payout_data.items():
            if payout >= min_payout and "_otc" in asset:
                pairs[asset] = {
                    'id': asset,  # Adjust if API provides specific ID
                    'payout': payout,
                    'type': 'binary',  # Assume binary for OTC
                    'history': deque(maxlen=100)
                }
        logger.info(f"Payout data loaded: {len(pairs)} pairs with payout >= {min_payout}%")
        return True
    except Exception as e:
        logger.error(f"Failed parsing payout: {e}")
        return False

# === HISTORY LOADER ===
async def get_df():
    try:
        for pair in pairs:
            candles = await api.get_candles(pair, period, 3600)  # Fetch 1 hour of candles
            if candles and isinstance(pairs[pair].get('history'), deque):
                for candle in candles:
                    pairs[pair]['history'].append({
                        'time': candle['time'],
                        'price': candle['close']  # Adjust if API uses different keys
                    })
            await asyncio.sleep(1)  # Avoid rate limits
        logger.info("Historical data loaded for all pairs")
        return True
    except Exception as e:
        logger.error(f"get_df() error: {e}")
        return False

# === ORDER EXECUTION ===
async def buy(amount, pair, action, expiration, last):
    try:
        if action == "call":
            (trade_id, trade_data) = await api.buy(asset=pair, amount=amount, time=expiration, check_win=False)
        else:
            (trade_id, trade_data) = await api.sell(asset=pair, amount=amount, time=expiration, check_win=False)
        await asyncio.sleep(expiration + 2)  # Wait for trade to complete
        outcome = await api.check_win(trade_id)
        result = outcome.get('result', 'unknown')
        log_trade(pair, action, amount, expiration, last, result)
        logger.info(f"TRADE: {pair} | {action.upper()} | Result: {result.upper()}")
    except Exception as e:
        logger.error(f"Trade failed: {pair}, {action}, Error: {e}")

# === RESAMPLE CANDLES ===
def make_df(history):
    if not history or not isinstance(history, (list, deque)):
        return pd.DataFrame()
    try:
        df = pd.DataFrame(history).reset_index(drop=True)
        df = df.sort_values(by='time').reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df['price'].resample(f'{period}s').ohlc()
        df.reset_index(inplace=True)
        df = df[df['time'] < datetime.fromtimestamp(wait(False))]
        return df
    except Exception as e:
        logger.error(f"make_df error: {e}")
        return pd.DataFrame()

# === ROTATING STRATEGY ===
async def rotate_pairs_strategy():
    global websocket_is_connected
    while True:
        try:
            all_pairs = list(pairs.keys())
            if len(all_pairs) < 5:
                logger.warning("Not enough 92% payout pairs.")
                if not await reconnect():
                    return
                continue

            cycle_time = 5 * 60  # 5 min
            random.shuffle(all_pairs)
            watchlist = all_pairs[:5]
            logger.info(f"Watching: {', '.join(watchlist)}")

            start_time = time.time()
            trade_count = 0

            while time.time() - start_time < cycle_time:
                if not websocket_is_connected:
                    raise Exception("Websocket connection lost")

                for pair in watchlist:
                    history = pairs[pair].get('history')
                    if not history or len(history) < 60:
                        continue

                    df = make_df(history)
                    if df.empty or len(df) < 40:
                        continue

                    df['sma12'] = ta.SMA(df['close'], timeperiod=12)
                    df['sma26'] = ta.SMA(df['close'], timeperiod=26)
                    df['macd'], df['macdsignal'], _ = ta.MACD(df['close'], 12, 26, 9)
                    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
                    df['sma_diff'] = df['sma12'] - df['sma26']
                    df['macd_hist'] = df['macd'] - df['macdsignal']
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                    df.dropna(inplace=True)

                    features = ['sma_diff', 'macd_hist', 'rsi']
                    X = df[features]
                    y = df['target']
                    if len(X) < 30:
                        continue

                    clf.fit(X[:-1], y[:-1])
                    joblib.dump(clf, MODEL_FILE)

                    last = df.iloc[-1]
                    X_pred = pd.DataFrame([{
                        'sma_diff': last['sma_diff'],
                        'macd_hist': last['macd_hist'],
                        'rsi': last['rsi']
                    }])
                    ml_prediction = clf.predict(X_pred)[0]
                    last_dict = last.to_dict()
                    log_sma(pair, last_dict)

                    signal_type = None
                    if last['sma12'] > last['sma26'] and last['macd'] > last['macdsignal'] and 40 < last['rsi'] < 60:
                        signal_type = "call"
                    elif last['sma12'] < last['sma26'] and last['macd'] < last['macdsignal'] and 40 < last['rsi'] < 60:
                        signal_type = "put"

                    if signal_type == "call" and ml_prediction == 1:
                        logger.info(f"CALL on {pair}")
                        await buy(100, pair, "call", expiration, last_dict)
                        trade_count += 1
                    elif signal_type == "put" and ml_prediction == 0:
                        logger.info(f"PUT on {pair}")
                        await buy(100, pair, "put", expiration, last_dict)
                        trade_count += 1

                await wait(sleep=True)

            logger.info(f"Cycle done — trades placed: {trade_count}")

        except Exception as e:
            logger.error(f"Strategy error: {e}")
            if not await reconnect():
                return
            continue

# === PREP + INIT ===
async def prepare():
    ok = await get_payout()
    if not ok:
        return False
    return await get_df()

async def wait(sleep=True):
    dt = int(datetime.now().timestamp()) - datetime.now().second
    dt += 60
    if sleep:
        await asyncio.sleep(dt - int(datetime.now().timestamp()))
    return dt

async def reconnect():
    """Attempt to reconnect to PocketOption API"""
    global api, websocket_is_connected
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.info(f"Reconnecting (attempt {retry_count + 1}/{max_retries})...")
            api = await init_api()
            balance = await api.balance()
            websocket_is_connected = True
            logger.info(f"Reconnected successfully. Balance: {balance}")
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            websocket_is_connected = False
            retry_count += 1
            await asyncio.sleep(5)
    return False

async def start():
    global api
    api = await init_api()
    while not websocket_is_connected:
        await asyncio.sleep(0.1)
    await asyncio.sleep(2)

    while True:
        try:
            balance = await api.balance()
            logger.info(f"Balance: {balance}")
            if await prepare():
                await rotate_pairs_strategy()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            if not await reconnect():
                logger.error("Max reconnection attempts reached. Exiting...")
                break
            continue

if __name__ == "__main__":
    asyncio.run(start())