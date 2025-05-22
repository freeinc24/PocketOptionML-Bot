import time, json, threading, csv, os, random
from datetime import datetime
from collections import deque
from pocketoptionapi.stable_api import PocketOption
import pocketoptionapi.global_value as global_value
import pandas as pd
import talib.abstract as ta
from sklearn.linear_model import LogisticRegression
import joblib
from config import SSID, DEMO, MIN_PAYOUT, PERIOD, EXPIRATION

# === GLOBAL CONFIG ===
global_value.loglevel = 'WARNING'  # Quiet mode
ssid = SSID
demo = DEMO
min_payout = MIN_PAYOUT
period = PERIOD
expiration = EXPIRATION
TRADE_LOG = "trades_log.csv"
SMA_LOG = "sma_values.csv"
MODEL_FILE = "ml_model.pkl"

# === INIT API ===
api = PocketOption(ssid, demo)
api.connect()

# === Load or Init Model ===
try:
    clf = joblib.load(MODEL_FILE)
    print("í´– Loaded saved ML model.")
except:
    clf = LogisticRegression()
    print("í´– New model will be trained.")

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

# === PAYOUT FILTER ===
def get_payout():
    try:
        d = json.loads(global_value.PayoutData)
        global_value.pairs = {}
        for pair in d:
            if len(pair) == 19 and pair[14] and pair[5] == min_payout and "_otc" in pair[1]:
                global_value.pairs[pair[1]] = {
                    'id': pair[0],
                    'payout': pair[5],
                    'type': pair[3]
                }
        return True
    except Exception as e:
        global_value.logger(f"Failed parsing payout: {e}", "ERROR")
        return False

# === HISTORY LOADER ===
def get_df():
    try:
        for i, pair in enumerate(global_value.pairs):
            df = api.get_candles(pair, period)
            if df and isinstance(global_value.pairs[pair].get('history'), deque):
                for entry in df:
                    global_value.pairs[pair]['history'].append(entry)
            time.sleep(1)
        return True
    except Exception as e:
        global_value.logger(f"get_df() error: {e}", "ERROR")
        return False

# === ORDER EXECUTION ===
def buy(amount, pair, action, expiration, last):
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    trade_id = result[1]
    outcome = api.check_win(trade_id)
    log_trade(pair, action, amount, expiration, last, outcome)
    print(f"âœ… TRADE: {pair} | {action.upper()} | Result: {outcome.upper()}")

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
        global_value.logger(f"make_df error: {e}", "ERROR")
        return pd.DataFrame()

# === ROTATING STRATEGY ===
def rotate_pairs_strategy():
    while True:
        try:
            all_pairs = list(global_value.pairs.keys())
            if len(all_pairs) < 5:
                print("âš ï¸ Not enough 92% payout pairs.")
                if not reconnect():
                    return
                continue

            cycle_time = 5 * 60  # 5 min
            random.shuffle(all_pairs)
            watchlist = all_pairs[:5]
            print(f"\ní´ Watching: {', '.join(watchlist)}")

            start_time = time.time()
            trade_count = 0

            while time.time() - start_time < cycle_time:
                if not global_value.websocket_is_connected:
                    raise Exception("Websocket connection lost")
                    
                for pair in watchlist:
                    history = global_value.pairs[pair].get('history')
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
                    # Create DataFrame for prediction with named features
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
                        print(f"íº€ CALL on {pair}")
                        threading.Thread(target=buy, args=(100, pair, "call", expiration, last_dict)).start()
                        trade_count += 1
                    elif signal_type == "put" and ml_prediction == 0:
                        print(f"íº€ PUT on {pair}")
                        threading.Thread(target=buy, args=(100, pair, "put", expiration, last_dict)).start()
                        trade_count += 1

                wait(sleep=True)

            print(f"â±ï¸ Cycle done â€” trades placed: {trade_count}")
                
        except Exception as e:
            print(f"âŒ Strategy error: {e}")
            if not reconnect():
                return
            continue

# === PREP + INIT ===
def prepare():
    ok = get_payout()
    if not ok:
        return False
    for pair in global_value.pairs:
        global_value.pairs[pair]['history'] = deque(maxlen=100)
    return get_df()

def wait(sleep=True):
    dt = int(datetime.now().timestamp()) - datetime.now().second
    dt += 60
    if sleep:
        time.sleep(dt - int(datetime.now().timestamp()))
    return dt

def reconnect():
    """Attempt to reconnect to PocketOption API"""
    global api
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"í´„ Reconnecting (attempt {retry_count + 1}/{max_retries})...")
            api = PocketOption(ssid, demo)
            api.connect()
            time.sleep(2)
            if global_value.websocket_is_connected:
                print("âœ… Reconnected successfully")
                return True
        except Exception as e:
            print(f"âŒ Reconnection failed: {e}")
            retry_count += 1
            time.sleep(5)
    return False

def start():
    while not global_value.websocket_is_connected:
        time.sleep(0.1)
    time.sleep(2)
    
    while True:
        try:
            print(f"í²° Balance: {api.get_balance()}")
            if prepare():
                rotate_pairs_strategy()
        except Exception as e:
            print(f"âŒ Error in main loop: {e}")
            if not reconnect():
                print("í»‘ Max reconnection attempts reached. Exiting...")
                break
            continue

if __name__ == "__main__":
    start()
