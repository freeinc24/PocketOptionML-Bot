# -*- coding: utf-8 -*-
import time
import json
import asyncio
import csv
import os
import random
import traceback
from datetime import datetime, timedelta
from collections import deque
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from BinaryOptionsToolsV2.tracing import start_logs, Logger
import pandas as pd
import numpy as np
import talib.abstract as ta
from config import SSID, DEMO, MIN_PAYOUT, TRADE_AMOUNT

# === CONFIG VALIDATION ===
def validate_config():
    assert 0 < MIN_PAYOUT <= 100, "MIN_PAYOUT must be between 0 and 100"
    # Add more as needed

validate_config()

# === GLOBAL CONFIG ===
start_logs(".", "WARNING", terminal=True)  # Initialize logging
logger = Logger()
ssid = SSID
demo = DEMO
min_payout = MIN_PAYOUT
period = 5  # 5 second candles
expiration = 5  # 5 second expiry
TRADE_LOG = "trades_log.csv"
FRACTAL_LOG = "fractal_values.csv"

CYCLE_TIME = 2 * 60  # 2 minutes cycle
MIN_HISTORY_LENGTH = 100  # Need more history for fractal analysis
MIN_DF_LENGTH = 50
FRACTAL_PERIOD = 5  # Williams Fractal period
CHAOS_BANDS_PERIOD = 10  # Reduced from 20 for faster response

# === GLOBAL STATE ===
pairs = {}  # Replace global_value.pairs
websocket_is_connected = True  # Track connection status

if not hasattr(logger, "warning"):
    logger.warning = logger.warn

class Logger:
    def warn(self, *args, **kwargs):
        print("[WARN]", *args)

    def warning(self, *args, **kwargs):
        return self.warn(*args, **kwargs)

# === INIT API ===
async def init_api():
    global api
    api = PocketOptionAsync(ssid)
    await asyncio.sleep(5)  # Wait for WebSocket connection
    return api

# === FRACTAL FUNCTIONS ===
def calculate_fractals(df, period=5):
    """Calculate Williams Fractals"""
    high = df['high'].values
    low = df['low'].values
    
    fractal_up = np.zeros(len(df))
    fractal_down = np.zeros(len(df))
    
    for i in range(period, len(df) - period):
        # Up Fractal (High surrounded by lower highs)
        if all(high[i] > high[i-j] for j in range(1, period+1)) and \
           all(high[i] > high[i+j] for j in range(1, period+1)):
            fractal_up[i] = high[i]
        
        # Down Fractal (Low surrounded by higher lows)
        if all(low[i] < low[i-j] for j in range(1, period+1)) and \
           all(low[i] < low[i+j] for j in range(1, period+1)):
            fractal_down[i] = low[i]
    
    return fractal_up, fractal_down

def calculate_chaos_bands(df, period=20):
    """Calculate Chaos Bands using Bollinger Band concept with fractal points"""
    close = df['close']
    
    # Calculate moving average
    sma = ta.SMA(close, timeperiod=period)
    
    # Calculate standard deviation
    std = close.rolling(window=period).std()
    
    # Chaos Bands
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    
    return upper_band, lower_band, sma

def detect_breakout(df, fractal_up, fractal_down, upper_band, lower_band):
    """Detect breakout signals based on fractal chaos bands"""
    signals = []
    
    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        signal = None
        
        # Simplified breakout detection - just use bands
        if pd.notna(upper_band.iloc[i]) and pd.notna(lower_band.iloc[i]):
            # Bullish breakout: Price breaks above upper band
            if current_price > upper_band.iloc[i]:
                signal = "call"
            # Bearish breakout: Price breaks below lower band  
            elif current_price < lower_band.iloc[i]:
                signal = "put"
        
        signals.append(signal)
    
    return signals

# === LOG TRADE ===
def log_trade(pair: str, direction: str, amount: float, expiration: int, last: dict, result: str = None) -> None:
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "pair": pair,
        "direction": direction,
        "amount": amount,
        "expiration": expiration,
        "signal": "Fractal_Chaos_Bands",
        "price": round(last['close'], 5),
        "upper_band": round(last.get('upper_band', 0), 5),
        "lower_band": round(last.get('lower_band', 0), 5),
        "sma": round(last.get('sma', 0), 5),
        "fractal_up": round(last.get('fractal_up', 0), 5),
        "fractal_down": round(last.get('fractal_down', 0), 5),
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
def log_fractal(pair: str, last: dict) -> None:
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "pair": pair,
        "price": round(last['close'], 5),
        "upper_band": round(last.get('upper_band', 0), 5),
        "lower_band": round(last.get('lower_band', 0), 5),
        "sma": round(last.get('sma', 0), 5),
        "fractal_up": round(last.get('fractal_up', 0), 5),
        "fractal_down": round(last.get('fractal_down', 0), 5)
    }
    file_exists = os.path.isfile(FRACTAL_LOG)
    with open(FRACTAL_LOG, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    logger.debug(f"Fractal logged: {pair}, Price: {last['close']}")

# === PAYOUT FILTER ===
async def get_payout():
    global pairs
    try:
        payout_data = await asyncio.wait_for(api.payout(), timeout=10)
        pairs = {}
        for asset, payout in payout_data.items():
            if payout >= min_payout and "_otc" in asset:
                pairs[asset] = {
                    'id': asset,
                    'payout': payout,
                    'type': 'binary',
                    'history': deque(maxlen=200)  # Increased for fractal analysis
                }
        logger.info(f"Payout data loaded: {len(pairs)} pairs with payout >= {min_payout}%")
        return True
    except asyncio.TimeoutError:
        logger.error("Timeout while fetching payout data.")
        return False
    except Exception as e:
        logger.error(f"Failed parsing payout: {e}")
        return False

# === HISTORY LOADER ===
async def get_df():
    try:
        for pair in pairs:
            candles = await api.get_candles(pair, period, 1200)  # Fetch more candles for 5s timeframe
            if candles and isinstance(pairs[pair].get('history'), deque):
                for candle in candles:
                    pairs[pair]['history'].append({
                        'time': candle['time'],
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close']
                    })
            await asyncio.sleep(0.5)  # Faster for 5s timeframe
        logger.info("Historical data loaded for all pairs")
        return True
    except Exception as e:
        logger.error(f"get_df() error: {e}")
        return False

# === ORDER EXECUTION ===
async def buy(amount: float, pair: str, action: str, expiration: int, last: dict):
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
        
        # Handle different timestamp formats
        if isinstance(df['time'].iloc[0], str):
            # ISO format timestamp (e.g., "2025-06-15T08:32:01.078Z")
            df['time'] = pd.to_datetime(df['time'], utc=True)
        else:
            # Unix timestamp
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        
        df.set_index('time', inplace=True)
        
        # For 5s timeframe, resample to ensure we have proper OHLC
        df_resampled = df.resample(f'{period}s').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        df_resampled.reset_index(inplace=True)
        
        # Remove future data - make datetime.now() timezone-aware for comparison
        now_utc = pd.Timestamp.now(tz='UTC')
        df_resampled = df_resampled[df_resampled['time'] < now_utc]
        
        return df_resampled
    except Exception as e:
        logger.error(f"make_df error: {e}")
        return pd.DataFrame()

# === FRACTAL CHAOS BANDS STRATEGY ===
async def fractal_chaos_strategy():
    global websocket_is_connected
    while True:
        try:
            all_pairs = list(pairs.keys())
            if len(all_pairs) < 3:
                logger.warning("Not enough pairs with sufficient payout.")
                if not await reconnect():
                    return
                continue

            random.shuffle(all_pairs)
            watchlist = all_pairs[:3]  # Reduced for 5s timeframe
            logger.info(f"Watching: {', '.join(watchlist)}")

            start_time = time.time()
            trade_count = 0

            while time.time() - start_time < CYCLE_TIME:
                if not websocket_is_connected:
                    raise Exception("Websocket connection lost")

                for pair in watchlist:
                    history = pairs[pair].get('history')
                    if not history or len(history) < MIN_HISTORY_LENGTH:
                        continue

                    df = make_df(history)
                    if df.empty or len(df) < MIN_DF_LENGTH:
                        continue

                    # Calculate Fractals
                    fractal_up, fractal_down = calculate_fractals(df, FRACTAL_PERIOD)
                    
                    # Calculate Chaos Bands
                    upper_band, lower_band, sma = calculate_chaos_bands(df, CHAOS_BANDS_PERIOD)
                    
                    # Add to dataframe
                    df['fractal_up'] = fractal_up
                    df['fractal_down'] = fractal_down
                    df['upper_band'] = upper_band
                    df['lower_band'] = lower_band
                    df['sma'] = sma
                    
                    # Detect breakout signals
                    signals = detect_breakout(df, fractal_up, fractal_down, upper_band, lower_band)
                    
                    if len(signals) == 0:
                        continue
                    
                    # Get the latest signal
                    latest_signal = signals[-1]
                    last = df.iloc[-1]
                    last_dict = last.to_dict()
                    
                    # Get price and band values
                    price = last['close']
                    upper = last['upper_band'] 
                    lower = last['lower_band']
                    middle = last['sma']
                    
                    # Log indicators
                    log_fractal(pair, last_dict)
                    
                    # Debug logging
                    logger.info(f"DEBUG {pair}: Price={price:.5f}, Upper={upper:.5f}, Lower={lower:.5f}, Signal={latest_signal}")
                    
                    # Simplified filters - remove restrictive conditions
                    band_width = (upper - lower) / middle if middle > 0 else 0
                    logger.info(f"DEBUG {pair}: BandWidth={band_width:.4f}, Middle={middle:.5f}")
                    
                    # Execute trades based on signals with relaxed conditions
                    if latest_signal == "call":
                        logger.info(f"FRACTAL BREAKOUT CALL SIGNAL on {pair} - Price: {price:.5f}, Upper: {upper:.5f}")
                        await buy(TRADE_AMOUNT, pair, "call", expiration, last_dict)
                        trade_count += 1
                    
                    elif latest_signal == "put":
                        logger.info(f"FRACTAL BREAKOUT PUT SIGNAL on {pair} - Price: {price:.5f}, Lower: {lower:.5f}")
                        await buy(TRADE_AMOUNT, pair, "put", expiration, last_dict)
                        trade_count += 1

                # Shorter wait for 5s timeframe
                await asyncio.sleep(2)

            logger.info(f"Cycle done - trades placed: {trade_count}")

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
    # For 5s timeframe, sync to 5s intervals
    current_time = datetime.now()
    seconds = current_time.second
    next_interval = ((seconds // 5) + 1) * 5
    if next_interval >= 60:
        next_interval = 0
        target_time = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
    else:
        target_time = current_time.replace(second=next_interval, microsecond=0)
    
    if sleep:
        sleep_time = (target_time - datetime.now()).total_seconds()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
    return int(target_time.timestamp())

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
                await fractal_chaos_strategy()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            if not await reconnect():
                logger.error("Max reconnection attempts reached. Exiting...")
                break
            continue

if __name__ == "__main__":
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
import csv
import os
from datetime import datetime

# New log file for OHLC verification
OHLC_VERIFICATION_LOG = "ohlc_verification.csv"
RAW_CANDLE_LOG = "raw_candles.csv"

def log_raw_candles(pair: str, candles: list) -> None:
    """Log raw candle data as received from API"""
    for candle in candles[-10:]:  # Log last 10 candles
        log_data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "pair": pair,
            "candle_time": candle.get('time'),
            "candle_time_formatted": datetime.fromtimestamp(candle.get('time', 0)).strftime('%Y-%m-%d %H:%M:%S') if isinstance(candle.get('time'), (int, float)) else str(candle.get('time')),
            "raw_open": candle.get('open'),
            "raw_high": candle.get('high'),
            "raw_low": candle.get('low'),
            "raw_close": candle.get('close'),
            "data_source": "API_RAW"
        }
        
        file_exists = os.path.isfile(RAW_CANDLE_LOG)
        with open(RAW_CANDLE_LOG, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)

def log_processed_ohlc(pair: str, df: pd.DataFrame, source: str = "PROCESSED") -> None:
    """Log processed OHLC data after resampling"""
    if df.empty:
        return
        
    # Log last 5 processed candles
    for idx in range(max(0, len(df)-5), len(df)):
        row = df.iloc[idx]
        log_data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "pair": pair,
            "candle_time": row['time'] if 'time' in df.columns else row.name,
            "candle_time_formatted": str(row['time'] if 'time' in df.columns else row.name),
            "processed_open": round(row['open'], 6),
            "processed_high": round(row['high'], 6),
            "processed_low": round(row['low'], 6),
            "processed_close": round(row['close'], 6),
            "data_source": source,
            "df_length": len(df),
            "resample_period": f"{period}s"
        }
        
        file_exists = os.path.isfile(OHLC_VERIFICATION_LOG)
        with open(OHLC_VERIFICATION_LOG, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)

def log_timing_sync(pair: str, api_time: any, local_time: datetime, sync_target: int) -> None:
    """Log timing synchronization data"""
    TIMING_LOG = "timing_sync.csv"
    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        "pair": pair,
        "api_time": str(api_time),
        "local_time": local_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        "sync_target": sync_target,
        "sync_target_formatted": datetime.fromtimestamp(sync_target).strftime('%Y-%m-%d %H:%M:%S'),
        "time_diff_seconds": (datetime.fromtimestamp(sync_target) - local_time).total_seconds()
    }
    
    file_exists = os.path.isfile(TIMING_LOG)
    with open(TIMING_LOG, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

# Enhanced get_df function with logging
async def get_df_with_logging():
    try:
        for pair in pairs:
            logger.info(f"Fetching candles for {pair}...")
            candles = await api.get_candles(pair, period, 1200)
            
            if candles:
                # Log raw candle data
                log_raw_candles(pair, candles)
                
                # Process candles
                if isinstance(pairs[pair].get('history'), deque):
                    for candle in candles:
                        pairs[pair]['history'].append({
                            'time': candle['time'],
                            'open': candle['open'],
                            'high': candle['high'],
                            'low': candle['low'],
                            'close': candle['close']
                        })
                
                # Create and log processed DataFrame
                history = pairs[pair].get('history')
                if history:
                    df = make_df(history)
                    if not df.empty:
                        log_processed_ohlc(pair, df, "INITIAL_LOAD")
                        
            await asyncio.sleep(0.5)
        
        logger.info("Historical data loaded and logged for all pairs")
        return True
    except Exception as e:
        logger.error(f"get_df_with_logging() error: {e}")
        return False

# Enhanced make_df function with detailed logging
def make_df_with_logging(history, pair_name="UNKNOWN"):
    if not history or not isinstance(history, (list, deque)):
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(history).reset_index(drop=True)
        df = df.sort_values(by='time').reset_index(drop=True)
        
        # Log before processing
        logger.info(f"Processing {len(df)} raw candles for {pair_name}")
        
        # Handle different timestamp formats
        if isinstance(df['time'].iloc[0], str):
            df['time'] = pd.to_datetime(df['time'], utc=True)
        else:
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        
        df.set_index('time', inplace=True)
        
        # Log before resampling
        logger.info(f"Resampling {len(df)} candles to {period}s intervals for {pair_name}")
        
        # Resample
        df_resampled = df.resample(f'{period}s').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        df_resampled.reset_index(inplace=True)
        
        # Remove future data
        now_utc = pd.Timestamp.now(tz='UTC')
        df_before_filter = len(df_resampled)
        df_resampled = df_resampled[df_resampled['time'] < now_utc]
        df_after_filter = len(df_resampled)
        
        logger.info(f"Filtered {df_before_filter - df_after_filter} future candles for {pair_name}")
        logger.info(f"Final processed candles for {pair_name}: {len(df_resampled)}")
        
        # Log processed data
        if not df_resampled.empty:
            log_processed_ohlc(pair_name, df_resampled, "RESAMPLED")
        
        return df_resampled
        
    except Exception as e:
        logger.error(f"make_df_with_logging error for {pair_name}: {e}")
        return pd.DataFrame()

# Enhanced wait function with timing logs
async def wait_with_logging(sleep=True):
    current_time = datetime.now()
    seconds = current_time.second
    next_interval = ((seconds // 5) + 1) * 5
    
    if next_interval >= 60:
        next_interval = 0
        target_time = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
    else:
        target_time = current_time.replace(second=next_interval, microsecond=0)
    
    # Log timing synchronization
    sync_target = int(target_time.timestamp())
    for pair in list(pairs.keys())[:1]:  # Log for first pair only
        log_timing_sync(pair, "N/A", current_time, sync_target)
    
    if sleep:
        sleep_time = (target_time - datetime.now()).total_seconds()
        if sleep_time > 0:
            logger.info(f"Syncing to next 5s interval in {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
    
    return sync_target

# Verification function to compare with PocketOption
def verify_ohlc_alignment(pair: str, expected_ohlc: dict, tolerance: float = 0.00001) -> bool:
    """
    Compare your bot's OHLC with PocketOption's displayed values
    expected_ohlc should be: {'open': x, 'high': x, 'low': x, 'close': x, 'time': 'YYYY-MM-DD HH:MM:SS'}
    """
    VERIFICATION_RESULTS = "ohlc_comparison.csv"
    
    try:
        # Get latest processed data for the pair
        history = pairs[pair].get('history')
        if not history:
            return False
            
        df = make_df_with_logging(history, pair)
        if df.empty:
            return False
            
        # Find the matching candle by time
        target_time = pd.to_datetime(expected_ohlc['time'], utc=True)
        
        # Find closest candle
        df['time_diff'] = abs(df['time'] - target_time)
        closest_candle = df.loc[df['time_diff'].idxmin()]
        
        # Compare values
        bot_ohlc = {
            'open': closest_candle['open'],
            'high': closest_candle['high'],
            'low': closest_candle['low'],
            'close': closest_candle['close']
        }
        
        # Calculate differences
        differences = {}
        matches = {}
        for key in ['open', 'high', 'low', 'close']:
            diff = abs(bot_ohlc[key] - expected_ohlc[key])
            differences[key] = diff
            matches[key] = diff <= tolerance
        
        # Log comparison results
        comparison_data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "pair": pair,
            "target_time": expected_ohlc['time'],
            "bot_time": str(closest_candle['time']),
            "time_diff_seconds": closest_candle['time_diff'].total_seconds(),
            "bot_open": round(bot_ohlc['open'], 6),
            "pocket_open": round(expected_ohlc['open'], 6),
            "open_diff": round(differences['open'], 6),
            "open_match": matches['open'],
            "bot_high": round(bot_ohlc['high'], 6),
            "pocket_high": round(expected_ohlc['high'], 6),
            "high_diff": round(differences['high'], 6),
            "high_match": matches['high'],
            "bot_low": round(bot_ohlc['low'], 6),
            "pocket_low": round(expected_ohlc['low'], 6),
            "low_diff": round(differences['low'], 6),
            "low_match": matches['low'],
            "bot_close": round(bot_ohlc['close'], 6),
            "pocket_close": round(expected_ohlc['close'], 6),
            "close_diff": round(differences['close'], 6),
            "close_match": matches['close'],
            "overall_match": all(matches.values())
        }
        
        file_exists = os.path.isfile(VERIFICATION_RESULTS)
        with open(VERIFICATION_RESULTS, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=comparison_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(comparison_data)
        
        logger.info(f"OHLC Verification {pair}: Match={all(matches.values())}, Diffs={differences}")
        return all(matches.values())
        
    except Exception as e:
        logger.error(f"OHLC verification error for {pair}: {e}")
        return False

# Enhanced strategy function with detailed logging
async def fractal_chaos_strategy_with_logging():
    global websocket_is_connected
    while True:
        try:
            all_pairs = list(pairs.keys())
            if len(all_pairs) < 3:
                logger.warning("Not enough pairs with sufficient payout.")
                if not await reconnect():
                    return
                continue

            random.shuffle(all_pairs)
            watchlist = all_pairs[:3]
            logger.info(f"Watching: {', '.join(watchlist)}")

            start_time = time.time()
            trade_count = 0

            while time.time() - start_time < CYCLE_TIME:
                if not websocket_is_connected:
                    raise Exception("Websocket connection lost")

                for pair in watchlist:
                    history = pairs[pair].get('history')
                    if not history or len(history) < MIN_HISTORY_LENGTH:
                        continue

                    # Use enhanced make_df with logging
                    df = make_df_with_logging(history, pair)
                    if df.empty or len(df) < MIN_DF_LENGTH:
                        continue

                    # Calculate indicators (your existing code)
                    fractal_up, fractal_down = calculate_fractals(df, FRACTAL_PERIOD)
                    upper_band, lower_band, sma = calculate_chaos_bands(df, CHAOS_BANDS_PERIOD)
                    
                    df['fractal_up'] = fractal_up
                    df['fractal_down'] = fractal_down
                    df['upper_band'] = upper_band
                    df['lower_band'] = lower_band
                    df['sma'] = sma
                    
                    # Log the final analysis data
                    log_processed_ohlc(pair, df.tail(1), "FINAL_ANALYSIS")
                    
                    # Continue with your existing strategy logic...
                    signals = detect_breakout(df, fractal_up, fractal_down, upper_band, lower_band)
                    
                    if len(signals) == 0:
                        continue
                    
                    latest_signal = signals[-1]
                    last = df.iloc[-1]
                    last_dict = last.to_dict()
                    
                    # Enhanced logging with exact values
                    price = last['close']
                    logger.info(f"EXACT VALUES {pair}: Time={last['time']}, O={last['open']:.6f}, H={last['high']:.6f}, L={last['low']:.6f}, C={last['close']:.6f}")
                    
                    # Rest of your strategy logic...
                    log_fractal(pair, last_dict)
                    
                    if latest_signal in ["call", "put"]:
                        logger.info(f"SIGNAL EXECUTION: {pair} | {latest_signal.upper()} | Price: {price:.6f}")
                        await buy(TRADE_AMOUNT, pair, latest_signal, expiration, last_dict)
                        trade_count += 1

                await asyncio.sleep(2)

            logger.info(f"Cycle done - trades placed: {trade_count}")

        except Exception as e:
            logger.error(f"Strategy error: {e}")
            if not await reconnect():
                return
            continue