import requests
import robin_stocks.robinhood as robin
import pyotp
import pandas as pd
import numpy as np
import talib
from colorama import Fore
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def calculate_technical_indicators(df):
  close_prices = df['close_price'].astype(float)
  sma = talib.SMA(close_prices, timeperiod=20)  # Simple Moving Average
  rsi = talib.RSI(close_prices, timeperiod=14)  # Relative Strength Index
  macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
  upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)  # Bollinger Bands

  df['sma'] = sma
  df['rsi'] = rsi
  df['macd'] = macdhist
  df['upper_band'] = upper
  df['lower_band'] = lower

  return df

def preprocess_data(df):
  df['time'] = pd.to_datetime(df['begins_at'])
  df['day_of_week'] = df['time'].dt.dayofweek
  df['hour'] = df['time'].dt.hour

  df = calculate_technical_indicators(df)

  # Drop unnecessary columns
  df = df.drop(columns=['time', 'begins_at', 'session', 'interpolated', 'symbol', 'volume'])

  # Fill missing values with 0
  df = df.fillna(0)

  return df

def train_model(X, y):
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

  # Train a support vector machine (SVM) classifier
  model = SVC()
  model.fit(X_train, y_train)

  # Train a random forest classifier
  # model = RandomForestClassifier()
  # model.fit(X_train, y_train)

  return model

def analyze_candlestick_patterns(df):
  patterns = {
    "engulfing": "CDLENGULFING",
    "hammer": "CDLHAMMER",
    "hanging_man": "CDLHANGINGMAN",
    "bullish_engulfing": "CDLENGULFING",
    "bearish_engulfing": "CDLENGULFING",
    "bullish_tri_star": "CDLTRISTAR",
    "bearish_tri_star": "CDLTRISTAR",
    "piercing_pattern": "CDLPIERCING",
    "dark_cloud_cover": "CDLDARKCLOUDCOVER",
    "bullish_harami": "CDLHARAMI",
    "bearish_harami": "CDLHARAMI",
    "bullish_kicker": "CDLKICKINGBYLENGTH",
    "bearish_kicker": "CDLKICKINGBYLENGTH",
    "morning_star": "CDLMORNINGSTAR",
    "evening_star": "CDLEVENINGSTAR",
    "three_white_soldiers": "CDL3WHITESOLDIERS",
    "three_black_crows": "CDL3BLACKCROWS",
    "upside_gap_two_crows": "CDLUPSIDEGAP2CROWS",
    "three_inside_up": "CDL3INSIDE",
    "three_inside_down": "CDL3INSIDE",
    "three_outside_up": "CDL3OUTSIDE",
    "three_outside_down": "CDL3OUTSIDE",
    "inverted_hammer": "CDLINVERTEDHAMMER",
    "bullish_abandoned_baby": "CDLABANDONEDBABY",
    "bullish_belthold": "CDLBELTHOLD",
    "bearish_belthold": "CDLBELTHOLD",
    "three_line_strike": "CDL3LINESTRIKE",
    "advance_block": "CDLADVANCEBLOCK",
    "bullish_stick_sandwich": "CDLSTICKSANDWICH",
    "bearish_stick_sandwich": "CDLSTICKSANDWICH",
    "matching_low": "CDLMATCHINGLOW",
    "ladder_bottom": "CDLLADDERBOTTOM",
    "bullish_breakaway": "CDLBREAKAWAY",
    "bearish_breakaway": "CDLBREAKAWAY",
    "tasuki_gap": "CDLTASUKIGAP",
    "bearish_separating_lines": "CDLSEPARATINGLINES",
    "soheil_pko": "CDL2CROWS",
    "shooting_star": "CDLSHOOTINGSTAR",
    "bearish_abandoned_baby": "CDLABANDONEDBABY",
    "bearish_meeting_line": "CDLMATHOLD"
  }

  df['pattern'] = sum(getattr(talib, pattern)(df['open_price'], df['high_price'], df['low_price'], df['close_price']) 
                      for pattern in patterns.values())

  return df

def get_index(ticker, asset_type):
  # Fetch historical candlestick data for the last year
  if asset_type == 'stocks':
    get_historicals = robin.get_stock_historicals
  elif asset_type == 'crypto':
    get_historicals = robin.get_crypto_historicals
  else:
    print("Invalid asset type selected.")
    return
  
  candlestick_data = get_historicals(ticker, interval='hour', span='month')
  
  df = pd.DataFrame(candlestick_data)
  df = preprocess_data(df)
  df = analyze_candlestick_patterns(df)

  # Determine the current state of the market
  current_price = float(df.iloc[-1]['close_price'])
  previous_close = float(df.iloc[-2]['close_price'])
  current_sma = df.iloc[-1]['sma']
  current_rsi = df.iloc[-1]['rsi']
  current_macd = df.iloc[-1]['macd']
  current_pattern = df.iloc[-1]['pattern']

  score = 0

  # Check if the stock is expected to fall based on indicators and patterns
  if current_price < previous_close:
    score += 1
  if current_price < current_sma:
    score += 1
  if current_rsi > 70:
    score += 1
  if current_macd < 0:
    score += 1
  if current_pattern < 0:
    score += 1

  threshold = 3

  expected_fall = score >= threshold

  return expected_fall

def check_portfolio(asset_type):
  if asset_type == 'stocks':
    owned_assets = robin.build_holdings()
  elif asset_type == 'crypto':
    owned_assets = robin.crypto.get_crypto_positions()
  else:
    print("Invalid asset type selected.")
    return

  if asset_type == 'stocks':
    for asset_ticker, asset_data in owned_assets.items():
      asset_name = asset_data['name']
      predicted_action = get_index(asset_ticker, asset_type)

      try:
        if predicted_action:
          print(Fore.RED + f"Recommended to sell {asset_name} ({asset_ticker})")
        else:
          print(Fore.GREEN + f"{asset_name} ({asset_ticker}) is not predicted to fall")
      except Exception as e:
        print(f"Error occurred for {asset_name} ({asset_ticker}): {e}")
  elif asset_type == 'crypto':
    for asset_data in owned_assets:
      asset_ticker = asset_data['currency']['code']
      asset_name = asset_data['currency']['name']
      predicted_action = get_index(asset_ticker, asset_type)

      try:
        if predicted_action:
          print(Fore.RED + f"Recommended to sell {asset_name} ({asset_ticker})")
        else:
          print(Fore.GREEN + f"{asset_name} ({asset_ticker}) is not predicted to fall")
      except Exception as e:
        print(f"Error occurred for {asset_name} ({asset_ticker}): {e}")

def execute_decisions():
  user_input = input("Select asset type (s for stock, c for crypto): ")
  asset_type = ""

  if user_input.lower() == "s":
    asset_type = "stocks"
  elif user_input.lower() == "c":
    asset_type = "crypto"
  else:
    print("Invalid asset type selected.")
    return

  check_portfolio(asset_type)  # Get the recommended decisions for each stock

  if asset_type == 'stocks':
    owned_assets = robin.build_holdings()
  elif asset_type == 'crypto':
    owned_assets = robin.crypto.get_crypto_positions()

  buying_power = float(robin.load_account_profile()['buying_power'])
  print(Fore.WHITE + f"Current buying power: ${buying_power}")

  if asset_type == 'stocks':
    for asset_ticker, asset_data in owned_assets.items():
      asset_name = asset_data['name']
      predicted_action = get_index(asset_ticker, asset_type)

      if not predicted_action:  # Recommended to buy
        decision = input(f"Do you want to buy more shares of {asset_name} ({asset_ticker})? (y/n): ")
        if decision.lower() == 'y':
          buy_amount = float(input("Enter the amount of dollars to invest: "))
          if buy_amount <= buying_power:
            BUY(asset_ticker, buy_amount)
          else:
            print("Insufficient buying power. Consider selling some assets.")
            sell_decision = input("Do you want to sell a certain number of dollars of one of your assets? (y/n): ")
            if sell_decision.lower() == 'y':
              for sell_asset_ticker, sell_asset_data in owned_assets.items():
                sell_asset_name = sell_asset_data['name']
                sell_asset_quantity = float(sell_asset_data['quantity'])
                sell_asset_price = float(sell_asset_data['price'])

                sell_amount = float(input(f"Enter the amount of dollars to sell from {sell_asset_name} ({sell_asset_ticker}): "))
                max_sell_amount = sell_asset_quantity * sell_asset_price

                if sell_amount <= max_sell_amount:
                  SELL(sell_asset_ticker, sell_amount)
                  buying_power += sell_amount
                  if buy_amount <= buying_power:
                    BUY(asset_ticker, buy_amount)
                    break
                else:
                  print("Invalid sell amount. Cannot sell more than the asset value.")
                  break
      else:  # Recommended to sell
        decision = input(f"Do you want to sell shares of {asset_name} ({asset_ticker})? (y/n): ")
        if decision.lower() == 'y':
          sell_option = input("Sell fully or specify a fraction? (full/fraction): ")
          if sell_option.lower() == 'full':
            SELL(asset_ticker, float(asset_data['quantity']) * float(robin.get_latest_price(asset_ticker)[0]))
          elif sell_option.lower() == 'fraction':
            sell_fraction = float(input("Enter the fraction of shares to sell (0-1): "))
            sell_quantity = float(asset_data['quantity']) * sell_fraction * float(robin.get_latest_price(asset_ticker)[0])
            SELL(asset_ticker, sell_quantity)
  elif asset_type == 'crypto':
    for asset_data in owned_assets:
      asset_ticker = asset_data['currency']['code']
      asset_name = asset_data['currency']['name']
      predicted_action = get_index(asset_ticker, asset_type)

      if not predicted_action:  # Recommended to buy
        decision = input(f"Do you want to buy more {asset_name} ({asset_ticker})? (y/n): ")
        if decision.lower() == 'y':
          buy_amount = float(input("Enter the amount of dollars to invest: "))
          if buy_amount <= buying_power:
            BUY(asset_ticker, buy_amount)
          else:
            print("Insufficient buying power. Consider selling some assets.")
            sell_decision = input("Do you want to sell a certain number of dollars of one of your assets? (y/n): ")
            if sell_decision.lower() == 'y':
              for sell_asset_data in owned_assets:
                sell_asset_ticker = sell_asset_data['asset_ticker']
                sell_asset_name = sell_asset_data['asset_name']
                sell_asset_quantity = float(sell_asset_data['quantity'])
                sell_asset_price = float(sell_asset_data['price'])

                sell_amount = float(input(f"Enter the amount of dollars to sell from {sell_asset_name} ({sell_asset_ticker}): "))
                max_sell_amount = sell_asset_quantity * sell_asset_price

                if sell_amount <= max_sell_amount:
                  SELL(sell_asset_ticker, sell_amount)
                  buying_power += sell_amount
                  if buy_amount <= buying_power:
                    BUY(asset_ticker, buy_amount)
                    break
                else:
                  print("Invalid sell amount. Cannot sell more than the asset value.")
                  break
      else:  # Recommended to sell
        decision = input(f"Do you want to sell {asset_name} ({asset_ticker})? (y/n): ")
        if decision.lower() == 'y':
          sell_option = input("Sell fully or specify a fraction? (full/fraction): ")
          if sell_option.lower() == 'full':
            SELL(asset_ticker, float(asset_data['quantity']) * float(robin.get_latest_price(asset_ticker)[0]))
          elif sell_option.lower() == 'fraction':
            sell_fraction = float(input(f"Enter the fraction of {asset_name} ({asset_ticker}) to sell (0-1): "))
            sell_quantity = float(asset_data['quantity']) * sell_fraction * float(robin.get_latest_price(asset_ticker)[0])
            SELL(asset_ticker, sell_quantity)

def QUOTE(ticker, asset_type):
  if asset_type == 'stocks':
    r = robin.get_latest_price(ticker)
  elif asset_type == 'crypto':
    r = robin.get_crypto_quote(ticker)
  else:
    print("Invalid asset type selected.")
    return
  print(ticker.upper() + ": $" + str(r[0]))

def BUY(ticker, amount):
  try:
    latest_price = float(robin.get_latest_price(ticker)[0])
    quantity = round(amount / latest_price, 8)
    r = robin.order_buy_market(ticker, quantity)
    print(r)
  except requests.exceptions.RequestException:
    print("Error occurred during the transaction. The market might be closed.")


def SELL(ticker, amount):
  try:
    latest_price = float(robin.get_latest_price(ticker)[0])
    quantity = round(amount / latest_price, 8)
    r = robin.order_sell_market(ticker, quantity)
    print(r)
  except requests.exceptions.RequestException:
    print("Error occurred during the transaction. The market might be closed.")


lines = open("keys.txt").read().splitlines()
KEY = lines[0]
EMAIL = lines[1]
PASSWORD = lines[2]
totp = pyotp.TOTP(KEY).now()
login = robin.login(EMAIL, PASSWORD, mfa_code=totp)

# Example usage
stock_ticker = "TSLA"
index = get_index(stock_ticker, "stocks")
if index:
  print(f"Recommended to sell {stock_ticker}")
else:
  print(f"{stock_ticker} is not predicted to fall")

execute_decisions()