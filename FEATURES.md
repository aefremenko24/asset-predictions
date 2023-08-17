# calculate_technical_indicators
- Used SMA over EMA to avoid getting whipsawed. (Less responsive to current trends but 
  generally a more effective indicator)
- RSI measures how much a stock has gained on its up days relative to how much it's lost
  on its down days (over the default of 14 trading periods)
- MACD (moving average convergence divergence) subtracts two exponential moving averages (standard
  26 day and 12 day period resspectively) to create the MACD line, the signal line is created
  by analysing the 9 day EMA; MACDhist is then created by subtraction and is used as an indicator
- Bollinger bands: a standard deviation from the moving average. Usually plotted two standard
  deviations away from the moving average (95% of security's historical price). Helps see when the 
  trends might be overextended and reverse. TO BE IMPLEMENTED

# preprocess_data
- Converts the raw data pulled from the API into the data ready to be processed by the TA-lib 
  functions
- The raw data comes out in columns: "begins_at (date), open_price, close_price, high_price, 
  low_price, volume session, interpolated symbol"
- After we format the time in the dataframe properly and analyze the technical indicators,
  we can drop the unnecessary columns, being "time, begins_at, session, interpolated, symbol,
  volume", since they are not needed anymore, as well as fill in missing values with 0s
  
# train_model
- Uses Support Vector Machine (SVM) classifier to analyze the collected dataframe
- We first standardize the data so that it has a mean of 0 and SD of 1 (using sklearn)
- We then use the train_test_split function from the sklearn.model_selection module to 
  randomly splits the data into training and testing subsets, where 80% of the data is used 
  for training (X_train and y_train) and 20% is reserved for testing (X_test and y_test). 
  The random_state=42 ensures reproducibility of the split.
- We create the SVM classifier from the sklearn.svm module to train the model using the fit
  function on previously created X_train y_train

# analyze_candlestick_patterns
- Makes additional predictions about the collected candlestick data using most well-known 
  candlestick patterns
- We first map candlestick pattern names to their respective ta-lib function names
- We then analyze the previously modified dataframe using the correspronding TA-lib function 
  determined by getattr(talib, pattern). We then applies that function to the OHLC price data 
  from the dataframe, where the result of each pattern analysis is accumulated using the sum() 
  function and stored in the new "pattern" column in the df.

# get_index
- Depending on the type of asset (stock/crypto), assign the function that will draw data
  from robinhood; draw the data for the past month in 1-hour intervals
- Convert data into a dataframe for more convenient use
- Run previously mentioned functions on the dataframe
- Determine the current state of the market by drawing the most recent rows from the dataframe
- Check if the stock is expected to fall based on indicators and patterns