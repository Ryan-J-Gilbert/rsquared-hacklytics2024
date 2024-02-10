from flask import Flask, render_template, request, flash, redirect, url_for
import matplotlib.pyplot as plt
import yfinance as yf
import time
from sklearn.ensemble import RandomForestRegressor
import os
from helpers import plot_helper, ml_plot_helper
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv('flasksecret')


# Define routes
@app.route('/')
def index():
    form_results = {}
    return render_template('index.html', results=form_results)

@app.route('/statarb')
def statarb():
    form_results = {}
    return render_template('statarb.html', results=form_results)

@app.route('/statresults', methods=['POST'])
def results():
    form_results={}
    # Extract user inputs from the form
    equity1 = request.form['equity1']
    equity2 = request.form['equity2']
    timeframe = request.form['timeframe']
    period = request.form['period']
    window_size = request.form['window_size']
    multiplier = request.form['multiplier']
    std_mult = request.form['std_mult']
    # confirm inputs are valid
    if not equity1 or not equity2 or not timeframe or not window_size or not multiplier:
        flash('All inputs must be provided', 'error')
        return redirect(url_for('statarb'))
    # try to cast window_size and multiplier to int, float
    try:
        window_size = int(window_size)
    except ValueError:
        flash('Window size must be an integer', 'error')
        return redirect(url_for('statarb'))
    try:
        multiplier = float(multiplier)
    except ValueError:
        flash('Multiplier must be a float', 'error')
        return redirect(url_for('statarb'))

    try:
        std_mult = int(std_mult)
    except ValueError:
        flash('Standard deviation multiplier must be an integer', 'error')
        return redirect(url_for('statarb'))

    form_results = {
        'equity1': equity1,
        'equity2': equity2,
        'timeframe': timeframe,
        'period': period,
        'window_size': window_size,
        'multiplier': multiplier,
        'std_mult': std_mult
    }

    equity1_df = yf.download(equity1, interval=timeframe, period=period)
    equity2_df = yf.download(equity2, interval=timeframe, period=period)

    if len(equity1_df) == 0:
        flash(f'No data found for {equity1}', 'error')
        return redirect(url_for('statarb'))
    if len(equity2_df) == 0:
        flash(f'No data found for {equity2}', 'error')
        return redirect(url_for('statarb'))
    
    # join the dfs with prefix eq1 and eq2
    equity1_df = equity1_df.add_prefix('eq1_')
    equity2_df = equity2_df.add_prefix('eq2_')

    # merge the dataframes
    df = equity1_df.join(equity2_df, how='outer')
    df['Difference'] = df['eq1_Close'] - df['eq2_Close']

    df[f'{window_size}_MA_Difference'] = df['Difference'].rolling(window=window_size).mean()
    df[f'{window_size}_MA_Difference_Difference'] = df['Difference'] - df[f'{window_size}_MA_Difference']
    
    df['Upper_Band'] = df[f'{window_size}_MA_Difference_Difference'].rolling(window=window_size*std_mult).std() * multiplier
    df['Lower_Band'] = -df[f'{window_size}_MA_Difference_Difference'].rolling(window=window_size*std_mult).std() * multiplier
    # add marks for when the difference difference is outside the bands
    df['Outside_Upper'] = df[f'{window_size}_MA_Difference_Difference'] > df['Upper_Band']
    df['Outside_Lower'] = df[f'{window_size}_MA_Difference_Difference'] < df['Lower_Band']

    # iterate through df, short the spread when the difference difference is above the upper band and long the spread when the difference difference is below the lower band
    position = 0
    df['return'] = 0
    df['position'] = 0
    for i in range(len(df)):
        if position == 0 and df[f'{window_size}_MA_Difference_Difference'].iloc[i] > df['Upper_Band'].iloc[i]:
            position = -1
            entry = df['Difference'].iloc[i]
            entry_cost = df['eq1_Close'].iloc[i] + df['eq2_Close'].iloc[i]
            df['position'].iloc[i] = -1
        elif position == 0 and df[f'{window_size}_MA_Difference_Difference'].iloc[i] < df['Lower_Band'].iloc[i]:
            position = 1
            entry = df['Difference'].iloc[i]
            entry_cost = df['eq1_Close'].iloc[i] + df['eq2_Close'].iloc[i]
            df['position'].iloc[i] = 1
        
        elif position == -1 and df[f'{window_size}_MA_Difference_Difference'].iloc[i] < df['Lower_Band'].iloc[i]:
            position = 0
            exitv = df['Difference'].iloc[i]
            df['return'].iloc[i] = (entry - exitv) / entry_cost

        elif position == 1 and df[f'{window_size}_MA_Difference_Difference'].iloc[i] > df['Upper_Band'].iloc[i]:
            position = 0
            exitv = df['Difference'].iloc[i]
            df['return'].iloc[i] = (exitv - entry) / entry_cost
        

    imgdata = plot_helper(df, window_size)
    return render_template('results.html', results=form_results, data = imgdata)




@app.route('/machinelearning')
def machinelearning():
    form_results = {}
    return render_template('mlindex.html', results=form_results)

@app.route('/mlresults', methods=['POST','GET'])
def mlresults():
    # get form data
    # these three same as before
    equity1 = request.form['equity1']
    timeframe = request.form['timeframe']
    period = request.form['period'] 
    # new
    estimators = request.form['estimators'] # 1 to 100
    features = request.form.getlist('features') # a list of the features keep in mind
    shift = request.form['shift'] # 0 to 10
    threshold = request.form['threshold'] # 0 to 1

    # confirm inputs are valid
    if not equity1 or not timeframe or not period or not estimators or not features or not shift or not threshold:
        flash('All inputs must be provided', 'error')
        return redirect(url_for('machinelearning'))

    # try to cast estimators, shift to int, threshold to float
    try:
        estimators = int(estimators)
    except ValueError:
        flash('Estimators must be an integer', 'error')
        return redirect(url_for('machinelearning'))
    
    try:
        shift = int(shift)
    except ValueError:
        flash('Shift must be an integer', 'error')
        return redirect(url_for('machinelearning'))
    
    try:
        threshold = float(threshold)
    except ValueError:
        flash('Threshold must be a float', 'error')
        return redirect(url_for('machinelearning'))
    
    form_results = {
        'equity1': equity1,
        'timeframe': timeframe,
        'period': period,
        'estimators': estimators,
        'features': features,
        'shift': shift,
        'threshold': threshold
    }

    # Fetch data using yfinance
    df = yf.download(equity1, interval=timeframe, period=period)
    # check if data was found
    if len(df) == 0:
        flash(f'No data found for {equity1}', 'error')
        return redirect(url_for('machinelearning'))

    # create features
    feature_list = []
    for feature in features:
        if feature == 'rsi':
            # calculate rsi from the close price
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            feature_list.append('rsi')
        elif feature == 'macd':
            # calculate the macd
            df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            feature_list.append('macd')
        elif feature == 'pctfrom100ma':
            # calculate the percent from the 100 day moving average
            df['100ma'] = df['Close'].rolling(window=100).mean()
            df['pctfrom100ma'] = (df['Close'] - df['100ma']) / df['100ma']
            feature_list.append('pctfrom100ma')
        elif feature == 'prevreturn':
            # calculate the previous day return
            df['prevreturn'] = df['Close'].pct_change()
            feature_list.append('prevreturn')

    # shift the features and add to list
    shifted_features = []
    for feature in feature_list:
        df[f'{feature}_shift{shift}'] = df[feature].shift(shift)
        shifted_features.append(f'{feature}_shift{shift}')

    # add shifted names to feature list
    for name in shifted_features:
        feature_list.append(name)

    # drop na
    df = df.dropna()

    # create target next (shift -1) Close higher than Open
    df['return'] = df['Close'].shift(-1) / df['Open'].shift(-1) 
    df['target'] = df['return'] > 1
    df['target'] = df['target'].astype(int)

    # create X and y
    X = df[feature_list]
    y = df['target']

    # get time index for split at 80% of data
    split_index = int(len(df) * 0.8)
    split_date = df.index[split_index]

    # split the data
    X_train = X[:split_date]
    X_test = X[split_date:]
    y_train = y[:split_date]
    y_test = y[split_date:]

    # print date rate for train and test
    print(f'Train date range: {X_train.index[0]} to {X_train.index[-1]}')
    print(f'Test date range: {X_test.index[0]} to {X_test.index[-1]}')

    # create the model
    model = RandomForestRegressor(n_estimators=estimators, random_state=42)
    model.fit(X_train, y_train)

    # get the predictions
    predictions = model.predict(X_test)

    # create the predictions dataframe
    preds = pd.DataFrame({'predictions': predictions, 'actual': y_test})

    # add back in return column
    preds['return'] = df['return'][split_date:]
    preds['met'] = preds['predictions'] > threshold
    preds['met'] = preds['met'].astype(int)

    # calculate the accuracy where threshold is met
    accuracy = preds[preds['predictions'] > threshold]['actual'].mean()

    # calculate returns, 1 if not met, return if met
    preds['return_strat'] = 1 + ((preds['return']-1) * preds['met'])

    data = ml_plot_helper(preds)

    # Render results template with the calculated results
    return render_template('mlresults.html', results=form_results, imgdata=data, accuracy=accuracy, count=preds['met'].sum())

if __name__ == '__main__':
    app.run(debug=True)
