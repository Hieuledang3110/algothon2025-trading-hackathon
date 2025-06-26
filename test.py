#Eric is using this for testing
 
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def SMA(timePeriod, stockPrices):
    if timePeriod > len(stockPrices)+1:
        raise ValueError
    
    priceSum = 0
    for price in stockPrices[len(stockPrices)-timePeriod:len(stockPrices)]:
        priceSum += price

    movingAverage = priceSum/timePeriod
    return movingAverage

def WMA(timePeriod, stockPrices):
    if timePeriod > len(stockPrices)+1:
        print(f'Error: WMA function was given a timePeriod of {timePeriod}, but the stock price history was only {len(stockPrices)+1} elements long')
        raise ValueError
    
    priceSum = 0
    weight = 1
    for i in range(len(stockPrices)-timePeriod,len(stockPrices)):
        priceSum += stockPrices[i]*weight
        weight += 1

    movingAverage = (priceSum*2)/(timePeriod*(timePeriod+1))
    return movingAverage

###Gotta do this
def EMA(stockPrices):
    stockValues = pd.DataFrame({'Stock_Values': stockPrices})
    result = stockValues.ewm(com=2).mean()
    numpy_array = result.to_numpy()
    return numpy_array

def getReturns(data):
    stockReturns = [[] for _ in range(len(data))]
    for i in range(len(data)):
        stockPrices = data[i]
        for j in range(1,len(stockPrices)):
            stockReturns[i].append((stockPrices[j]-stockPrices[j-1])/stockPrices[j-1])
    return stockReturns

def getCorrelations(data,maxLag):

    correlations = [[[] for _ in range(len(data))] for __ in range(2*maxLag+2)]
    p_values = [[[] for _ in range(len(data))] for __ in range(2*maxLag+2)]

    stockReturns = getReturns(data)

    for lag in range(-maxLag,maxLag+1):
        for i in range(len(stockReturns)):
            stock1Returns = stockReturns[i]
            for j in range(len(stockReturns)):
                stock2Returns = stockReturns[j]

                if lag > 0:  #Note: if lag is positive, stock2 lags behind stock1. Opposite otherwise.
                    buffer = ["This should not be read" for _ in range(lag)]
                    offsetStock2Returns = buffer + stock2Returns
                    correlation, p_value = pearsonr(stock1Returns[abs(lag):], offsetStock2Returns[abs(lag):-lag])
                elif lag < 0:
                    buffer = ["This should not be read" for _ in range(-lag)]
                    offsetStock1Returns = buffer + stock1Returns
                    correlation, p_value = pearsonr(offsetStock1Returns[abs(lag):lag], stock2Returns[abs(lag):])
                elif lag == 0:
                    correlation, p_value = pearsonr(stock1Returns, stock2Returns)

                lagIndex = maxLag + lag
                correlations[lagIndex][i].append(correlation)
                p_values[lagIndex][i].append(p_value)

    return correlations, p_values   

def linear_regression_single_array(values):
    X = np.arange(len(values)).reshape(-1, 1)  # Time indices as X
    y = np.array(values)
    
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Predict next value
    next_value = model.predict([[len(values)]])[0]

    return slope, intercept, next_value    


def momentumStrategy(data):
    maxDollarPosition = 10000
    timeHorizon = 5
    movingAverageLength = 30


    positions = []

    for prices in data:
        WMAs = [0 for _ in range(timeHorizon)]
        SMAs = [0 for _ in range(timeHorizon)]
        buyFactor = 0

        firstDay = len(prices)-timeHorizon
        for i in range(len(WMAs)):
            WMAs[i] = WMA(movingAverageLength,prices[:firstDay + i])
        for i in range(len(SMAs)):
            SMAs[i] = SMA(movingAverageLength,prices[:firstDay + i])

        significantDeviation = False
        for i in range(len(SMAs)):
            difference = abs(SMAs[i]-WMAs[i])
            average = (SMAs[i]+WMAs[i])/2
            if difference/average > 0.006:
                significantDeviation = True
        
        # if significantDeviation:
        #     if WMAs[-1] > WMAs[-4]*1.004:
        #         buyFactor = 1
        #     elif WMAs[-1] < WMAs[-4]/1.004:
        #         buyFactor = -1

        if WMAs[-1] > SMAs[-1]*1.01:
            buyFactor = 1
        

        
        position = buyFactor*maxDollarPosition/prices[-1]
        positions.append(position)
    
    return positions

#Extremely simple mean reversion just to test things out. No shorting for now.
def getMyPosition(data):

    # data = np.rot90(data)
    positions = []

    maxDollarPosition = 10000

    for prices in data:
        iLast = len(prices)

        longTermLength = 500
        midTermLength = 89
        shortTermLength = 29

        upTrendTolerance = 1.05
        downTrendTolerance = 0.83
        
        #Getting the average price during the last shotTermLength days
        if iLast < shortTermLength:
            recentPrices = data
        else:
            recentPrices = data[:shortTermLength]
        averagePrice = np.average(recentPrices)

        #maxVariation is a really basic approximation of each stock's volatility
        maxPrice = np.max(recentPrices)
        minPrice = np.min(recentPrices)
        maxVariation = max(np.abs(float(averagePrice-minPrice)), np.abs(float(maxPrice-averagePrice)))

        currentPrice = prices[-1]
        buyFactor = 0

        #By this definition, something is 'trending up' if their stock is up since midTermLength days ago :skull: 
        #This is unused and will probably stay unused. It seems pretty useless in its current form
        isTrendingUp = False
        isTrendingDown = False
        isLongTrendingUp = False
        isLongTrendingDown = False

        if iLast < midTermLength:
            historicalPrice = prices[0]
        else:
            historicalPrice = prices[-midTermLength]
        if currentPrice < historicalPrice*downTrendTolerance:
            isTrendingDown = True
        elif currentPrice > historicalPrice*upTrendTolerance:
            isTrendingUp = True

        if iLast < longTermLength:
            historicalPrice = prices[0]
        else:
            historicalPrice = prices[-longTermLength]
        if currentPrice < historicalPrice*downTrendTolerance:
            isLongTrendingDown = True
        elif currentPrice > historicalPrice*upTrendTolerance:
            isLongTrendingUp = True



            
        #Checking how far away the current price is from the short term average
        devianceFactor = (currentPrice-averagePrice)/maxVariation
        devianceFactor = max(min(devianceFactor,1),-1)  #clamp between -1,1


        #It seems that full sending when buying is the most profitable bruh.
        if (devianceFactor > 0.329):   
            if (devianceFactor < 0.541):
                buyFactor = (devianceFactor+1)/0.54
                buyFactor = 0

        #Upper and lower bounds for the devianceFactor to determine when to buy
        a = -1
        b = -0.145
        fullSendThreshold = 0
        tightnessPower = 3
        #It seems that just full send buying when the stock is down is also profitable bruh. The fancy formula here is completely uselss
        #This is probably where machine learning is needed
        if (devianceFactor < b):
            if (devianceFactor > a):
                c = 1/(b+1)
                buyFactor = -(c*(devianceFactor-fullSendThreshold))**tightnessPower + 1


        buyAmount = maxDollarPosition*buyFactor
        sharePosition = buyAmount/currentPrice

        positions.append(int(sharePosition))
        final_positions = np.array([int(pos) for pos in positions])
    
    return positions


def plotHeatmap(thisCor,title):
    plt.figure(figsize=(8, 6))
    plt.imshow(thisCor, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title)


def lagTrade(data):
    maxDollarPosition = 10000
    positions = [0 for _ in range(len(data))]

    EMAs = [[] for _ in range(len(data))]
    for i in range(len(EMAs)):
        EMAs[i] = EMA(data[i])

    correlations, p_values = getCorrelations(EMAs,1)
    correlatedPairs = []

    oneDayLagCorrelations = correlations[0]
    oneDayLagP_Values = p_values[0]
    for i in range(len(oneDayLagCorrelations)):
        for j in range(len(oneDayLagCorrelations[i])):
            if oneDayLagCorrelations[i][j] > 0.22:
                if oneDayLagP_Values[i][j] < 0.01:
                    correlatedPairs.append([i,j])
    
    stockReturns = getReturns(data)
    lastReturns = []
    for stock in stockReturns:
        lastReturns.append(stock[-1])

    for pair in correlatedPairs:
        lagger = pair[0]
        setter = pair[1]
        if lastReturns[setter] > 0.01:
            positions[lagger] = maxDollarPosition/data[lagger][-1]
    
    return positions


def smooth_trend_regression(values, long_window=200, smooth_window=20):
    # Step 1: Get long-term trend using all data
    X_all = np.arange(len(values)).reshape(-1, 1)
    long_trend = LinearRegression().fit(X_all, values)
    
    # Step 2: Smooth recent residuals
    trend_line = long_trend.predict(X_all)
    residuals = values - trend_line
    recent_residuals = residuals[-smooth_window:]
    smoothed_residual = np.mean(recent_residuals)
    
    # Combine: long trend + smoothed recent deviation
    next_trend = long_trend.predict([[len(values)]])[0]
    return next_trend + smoothed_residual


def get_single_day_prediction(prices, target_day, long_window=200, smooth_window=20):
    """Get prediction for just one specific day"""
    if target_day >= len(prices):
        return None
    
    # Use data up to target_day
    current_data = prices[:target_day]
    
    # Apply the smooth trend regression
    prediction = smooth_trend_regression(current_data, long_window, smooth_window)
    
    return prediction

def trendStrategy(data):
    positions = []
    maxDollarPosition = 10000

    for i in range(len(data)):
        prices = data[i]
        latestPrice = prices[-1]
        buyFactor = 0
        SMALength = 15
        gradientThreshold = 0.006


        smooth_predictions = []
        for i in range(4):
            if i == 0:
                smooth_predictions.append(smooth_trend_regression(prices,long_window=200,smooth_window=5))
            else:
                smooth_predictions.append(smooth_trend_regression(prices[:-i],long_window=200,smooth_window=5))


        
        gradient = smooth_predictions[0] - smooth_predictions[1]
        prevGradient = smooth_predictions[1] - smooth_predictions[2]

        gradient /= latestPrice
        prevGradient /= latestPrice

        #print(smooth_predictions)
        #print(smooth_predictions[-1],smooth_predictions[-2])
            
        #direction/gradient of the 30 day EMA is used to indicate the trend
        #ema_30 = pd.Series(prices).ewm(span=45).mean()
        #ema_30 = ema_30.values
        #sma_of_ema = pd.Series(ema_30).rolling(4).mean()
        #sma_of_ema = sma_of_ema.values
        #gradient = sma_of_ema[-1] - sma_of_ema[-2]

        #Alternatively use the ema of an sma of smooth predictions
        #sma_10 = pd.Series(prices).rolling(20).mean()
        #ema_of_sma = pd.Series(sma_10).ewm(span=4).mean()
        #ema_of_sma = ema_of_sma.values
        #gradient = ema_of_sma[-1] - ema_of_sma[-2]

        #This is just the quick ema of smooth predictions
        #smoother_predictions = pd.Series(smooth_predictions).ewm(span=4).mean()
        #smoother_predictions = smoother_predictions.values
        #gradient = smoother_predictions[-1] - smoother_predictions[-2]

        #gradient = smooth_predictions[-1] - smooth_predictions[-2]

        #Triple EMA
        #ema1 = pd.Series(prices).ewm(span=10).mean()
        #ema2 = ema1.ewm(span=10).mean()
        #ema3 = ema2.ewm(span=10).mean()
        #tema = 3 * ema1 - 3 * ema2 + ema3

        # Create upper and lower bounds
        #percentage = 0.02  # % bands
        #upper_bound = smooth_predictions[-1] * (1 + percentage)
        #lower_bound = smooth_predictions[-1] * (1 - percentage)
        
        #ema_4 = pd.Series(prices[-5:-1]).ewm(span=4).mean().values
        #lastEma_4 = ema_4[-1]

        #If momentum and price are both above/below the prediction, start thinning positions
        #print(i, latestPrice, ema_30[-1], gradient)
        #if latestPrice < ema_30[-1] and gradient < 0:
        #    buyFactor = -1
        if gradient < -gradientThreshold and gradient < prevGradient:
            buyFactor = -1
        elif gradient > gradientThreshold and gradient > prevGradient:
            buyFactor = 1


        buyAmount = maxDollarPosition*buyFactor
        sharePosition = buyAmount/latestPrice
        positions.append(int(sharePosition))

        #print(i, prediction, upper_bound, lower_bound, latestPrice, buyFactor)

    #print(positions)
    return positions  

def reversionStrategy(data):
    positions = []
    maxDollarPosition = 10000

    for i in range(len(data)):
        prices = data[i]
        latestPrice = prices[-1]
        buyFactor = 0

        prediction = smooth_trend_regression(prices, long_window=200, smooth_window=20)
        prevPrediction = smooth_trend_regression(prices[:-1], long_window=200, smooth_window=20)

        #direction/gradient of the 30 day EMA is used to indicate the trend
        ema_30 = pd.Series(prices).ewm(span=30).mean()
        ema_30 = ema_30.values
        gradient = ema_30[-1] - ema_30[-2]

        sma_30 = pd.Series(prices).rolling(30).mean()
        sma_30 = sma_30.values

        # Create upper and lower bounds
        percentage = 0.10  # % bands
        upper_bound = (prediction) * (1 + percentage)
        lower_bound = (prediction) * (1 - percentage)
        
        ema_4 = pd.Series(prices[-5:-1]).ewm(span=4).mean().values
        lastEma_4 = ema_4[-1]

        if i == 29:
            print(len(prices),latestPrice,gradient)

        
        #If momentum is up, buy low sell high
        if gradient > 0:
            if latestPrice < prediction:
                buyFactor = 1
            else:
                buyFactor = (latestPrice - prediction)/(percentage*prediction)
        
        #If momentum is down, short high buy low
        if gradient < 0:
            if latestPrice > prediction:
                buyFactor = -1
            else:
                buyFactor = (latestPrice - prediction)/(percentage*prediction)


        buyAmount = maxDollarPosition*buyFactor
        sharePosition = buyAmount/latestPrice
        positions.append(int(sharePosition))

        #print(i, prediction, upper_bound, lower_bound, latestPrice, buyFactor)

    #print(positions)
    return positions  

def breakoutStrategy2(prcSoFar):
    """
    Breakout strategy that buys when price exits above upper bound,
    sells when price exits below lower bound
   
    Args:
        prcSoFar: numpy array of shape (nInst, nDays) with price history
       
    Returns:
        numpy array of positions for each instrument
    """
    nInst, nDays = prcSoFar.shape
    positions = []

    print(nInst)
   
    for i in range(nInst):
        prices = prcSoFar[i, :]
    

        # Get current prediction and bounds
        current_prediction = smooth_trend_regression(prices, long_window=200, smooth_window=20)
           
        # Calculate bounds (2% bands)
        percentage = 0.02
        upper_bound = current_prediction * (1 + percentage)
        lower_bound = current_prediction * (1 - percentage)
           
        current_price = prices[-1]
           
        # Check for breakout (need at least 2 days)
        if nDays >= 2:
            # Get previous day's prediction and bounds
            prev_prices = prices[:-1]
            prev_prediction = smooth_trend_regression(prev_prices, long_window=200, smooth_window=20)
            prev_upper = prev_prediction * (1 + percentage)
            prev_lower = prev_prediction * (1 - percentage)
            previous_price = prices[-2]
               
            # Detect breakouts
            broke_above = (previous_price <= prev_upper) and (current_price > upper_bound)
            broke_below = (previous_price >= prev_lower) and (current_price < lower_bound)
               
            # Position sizing
            max_position_value = 10000 * 0.8  # 80% of limit for safety
            shares_to_trade = int(max_position_value / current_price) if current_price > 0 else 0
               
            if broke_above:
                positions.append(shares_to_trade)  # Buy
            elif broke_below:
                positions.append(shares_to_trade)  # Sell
            else:
                positions.append(0)  # No position

    return positions

def trendFollow(data):
    
    positions = []
    for i in range(len(data)):
        stockPrices = data[i]
        lastPrice = stockPrices[-1]
        smaRange = 20 # Change this below as well
        smoothingStrength = 3
        positionLeeway = 0.00

        sma_series = pd.Series(stockPrices).rolling(20).mean().dropna()
        ema_smoothed_sma = sma_series.ewm(span=smoothingStrength).mean()
        #print(ema_smoothed_sma, sma_series, stockPrices)
        diff = ema_smoothed_sma.iloc[-1] - ema_smoothed_sma.iloc[-2]

        prediction = smooth_trend_regression(stockPrices, long_window=200, smooth_window=20)

        buyFactor = 0

        #if the moving average suggests we are trending up
        if diff > 0:
            if prediction < lastPrice:
                buyFactor = 1
            elif prediction-positionLeeway < lastPrice:
                buyFactor = (prediction-lastPrice)/positionLeeway
            else:
                buyFactor = 0
        #if the moving average suggests we are trending down
        else:
            if prediction > lastPrice:
                buyFactor = -1
                buyFactor = 0
            elif prediction+positionLeeway > lastPrice:
                buyFactor = -(lastPrice-prediction)/positionLeeway
                buyFactor = 0
            else:
                buyFactor = 0
        
        buyAmount = 10000*buyFactor
        sharePosition = buyAmount/lastPrice
        positions.append(int(sharePosition))

    positions.reverse()
    return(positions)




data = np.loadtxt("prices.txt")
data = np.rot90(data)

days = list(range(1,len(data[0])+1))


'''
#average/median price of all stocks on a day
averages = np.mean(data, axis = 0)
temp = data # want the median to be the middle stock so im making it odd
medians = np.median(temp, axis = 0)
stockIndex = 4


for i in range(1):
    plt.plot(days,data[i])
    plt.plot(days,EMA(data[i])) 
plt.plot(days,averages, color="black", linewidth=5)

# plt.plot(days,medians, color="red", linewidth=5)

plt.show()



instrument = 6
prices = data[instrument]

# Calculate predictions from day 150 onwards
predictions1 = []
upperBound1 = []
lowerBound1 = []
predictions2 = []
upperBound2 = []
lowerBound2 = []
actual_days1 = []
actual_days2 = []
defaultFreedomFactor = 0.04
maxExpansionFactor = 16
maxDifferenceConsidered = 0.05


# Add SMA
smaRange = 20
sma = []
for i in range(smaRange-1, len(prices)):  # Start from day 30 (index 29)
    sma.append(np.mean(prices[i-smaRange+1:i+1]))

# Add EMA (e.g., 20-day EMA)
ema_20 = pd.Series(prices).ewm(span=20).mean()


# Or multiple EMAs for crossover strategies
ema_12 = pd.Series(sma).ewm(span=3).mean()
ema_4 = pd.Series(prices).ewm(span=4).mean()
ema_26 = pd.Series(prices).ewm(span=26).mean()

delta = pd.Series(prices).diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))


for day in range(20, len(prices)):
    # Use data up to current day to predict next day
    current_data = prices[:day]
    #slope, intercept, next_value1 = linear_regression_single_array(current_data[max(0,len(current_data)-300):])
    slope, intercept, next_value1 = linear_regression_single_array(ema_4[max(0,day-20):day])
    predictions1.append(next_value1)
    #difference1 = abs(sma[day-smaRange+1]-ema_12[day])/prices[day]
    #difference2 = abs(next_value1-ema_12[day])/prices[day]
    #expansionFactor = max(difference1,difference2)*(defaultFreedomFactor)*(maxExpansionFactor)/maxDifferenceConsidered
    #lowerBound.append(next_value + next_value*(defaultFreedomFactor+expansionFactor))
    #upperBound.append(next_value - next_value*(defaultFreedomFactor+expansionFactor))
    lowerBound1.append(next_value1 + next_value1*(defaultFreedomFactor))
    upperBound1.append(next_value1 - next_value1*(defaultFreedomFactor))
    actual_days1.append(day)


# Plot
plt.figure(figsize=(12, 6))
plt.plot(range(len(prices)), prices, color='black', label='Actual Prices', alpha=0.7)
#plt.plot(actual_days1, predictions1, 'r-', label='Daily Predictions 1', markersize=3)
#plt.plot(actual_days1, upperBound1, 'b-', label='Upper Bound 1', markersize=2)
#plt.plot(actual_days1, lowerBound1, 'b-', label='Lower Bound 1', markersize=2)
#plt.plot(actual_days1, upperBound2, 'g-', label='Upper Bound 2', markersize=2)
#plt.plot(actual_days1, lowerBound2, 'g-', label='Lower Bound 2', markersize=2)
plt.plot(range(smaRange-1, len(prices)), sma, 'g-', label='30-day SMA', linewidth=2)
#plt.plot(range(len(prices)), ema_20, 'orange', label='20-day EMA', linewidth=2)
plt.plot(range(smaRange-1, len(prices)), ema_12, 'cyan', label='12-day EMA')
plt.plot(range(len(prices)), ema_4, 'cyan', label='4-day EMA')
#plt.plot(range(len(prices)), ema_26, 'magenta', label='26-day EMA')



smooth_predictions = []
actual_days = []

for day in range(150, len(prices)):
    current_data = prices[:day]
    smooth_pred = smooth_trend_regression(current_data, long_window=200, smooth_window=20)
        
    smooth_predictions.append(smooth_pred)
    actual_days.append(day)

# Create upper and lower bounds
percentage = 0.07  # % bands
upper_bound = np.array(smooth_predictions) * (1 + percentage)
lower_bound = np.array(smooth_predictions) * (1 - percentage)

# Plot everything
plt.plot(range(len(prices)), prices, 'b-', label='Actual Prices', alpha=0.7)
plt.plot(actual_days, smooth_predictions, 'ro-', label='Smooth Predictions', markersize=3)
plt.plot(actual_days, upper_bound, 'g--', label='Upper Bound', alpha=0.7)
plt.plot(actual_days, lower_bound, 'g--', label='Lower Bound', alpha=0.7)

# Optional: Fill the area between bounds
plt.fill_between(actual_days, lower_bound, upper_bound, alpha=0.2, color='green')


plt.xlabel('Day')
plt.ylabel('Price')
plt.title(f'Daily Price Predictions from Day 75 - Instrument {instrument}')
plt.legend()
plt.grid(True)


#Add to plot (usually plotted separately due to 0-100 scale)
plt.figure(1)
plt.figure(figsize=(12, 4))
plt.plot(range(len(prices)), rsi, 'red', label='RSI')
plt.axhline(70, color='gray', linestyle='--', alpha=0.7)  # Overbought
plt.axhline(30, color='gray', linestyle='--', alpha=0.7)  # Oversold
plt.ylabel('RSI')
plt.legend()


#plt.show()


EMAs = [[] for _ in range(len(data))]
for i in range(len(EMAs)):
    EMAs[i] = EMA(data[i])

maxLag = 5
curLag = 5
correlations, p_values = getCorrelations(EMAs,maxLag)

count = 0

plt.figure(figsize=(8, 6))
plt.imshow(correlations[maxLag+curLag], cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title(f'Lag = {curLag}')

curLag = 0
plt.figure(1)
plt.figure(figsize=(8, 6))
plt.imshow(correlations[maxLag+curLag], cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title(f'Lag = {curLag}')
'''
