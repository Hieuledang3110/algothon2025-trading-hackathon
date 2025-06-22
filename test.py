#Eric is using this for testing

from matplotlib import pyplot as plt
import pandas 
import numpy as np

data = np.loadtxt("prices.txt")
data = np.rot90(data)
days = list(range(1,len(data[0])+1))

#average/median price of all stocks on a day
averages = np.mean(data, axis = 0)
temp = data # want the median to be the middle stock so im making it odd
medians = np.median(temp, axis = 0)

for i in range(20,40):
    plt.plot(days,data[i])


plt.plot(days,averages, color="red", linewidth=5)
plt.plot(days,medians, color="black", linewidth=5)

plt.xlim(0,100)
# plt.show()


#Extremely simple mean reversion just to test things out. No shorting for now.
def getMyPosition(data):
    # data = np.rot90(data)
    positions = []

    maxDollarPosition = 10000

    for prices in data:
        iLast = len(prices)

        midTermLength = 89
        shortTermLength = 29

        upTrendTolerance = 1.05
        downTrendTolerance = 0.941
        

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
        isTrendingUp = False
        isTrendingDown = False

        if iLast < midTermLength:
            historicalPrice = prices[0]
        else:
            historicalPrice = prices[-midTermLength]
        if currentPrice < historicalPrice*downTrendTolerance:
            isTrendingDown = True
        elif currentPrice > historicalPrice*upTrendTolerance:
            isTrendingUp = True
            
        #Checking how far away the current price is from the short term average
        devianceFactor = (currentPrice-averagePrice)/maxVariation
        devianceFactor = max(min(devianceFactor,1),-1)  #clamp between -1,1

        #It seems that full sending when buying is the most profitable bruh. This might be partially due to the savings from comissions
        if (devianceFactor > 0.329):
            if (devianceFactor < 0.541):
                buyFactor = (devianceFactor+1)/0.54
                buyFactor = 1
        #It seems that being cautious with shorting is important, possibly because potential for downside is greater than potential for upside
        #Here, we short if the stock is moderately down AND is down since midTermLength days ago.
        if (devianceFactor < -0.145 and isTrendingDown):
            if (devianceFactor > -0.52):
                buyFactor = -(devianceFactor-0.38)/0.52
        
        buyAmount = maxDollarPosition*buyFactor
        sharePosition = buyAmount/currentPrice

        positions.append(int(sharePosition))
        final_positions = np.array([int(pos) for pos in positions])
    
    return positions

newData = np.loadtxt("prices.txt")
print(getMyPosition(newData))   
