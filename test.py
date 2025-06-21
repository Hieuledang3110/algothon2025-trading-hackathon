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

    #As a greedy I should disable this strategy if I think price movement is too extreme

    for prices in data:
        iLast = len(prices)

        #Getting the average price during the last 30 days
        if iLast < 29:
            recentPrices = data
        else:
            recentPrices = data[:29]
        averagePrice = np.average(recentPrices)

        #Really shitty approximation of volatility
        maxPrice = np.max(recentPrices)
        minPrice = np.min(recentPrices)
        maxVariation = max(np.abs(float(averagePrice-minPrice)), np.abs(float(maxPrice-averagePrice)))

        #Buying if, based on average price, the stock is overvalued. 
        currentPrice = prices[-1]
        buyFactor = 0
        
        devianceFactor = (currentPrice-averagePrice)/maxVariation
        devianceFactor = max(min(devianceFactor,1),-1)  #clamp between -1,1
        if (devianceFactor > 0.33):
            if (devianceFactor < 0.54):
                buyFactor = (devianceFactor-0.2)/0.55
        # if (devianceFactor < -0.4):
        #     if (devianceFactor > -0.6):
        #         buyFactor = (devianceFactor-0.2)/0.55
        
        buyAmount = maxDollarPosition*buyFactor
        sharePosition = buyAmount/currentPrice

        positions.append(int(sharePosition))
        final_positions = np.array([int(pos) for pos in positions])
    
    return positions

newData = np.loadtxt("prices.txt")
print(getMyPosition(newData))   
