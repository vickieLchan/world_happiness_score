# Predicting a Country's Happiness Score

This project aims to see if a country's happiness score can be predicted using a linear regression model along with a dataset from the World Happiness Report. The dataset had information for about 160 countries ranging from 2012-2019. The happiness score is measured by the Cantril Ladder on a scale from 0-10. Some of the variables examined for being predictors include the country's GDP, level of social support, life expectancy, freedom, generosity, and corruption. Additionally, I web-scraped Wikipedia for the annual amount of sunshine for each of the countries in the dataset to see if sunshine was a predictor of happiness.


## Getting Started
To examine the datasets, download the 'whr_2019_data' csv file in the repository. 

## Packages Used
* Pandas
* Numpy
* Sklearn
* Matplotlib
* Seaborn
* BeautifulSoup

## Results
The best adjusted R-squared the model produced was 0.73. The most significant coefficients were social support and corruption. With every 1% increase in social support and corruption, there was a 2.5% increase and 1.2% decrease in happiness, respectively.

### Does Sunshine Predict Happiness?
The answer is yes! But not in the way I expected. When I added sunshine as a feature to my model, the adjusted R-square became 0.78. However, sunshine has an inverse relationship with happiness. For every 1% increase in sunshine, there's a 1.2% decrease in happiness.
