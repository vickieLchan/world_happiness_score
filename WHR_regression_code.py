#!/usr/bin/env python
# coding: utf-8

# # World Happiness Linear Regression Project

# In[1]:


import csv
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
import patsy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
get_ipython().magic('matplotlib inline')


# In[2]:


df = pd.read_csv('whr_2019_data.csv')


# ## Data cleaning

# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


#dropping all columns that are not the main variables focused on in the report

drop_df = df.drop(columns=['Standard deviation of ladder by country-year', 
           'Standard deviation/Mean of ladder by country-year', 
           'GINI index (World Bank estimate)', 
           'GINI index (World Bank estimate), average 2000-16',
           'gini of household income reported in Gallup, by wp5-year',
           'Most people can be trusted, Gallup',
           'Most people can be trusted, WVS round 1981-1984',
           'Most people can be trusted, WVS round 1989-1993',
           'Most people can be trusted, WVS round 1994-1998',
           'Most people can be trusted, WVS round 1999-2004',
           'Most people can be trusted, WVS round 2005-2009',
           'Most people can be trusted, WVS round 2010-2014',
           'Confidence in national government',
           'Democratic Quality',
           'Delivery Quality'])
drop_df.head()


# In[6]:


drop_df.columns = drop_df.columns.str.lower()


# In[7]:


drop_df.info()


# In[8]:


renamed_df = drop_df.rename(columns = {'country name': 'country', 'life ladder': 'life_ladder', 
                                       'log gdp per capita': 'gdp_per_capita', 
                                       'social support': 'social_support',
                                       'healthy life expectancy at birth': 'life_expectancy', 
                                       'freedom to make life choices': 'freedom', 
                                       'perceptions of corruption': 'corruption',
                                       'positive affect': 'positive_affect', 
                                       'negative affect': 'negative_affect'})


# In[9]:


renamed_df.head()


# In[10]:


renamed_df[renamed_df.gdp_per_capita.isnull()]


# ### dropping all NaNs

# In[11]:


clean_df = renamed_df.dropna(how = 'any')


# In[12]:


clean_df.info()


# ## Linear Regression with original WHR data set

# In[13]:


lr_df = clean_df.drop(columns = ['country', 'year', 'positive_affect', 'negative_affect'])

#can take out generosity and corruption as well


# In[14]:


lr_df.corr()


# In[15]:


sns.pairplot(lr_df)


# In[16]:


lr_df.head()


# ### Cross validation

# In[17]:


from sklearn.model_selection import KFold

x, y = lr_df.drop('life_ladder', axis=1), lr_df['life_ladder']

kf = KFold(n_splits = 5, shuffle = True, random_state = 987)

x, x_test, y, y_test = train_test_split(x, y, test_size=.2, random_state=41)
x, y = np.array(x), np.array(y)

lr_r2 = []
lr_r2_tr = []

lr_rmse = []
lr_rmse_tr = []


# In[18]:


for tr, val in kf.split(x, y):
    
    lr = LinearRegression()

    x_train, x_val = x[tr], x[val]
    y_train, y_val = y[tr], y[val]
    
    lr.fit(x_train, y_train)   

    lr_r2.append(r2_score(y_val, lr.predict(x_val)))
    lr_r2_tr.append(r2_score(y_train, lr.predict(x_train)))

    lr_rmse.append(np.sqrt(mean_squared_error(y_val, lr.predict(x_val))))
    lr_rmse_tr.append(np.sqrt(mean_squared_error(y_train, lr.predict(x_train))))


# In[19]:


print(lr_r2) 
print(lr_r2_tr)

print(lr_rmse) 
print(lr_rmse_tr) 


# In[20]:


#r2 averages
print(f'Simple mean cv r^2: {np.mean(lr_r2):.3f} +- {np.std(lr_r2):.3f}')
print(f'Simple mean cv r^2: {np.mean(lr_r2_tr):.3f} +- {np.std(lr_r2_tr):.3f}')

#root mean squared error averages
print(f'Simple mean cv rmse: {np.mean(lr_rmse):.3f} +- {np.std(lr_rmse):.3f}')
print(f'Simple mean cv rmse: {np.mean(lr_rmse_tr):.3f} +- {np.std(lr_rmse_tr):.3f}')


# In[21]:


#test
y_pred = lr.predict(x_test)
r2_score(y_test, y_pred)


# In[22]:


pred_y_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

pred_y_df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ### Coefficients and P values

# In[23]:


x = lr_df.drop('life_ladder', axis=1)
y = lr_df['life_ladder']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
print(model.summary())


# ## Visualizations

# In[24]:


clean_df.head()


# In[25]:


#scatter plot with GDP per capita and life ladder
ax1 = sns.regplot(x='gdp_per_capita', y='life_ladder', data=clean_df, scatter_kws={"color": "lightcoral"}, 
                  line_kws={"color": "maroon"})


# In[26]:


#scatter plot with social support and life ladder
ax2 = sns.regplot(x='social_support', y='life_ladder', data=clean_df, scatter_kws={"color": "lightskyblue"}, 
                  line_kws={"color": "maroon"})


# In[27]:


#scatter plot with life expectancy and life ladder
ax3 = sns.regplot(x='corruption', y='life_ladder', data=clean_df, scatter_kws={"color": "plum"}, 
                  line_kws={"color": "maroon"})


# ## Web scraping Wikipedia

# In[28]:


from __future__ import print_function, division
import requests

requests.__path__


# In[29]:


url = 'https://en.wikipedia.org/wiki/List_of_cities_by_sunshine_duration'

response = requests.get(url)


# In[30]:


response.status_code


# In[31]:


page = response.text


# In[32]:


from bs4 import BeautifulSoup

soup = BeautifulSoup(page, "html5")


# In[33]:


# print(soup.prettify())


# ### Table 1 - Africa

# In[34]:


table = soup.find('table', {'class': 'wikitable plainrowheaders sortable'})
# print(table)

links = table.find_all('a')
countries = []

for country in links: 
    countries.append(country.get('title'))
new_list = [country for country in countries if country is not None]

# print(new_list)
new_list = new_list[::2]
len(new_list)

sunshine_df = pd.DataFrame()
sunshine_df['country'] = new_list
sunshine_df.tail(10)
# sunshine_df.shape

rows = table.find_all('tr')
rows = [row for row in rows if ('Country' not in row.text)]
sunshine_hours = []

for i in range(0, len(rows)):
    columns = rows[i].find_all('td')
    sunshine_hours.append(columns[14].get_text().strip())

len(sunshine_hours)
# sunshine_hours

sunshine_df['hours_of_sunshine'] = [i for i in sunshine_hours]
sunshine_df['hours_of_sunshine'] = sunshine_df['hours_of_sunshine'].str.replace(',', '').astype(float)
sunshine_df.head(10)


# In[35]:


africa_s = sunshine_df.groupby(['country']).hours_of_sunshine.mean()
africa_s


# ### Table 2 - Asia

# In[36]:


table2 = soup.find_all('table', {'class': 'wikitable plainrowheaders sortable'})[1]

links2 = table2.find_all('a')
# print(links2)
countries2 = []

for country in links2: 
    countries2.append(country.get('title'))
new_list2 = [country for country in countries2 if country is not None]

new_list2 = new_list2[::2]
# print(new_list2)
# len(new_list2)

sunshine_df2 = pd.DataFrame()
sunshine_df2['country'] = new_list2
# sunshine_df2.head(10)

rows2 = table2.find_all('tr')
rows2 = [row for row in rows2 if ('Country' not in row.text)]
sunshine_hours2 = []

for i in range(0, len(rows2)):
    columns2 = rows2[i].find_all('td')
    sunshine_hours2.append(columns2[14].get_text().strip())
# len(sunshine_hours2)

sunshine_df2['hours_of_sunshine'] = [i for i in sunshine_hours2]
sunshine_df2['hours_of_sunshine'] = sunshine_df2['hours_of_sunshine'].str.replace(',', '').astype(float)
sunshine_df2.head()


# In[37]:


asia_s = sunshine_df2.groupby(['country']).hours_of_sunshine.mean()
asia_s


# ### Table 3 - Europe

# In[38]:


table3 = soup.find_all('table', {'class': 'wikitable plainrowheaders sortable'})[2]

links3 = table3.find_all('a')
# print(links3)
countries3 = []

for country in links3: 
    countries3.append(country.get('title'))
new_list3 = [country for country in countries3 if country is not None]

new_list3 = new_list3[::2]
# print(new_list3)
# len(new_list3)

sunshine_df3 = pd.DataFrame()
sunshine_df3['country'] = new_list3
# sunshine_df3.head(10)

rows3 = table3.find_all('tr')
rows3 = [row for row in rows3 if ('Country' not in row.text)]
sunshine_hours3 = []

for i in range(0, len(rows3)):
    columns3 = rows3[i].find_all('td')
    sunshine_hours3.append(columns3[14].get_text().strip())
# len(sunshine_hours3)

sunshine_df3['hours_of_sunshine'] = [i for i in sunshine_hours3]
sunshine_df3['hours_of_sunshine'] = sunshine_df3['hours_of_sunshine'].str.replace(',', '').astype(float)
sunshine_df3.head()


# In[39]:


europe_s = sunshine_df3.groupby(['country']).hours_of_sunshine.mean()
europe_s


# ### Table 4 - North and Central America

# In[40]:


table4 = soup.find_all('table', {'class': 'wikitable plainrowheaders sortable'})[3]

links4 = table4.find_all('a')
# print(links4)
countries4 = []

for country in links4: 
    countries4.append(country.get('title'))
new_list4 = [country for country in countries4 if country is not None]

new_list4 = new_list4[::2]
# print(new_list4)
# len(new_list4)

sunshine_df4 = pd.DataFrame()
sunshine_df4['country'] = new_list4
# sunshine_df4.head(10)

rows4 = table4.find_all('tr')
rows4 = [row for row in rows4 if ('Country' not in row.text)]
sunshine_hours4 = []

for i in range(0, len(rows4)):
    columns4 = rows4[i].find_all('td')
    sunshine_hours4.append(columns4[14].get_text().strip())
# len(sunshine_hours4)

sunshine_df4['hours_of_sunshine'] = [i for i in sunshine_hours4]
sunshine_df4['hours_of_sunshine'] = sunshine_df4['hours_of_sunshine'].str.replace(',', '').astype(float)
sunshine_df4.head()


# In[41]:


north_central_america_s = sunshine_df4.groupby(['country']).hours_of_sunshine.mean()
north_central_america_s


# ### Table 5 - South America

# In[42]:


table5 = soup.find_all('table', {'class': 'wikitable plainrowheaders sortable'})[4]

links5 = table5.find_all('a')
# print(links5)
countries5 = []

for country in links5: 
    countries5.append(country.get('title'))
new_list5 = [country for country in countries5 if country is not None]

new_list5 = new_list5[::2]
# print(new_list5)
# len(new_list5)

sunshine_df5 = pd.DataFrame()
sunshine_df5['country'] = new_list5
# sunshine_df5.head(10)

rows5 = table5.find_all('tr')
rows5 = [row for row in rows5 if ('Country' not in row.text)]
sunshine_hours5 = []

for i in range(0, len(rows5)):
    columns5 = rows5[i].find_all('td')
    sunshine_hours5.append(columns5[14].get_text().strip())
# len(sunshine_hours5)

sunshine_df5['hours_of_sunshine'] = [i for i in sunshine_hours5]
sunshine_df5['hours_of_sunshine'] = sunshine_df5['hours_of_sunshine'].str.replace(',', '').astype(float)
sunshine_df5.head()


# In[43]:


south_america_s = sunshine_df5.groupby(['country']).hours_of_sunshine.mean()
south_america_s


# ### Table 6 - Oceania

# In[44]:


table6 = soup.find_all('table', {'class': 'wikitable plainrowheaders sortable'})[5]

links6 = table6.find_all('a')
# print(links6)
countries6 = []

for country in links6: 
    countries6.append(country.get('title'))
new_list6 = [country for country in countries6 if country is not None]

new_list6 = new_list6[::2]
# print(new_list6)
# len(new_list6)

sunshine_df6 = pd.DataFrame()
sunshine_df6['country'] = new_list6
# sunshine_df6.head(10)

rows6 = table6.find_all('tr')
rows6 = [row for row in rows6 if ('Country' not in row.text)]
sunshine_hours6 = []

for i in range(0, len(rows6)):
    columns6 = rows6[i].find_all('td')
    sunshine_hours6.append(columns6[14].get_text().strip())
# len(sunshine_hours5)

sunshine_df6['hours_of_sunshine'] = [i for i in sunshine_hours6]
sunshine_df6['hours_of_sunshine'] = sunshine_df6['hours_of_sunshine'].str.replace(',', '').astype(float)
sunshine_df6.head()


# In[45]:


oceania_s = sunshine_df6.groupby(['country']).hours_of_sunshine.mean()
oceania_s


# ### Joining all tables into new DF

# In[46]:


world_s = pd.concat([africa_s, asia_s, europe_s, north_central_america_s, south_america_s, oceania_s], ignore_index=False)
# world_s


# In[47]:


world_s.count()
world_sun_df = world_s.to_frame(name='sunshine')
world_sun_df.head()


# In[48]:


new_df = clean_df.copy()


# In[49]:


new_df['year'].astype('int64')


# In[50]:


new_df.head()


# In[51]:


new_df.info()


# In[52]:


new_df.nunique(axis=0) #157 different countries


# In[53]:


country_df = new_df.loc[new_df.year == 2017].reset_index(drop=True)
#it looks like the year 2017 has the most countries recorded  (133) compared to the other years. 
#So we can pin this against the sunshine data.
country_df.head()


# In[54]:


merged_df = pd.merge(country_df, world_sun_df, on=['country'], how='inner') #inner only returns 108 rows, some of the 
#country names probably don't match exactly. We have to clean it up if we want to salvage more country matches 
#using outer instead of inner. However, I will opt to drop the countries that don't match exactly for the sake of time.
# merged_df.info()
merged_df.head()


# In[55]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
merged_df['sunshine_scaled'] = scaler.fit_transform(merged_df['sunshine'].values.reshape(-1,1))


# In[56]:


merged_df.tail()


# In[57]:


# merged_df.sunshine_scaled.min()
merged_df.sunshine_scaled.max()


# ## Linear regression with sunshine as additional variable

# In[58]:


#clean up the df to only show columns that are signifcant, p-value > 0.05
merged_sun_df = merged_df.drop(columns = ['country', 'year', 'gdp_per_capita', 'corruption', 'positive_affect', 'negative_affect', 'sunshine'])


# In[59]:


merged_sun_df.corr()


# In[60]:


#plot of sunshine vs. life ladder
ax = sns.regplot(x='sunshine_scaled', y='life_ladder', data=merged_sun_df, scatter_kws={"color": "mediumaquamarine"}, 
                  line_kws={"color": "slategrey"})


# ### Cross Validation with sunshine

# In[61]:


from sklearn.model_selection import KFold

sunx, suny = merged_sun_df.drop('life_ladder', axis=1), merged_sun_df['life_ladder']

kf = KFold(n_splits = 10, shuffle = True, random_state = 80)


# In[62]:


sunx, sunx_test, suny, suny_test = train_test_split(sunx, suny, test_size=.2, random_state=88)
sunx, suny = np.array(sunx), np.array(suny)

sunlr_r2 = []
sunlr_r2_tr = []
sunl1_r2 = []
sunl1_r2_tr = []

sunlr_rmse = []
sunlr_rmse_tr = []
sunl1_rmse = []
sunl1_rmse_tr = []


# In[63]:


for tr, val in kf.split(sunx, suny):
    
    lr = LinearRegression()
    l1 = Lasso(alpha=1.8)
    
    sunx_train, sunx_val = sunx[tr], sunx[val]
    suny_train, suny_val = suny[tr], suny[val]
    
    lr.fit(sunx_train, suny_train)   
    l1.fit(sunx_train, suny_train)
    
    sunlr_r2.append(r2_score(suny_val, lr.predict(sunx_val)))
    sunlr_r2_tr.append(r2_score(suny_train, lr.predict(sunx_train)))
    sunl1_r2.append(r2_score(suny_val, l1.predict(sunx_val)))
    sunl1_r2_tr.append(r2_score(suny_train, l1.predict(sunx_train)))
    
    sunlr_rmse.append(np.sqrt(mean_squared_error(suny_val, lr.predict(sunx_val))))
    sunlr_rmse_tr.append(np.sqrt(mean_squared_error(suny_train, lr.predict(sunx_train))))
    sunl1_rmse.append(np.sqrt(mean_squared_error(suny_val, l1.predict(sunx_val))))
    sunl1_rmse_tr.append(np.sqrt(mean_squared_error(suny_train, l1.predict(sunx_train))))


# In[64]:


print(sunlr_r2) 
print(sunlr_r2_tr)
print(sunl1_r2) 
print(sunl1_r2_tr) 

print(sunlr_rmse) 
print(sunlr_rmse_tr) 
print(sunl1_rmse) 
print(sunl1_rmse_tr) 


# In[65]:


#R^2 averages
print(f'Simple mean cv r^2: {np.mean(sunlr_r2):.3f} +- {np.std(sunlr_r2):.3f}')
print(f'Simple mean cv r^2: {np.mean(sunlr_r2_tr):.3f} +- {np.std(sunlr_r2_tr):.3f}')
print(f'Lasso mean cv r^2: {np.mean(sunl1_r2):.3f} +- {np.std(sunl1_r2):.3f}')
print(f'Lasso mean cv r^2: {np.mean(sunl1_r2_tr):.3f} +- {np.std(sunl1_r2_tr):.3f}')

#root mean squared error averages
print(f'Simple mean cv rmse: {np.mean(sunlr_rmse):.3f} +- {np.std(sunlr_rmse):.3f}')
print(f'Simple mean cv rmse: {np.mean(sunlr_rmse_tr):.3f} +- {np.std(sunlr_rmse_tr):.3f}')
print(f'Lasso mean cv rmse: {np.mean(sunl1_rmse):.3f} +- {np.std(sunl1_rmse):.3f}')
print(f'Lasso mean cv rmse: {np.mean(sunl1_rmse_tr):.3f} +- {np.std(sunl1_rmse_tr):.3f}')


# In[66]:


#test
suny_pred = l1.predict(sunx_test)
r2_score(suny_test, suny_pred)


# In[68]:


y_pred = l1.predict(sunx_test)
actualpred_y_df = pd.DataFrame({'Actual': suny_test, 'Predicted': y_pred})
actualpred_y_df.head(10)


# In[69]:


actualpred_y_df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ### Coefficients and P values

# In[70]:


#done in scikit-learn
sunx = sm.add_constant(sunx)
est = sm.OLS(suny, sunx)
est2 = est.fit()
print(est2.summary())


# ### Predicting Y with Made-up X variables

# In[71]:


clean_df.sample(20)


# In[72]:



# wakanda = [[1.00],[0.96],[80.21],[0.85],[0.09],[0.75]]
# y_wakanda = lr.predict(wakanda)
# print("X=%s, Predicted=%s" % (wakanda[0], y_wakanda[0]))


# In[ ]:




