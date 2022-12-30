import os
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn import feature_selection
import geopandas
import country_converter as coco
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#---------------------------------------------------------------------------------------------------------------#
# FAO DATA FOR 2010 - 2019
#---------------------------------------------------------------------------------------------------------------#

# working directory with CSV-files
work_dir = r'PATH_TO_GLOBAL_FOREST_TREND_FOLDER\data'

# note: area is expressed in thousand hectares (000 ha)
fao_countries = 'FAOSTAT_forest_area_by_countries.csv'
path_fao_countries = os.sep.join([work_dir, fao_countries])
fao_countries = pd.read_csv(path_fao_countries)

#---------------------------------------------------------------------------------------------------------------#
# DATA PERPARATION
#---------------------------------------------------------------------------------------------------------------#

# data preparation
labels = ['Domain Code', 'Domain', 'Area Code (M49)', 'Element Code', 'Element', 'Item Code', 'Year Code', 'Flag', 'Flag Description']
fao_countries.drop(columns=labels, inplace=True)

# remove 'China' because of double entry (China, mainland)
fao_countries = fao_countries[fao_countries.Area != 'China']
# remove 'Netherlands Antilles (former)'
fao_countries = fao_countries[fao_countries.Area != 'Netherlands Antilles (former)']
# remove 'Channel Islands' --> no real country with no country code
fao_countries = fao_countries[fao_countries.Area != 'Channel Islands']
# insert interpolated values for South Sudan (2010, 2011)
fao_countries = fao_countries.append({'Area': 'South Sudan', 'Item': 'Forest land', 'Year': 2010, 'Unit': '1000 ha', 'Value': 7157.0}, ignore_index=True)
fao_countries = fao_countries.append({'Area': 'South Sudan', 'Item': 'Forest land', 'Year': 2011, 'Unit': '1000 ha', 'Value': 7157.0}, ignore_index=True)
# subtract value of South Sudan of value for Sudan (former) in order to have consistent data
south_sudan_forest = fao_countries.at[2323,'Value']
fao_countries.loc[(fao_countries.Area == 'Sudan (former)') & (fao_countries.Year < 2013), 'Value'] = \
(fao_countries[(fao_countries.Area == 'Sudan (former)') & (fao_countries.Year < 2013)]['Value'] - south_sudan_forest)


fao_countries.sort_values(by=['Area', 'Year'], inplace=True)
fao_countries = fao_countries.reset_index(drop=True)


# assigning ISO3 country codes
cc = coco.CountryConverter()
countries = fao_countries['Area'] 
fao_countries['ISO3'] = cc.convert(names=countries, to='ISO3')
# change ISO3 of French Guyana (falsely assigned to GUY)
fao_countries.loc[fao_countries.Area == 'French Guyana', 'ISO3'] = 'GUF'

# calculating each country's share of worlds total forest area (for each year)
years = []
for i in range(2010,2020):
    years.append(i)

for year in years:
    total_forest = fao_countries[fao_countries.Year == year].Value.sum()
    fao_countries.loc[fao_countries.Year == year, 'worlds_forest_share[%]'] = (fao_countries.Value / total_forest) * 100

#---------------------------------------------------------------------------------------------------------------#
# APPENDING FAO DATA FOR 2020
#---------------------------------------------------------------------------------------------------------------#

# forest data for 2020
fra2020 = 'fra2020_extentOfForest.csv'
path_fra2020 = os.sep.join([work_dir, fra2020])
fra2020 = pd.read_csv(path_fra2020, skiprows=1)

# unit = 1000 ha
fra2020 = fra2020.loc[:235]
fra2020 = fra2020.rename({'Unnamed: 0' : 'Area', '2020' : 'Value'}, axis=1)
fra2020['Year'] = 2020
fra2020['Item'] = 'Forest land'
fra2020['Unit'] = '1000 ha'
fra2020['Area'] = fra2020['Area'].map(lambda x: x[:-13] if x[-13:] == ' (Desk study)' else x)

# assigning ISO3 country codes
cc = coco.CountryConverter()
countries = fra2020['Area'] 
fra2020['ISO3'] = cc.convert(names=countries, to='ISO3')
# change ISO3 of French Guyana (falsely assigned to GUY)
fra2020.loc[fra2020.Area == 'French Guyana', 'ISO3'] = 'GUF'

# calculating each country's share of worlds forest area for 2020
global_forest_area = fra2020.copy()
global_forest_area = global_forest_area.Value.sum()
fra2020['worlds_forest_share[%]'] = (fra2020['Value'] / global_forest_area) * 100

# append on 2010-2019 data
fao_countries = fao_countries.append(fra2020)
fao_countries.sort_values(by=['Area', 'Year'], inplace=True)
fao_countries = fao_countries.reset_index(drop=True)

# drop uncomplete data
fao_countries = fao_countries[fao_countries.Area != 'Svalbard and Jan Mayen Islands']
fao_countries = fao_countries[fao_countries.Area != 'Saint Helena']
fao_countries = fao_countries[fao_countries.Area != 'Guernsey']
fao_countries = fao_countries[fao_countries.Area != 'Jersey']

# calculate forest area in different units
fao_countries['forest_area[km^2]'] = fao_countries['Value'] * 10
fao_countries['forest_area[ha]'] = fao_countries['Value'] * 1000

#---------------------------------------------------------------------------------------------------------------#
# DATA OF COUNTRY SIZES
#---------------------------------------------------------------------------------------------------------------#

# data of country sizes
fra_country_size = 'fra2012-2020_total_land_area.csv'
path_fra_country_size = os.sep.join([work_dir, fra_country_size])
fra_country_size = pd.read_csv(path_fra_country_size, skiprows=1)
fra_country_size = fra_country_size.rename({'Unnamed: 0' : 'Area', '2020' : 'total_land_area[1000ha]'}, axis=1)
fra_country_size = fra_country_size.loc[:235]
# assigning ISO3 country codes
cc = coco.CountryConverter()
countries = fra_country_size['Area'] 
fra_country_size['ISO3'] = cc.convert(names=countries, to='ISO3')
# change ISO3 of French Guyana (falsely assigned to GUY)
fra_country_size.loc[fra_country_size.Area == 'French Guyana', 'ISO3'] = 'GUF'
# drop not necessary columns
fra_country_size.drop(columns='Area', inplace=True)

# merge with fao_countries (forest data)
fao_countries = fao_countries.merge(fra_country_size, how='left', on='ISO3')

# calculate share of total land covered by forest for each country
fao_countries['country_forest_share[%]'] = (fao_countries['Value'] / fao_countries['total_land_area[1000ha]']) * 100

#---------------------------------------------------------------------------------------------------------------#
# TREND ANALYSIS
#---------------------------------------------------------------------------------------------------------------#

# global trend analysis
X = fao_countries.Year.unique()

# calculating global forest extent for each year
years.append(2020)
y = np.array([])
global_forest_list = []
for year in years:
    global_forest_sum = fao_countries[fao_countries.Year == year]['Value'].sum()
    global_forest_list.append(global_forest_sum)
y = np.array(global_forest_list)


slope, intercept, r, p, se = scipy.stats.linregress(X, y)
print('slope: ', slope,'\nintercept: ', intercept, '\nr_squared: ', np.power(abs(r), 2), '\np-value: ', p, '\nstandard error: ', se)






# regression analysis for each country
np.seterr(divide='ignore', invalid='ignore')
ISO3 = []
trend = []
r_squared = []
p_value = []
fao_countries_grouped = fao_countries.groupby(['ISO3'])

for group in fao_countries_grouped.groups.keys():
    country_df = fao_countries_grouped.get_group(group)
    X = np.array(country_df[['Year']]).reshape(-1, 1) # series does not have reshape function, thus you need to convert to array
    y = np.array(country_df.Value)
    model = LinearRegression()  # <--- this does not accept (X, y)
    results = model.fit(X, y)
    f_results = feature_selection.f_regression(X, y)
    r_sq = model.score(X, y)
    ISO3.append(group)
    trend.append(results.coef_)
    r_squared.append(r_sq)
    p_value.append(f_results[1])


dict = {'ISO3': ISO3, 'regress_coeff': trend, 'r_squared': r_squared, 'p_value': p_value} 
trends = pd.DataFrame(dict)
trends['regress_coeff'] = trends['regress_coeff'].astype(float)
trends['r_squared'] = trends['r_squared'].astype(float)
trends['p_value'] = trends['p_value'].astype(float)

# merge with fao_countries --> slope of regression line is assigned to every row for each country
fao_countries = fao_countries.merge(trends, how='left', on='ISO3')
# country relative trend
fao_countries['country_relative_trend'] = (fao_countries['regress_coeff'] / fao_countries['Value']) * 100

#---------------------------------------------------------------------------------------------------------------#
# MERGE WITH GEOPANDAS WORLD MAP DATASET
#---------------------------------------------------------------------------------------------------------------#

# 'naturalearth_lowres' is geopandas world dataset including geometries
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# drop Antarctica because of no available forest data
world = world.drop(index=159)

# France and Norway do not have ISO3 country codes in geopandas --> assign FRA and NOR
# (for checking: print(world[world.iso_a3 == '-99']))
world.loc[world.name == 'France', 'iso_a3'] = 'FRA'
world.loc[world.name == 'Norway', 'iso_a3'] = 'NOR'

# drop not necessary columns
labels = ['pop_est', 'name', 'gdp_md_est']
world.drop(columns=labels, inplace=True)

# merge FAO forest data with world data
world_fao = world.merge(fao_countries, how="left", left_on=['iso_a3'], right_on=['ISO3'])
# drop NaNs from geopandas world dataset
world_fao.dropna(subset=['Year'], inplace=True)
world_fao = world_fao.sort_values(by=['Area', 'Year'])
world_fao = world_fao.reset_index(drop=True)

#---------------------------------------------------------------------------------------------------------------#
# COIUNTRY SIZE CALCULATION (NOT USED DUE TO EXISTING DATA FOUND AFTER IMPLEMENTATION)
#---------------------------------------------------------------------------------------------------------------#

"""
# country area calculation
# commented out because existing FAO data with size information was found after implementation

# calculate area from geopandas world map
area_calc = world_fao.copy()
# area_calc = area_calc.to_crs({'proj':'cea'})  # 'cea' = Equal Area Cylindrical ; other option would be EPSG: 6933 --> next line
# reference for 6933: https://www.mdpi.com/2220-9964/1/1/32/htm
area_calc = area_calc.to_crs(epsg=6933)
area_calc["area"] = area_calc['geometry'].area / 10**6  # convert to square kilometers
area = area_calc['area']
world_fao['land_area_calc[km^2]'] = area
world_fao['land_area_calc[ha]'] = world_fao['land_area_calc[km^2]'] * 100
world_fao = world_fao.reset_index(drop=True)

"""

#---------------------------------------------------------------------------------------------------------------#
# GLOBAL TREND AND WORLD MAP PLOTS
#---------------------------------------------------------------------------------------------------------------#

### plot global trend ###
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(X, y, 'o', label='global forest area')
ax.plot(X, intercept + slope*X, 'r', label='fitted line')
plt.legend()
plt.title('Trend in global forest area')
plt.xlabel('Jahr')
plt.ylabel('globale Waldfläche [1000 ha]')
plt.rcParams.update({'font.size': 12})
plt.text(2018, 4098000, "R\N{SUPERSCRIPT TWO} = " + str(round(np.power(abs(r), 2), 3)))
plt.ylim([4050000, 4110000])
#plt.savefig(os.sep.join([work_dir, '2010_2020_trend_in_forest_area_regression_line']), facecolor='white', bbox_inches='tight', pad_inches=0.05, dpi=600)





### forest distribution ###

# 2020: Share of world's land area covered by forest
world_fao_20 = world_fao[world_fao.Year == 2020].copy()

def addColorbar():
    fig, ax = plt.subplots(1, 1, figsize=(25, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad="0.5%")
    return ax, cax
# create new colormap (red to green with classes)
greens_cmap = cm.get_cmap('YlGn', 16)
ax, cax = addColorbar()
world_fao_20["geometry"].boundary.plot(ax=ax, figsize=(25, 20), edgecolor='black', linewidth=0.6)
world_fao_20.plot(column="worlds_forest_share[%]",
               ax=ax, 
               cax=cax, 
               cmap=greens_cmap, vmin=0, vmax=15,
               legend=True,
               legend_kwds={"label": "share of world's land area covered by forest [%]"})
ax.set_title("2020: Share of world's land area covered by forest", fontsize=24)
ax.set_xlabel("longitude [°]")
ax.set_ylabel("latitude [°]")
#plt.savefig(os.sep.join([work_dir, '2020_share_of_worlds_land_area_covered_by_forest']), facecolor='white', bbox_inches='tight', pad_inches=0.05, dpi=600) # use format=eps for latex document





# 2020: Share of total land area covered by forest

ax, cax = addColorbar()
world_fao_20["geometry"].boundary.plot(ax=ax, figsize=(25, 20), edgecolor='black', linewidth=0.6)
world_fao_20.plot(column="country_forest_share[%]", 
               ax=ax, 
               cax=cax, 
               cmap='YlGn', vmin=0, vmax=100,
               legend=True,
               legend_kwds={"label": "share of total land area covered by forest [%]"})
ax.set_title("2020: Share of total land area covered by forest", fontsize=24)
ax.set_xlabel("longitude [°]")
ax.set_ylabel("latitude [°]")
#plt.savefig(os.sep.join([work_dir, '2020_share_of_total_land_area_covered_by_forest']), facecolor='white', bbox_inches='tight', pad_inches=0.05, dpi=600) # use format=eps for latex document





### trend distribution ###
# absolute trends
world_fao_20 = world_fao[world_fao.Year == 2020].copy()

plt.rcParams.update({'font.size': 24})
def addColorbar():
    fig, ax = plt.subplots(1, 1, figsize=(25, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad="0.5%")
    return ax, cax
ax, cax = addColorbar()
# create new colormap (red to green with classes)
trend_cmap = cm.get_cmap('seismic_r', 16)
world_fao_20["geometry"].boundary.plot(ax=ax, figsize=(25, 20), edgecolor='black', linewidth=0.6)
world_fao_20.plot(column="regress_coeff",
               ax=ax, 
               cax=cax, 
               cmap=trend_cmap, vmax=1000, vmin=-1000,         # seismic_r, coolwarm_r, bwr_r, RdBu, PRGn
               legend=True,
               legend_kwds={"label": "change in forest area \nfor a one-unit shift (one year) [1000 ha]"})
ax.set_title("2010 - 2020: Trends in forest area by country", fontsize=34)
ax.set_xlabel("longitude [°]")
ax.set_ylabel("latitude [°]")
#plt.savefig(os.sep.join([work_dir, '2010_2020_trend_in_forest_area_new']), facecolor='white', bbox_inches='tight', pad_inches=0.05, dpi=600) # use format=eps for latex document





# trends relative to country's forest extent
world_fao_20 = world_fao[world_fao.Year == 2020].copy()
plt.rcParams.update({'font.size': 24})
def addColorbar():
    fig, ax = plt.subplots(1, 1, figsize=(25, 20))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad="0.5%")
    return ax, cax
ax, cax = addColorbar()
# create new colormap (red to green with classes)
trend_cmap = cm.get_cmap('seismic_r', 16)
world_fao_20["geometry"].boundary.plot(ax=ax, figsize=(25, 20), edgecolor='black', linewidth=0.6)
world_fao_20.plot(column="country_relative_trend",
               ax=ax, 
               cax=cax, 
               cmap=trend_cmap, vmin=-4, vmax=4,       # seismic_r, coolwarm_r, bwr_r, RdBu, PRGn
               legend=True,
               legend_kwds={"label": "change relative to countries 2020 forest area \nfor a one-unit shift (one year) [%]"})
ax.set_title("2010 - 2020: Trends in forest area relative to each countries 2020 forest extent", fontsize=34)
ax.set_xlabel("longitude [°]")
ax.set_ylabel("latitude [°]")
#plt.savefig(os.sep.join([work_dir, '2010_2020_country_relative_trend_in_forest_area']), facecolor='white', bbox_inches='tight', pad_inches=0.05, dpi=600) # use format=eps for latex document
