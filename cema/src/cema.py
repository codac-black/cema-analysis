#!/usr/bin/env python
# coding: utf-8

# # Task 1
# You are provided with a dataset from the World Health Organization (WHO) Global Observatory, containing data on people living with HIV at the country level from 2000 to 2023.
# 
# Using this dataset, we would like you to:
# 
# - Create a visualization that shows the trend of HIV cases in the countries that contribute to 75% of the global burden
# - Generate a visualization that displays the trend of HIV cases in the countries contributing to 75% of the burden within each WHO region (column called ParentLocationCode contains the WHO regions)
# - You have also been provided with World Bank data on the multidimensional poverty headcount ratio, which includes factors such as income, educational attainment, school enrolment, electricity access, sanitation and drinking water.
# 
# We would like you to merge this dataset with the HIV data above and analyze the relationship between people living with HIV and multidimensional poverty, and the individual factors that contribute to the ratio. Remember to account for the random effects (country, year).
# 
# Write a paragraph on your findings.

# In[ ]:


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os


# In[2]:


plt.style.use('seaborn-v0_8-darkgrid')


# In[3]:


hiv_data = pd.read_csv('HIV data 2000-2023.csv', encoding='latin1')


# In[4]:


poverty_data = pd.read_excel('multidimensional_poverty.xlsx')


# In[5]:


hiv_data.head()


# In[6]:


hiv_data.isna().sum()


# In[7]:


poverty_data.head()


# In[8]:


poverty_data.isna().sum()


# In[9]:


hiv_data['Value_cleaned'] = hiv_data['Value'].astype(str)
hiv_data['Value_cleaned'] = hiv_data['Value_cleaned'].str.split('[').str[0]
hiv_data['Value_cleaned'] = hiv_data['Value_cleaned'].str.replace(' ', '').str.strip()
hiv_data['Value_numeric'] = pd.to_numeric(hiv_data['Value_cleaned'], errors='coerce')


# In[10]:


hiv_data.isna().sum()


# In[11]:


# Group the HIV data by 'Location' and 'Period' to calculate the total 'Value_numeric' for each combination
hiv_by_country = hiv_data.groupby(['Location', 'Period'])['Value_numeric'].sum().reset_index()


# In[ ]:


global_total = hiv_by_country.groupby('Period')['Value_numeric'].sum().reset_index()
global_total.columns = ['Year', 'global_cases']


# In[13]:


hiv_by_country['Period'] = hiv_by_country['Period'].astype(int)
global_total['Year'] = global_total['Year'].astype(int)


# In[14]:


hiv_by_country.head()


# In[15]:


global_total.head()


# In[16]:


country_contribution = hiv_by_country.merge(
    global_total, 
    left_on='Period',
    right_on='Year' ,
    how='left'
    ) 


# In[17]:


country_contribution = country_contribution.drop('Period',axis=1)


# In[18]:


country_contribution.head()


# In[19]:


country_contribution['contribution'] = np.where(
        country_contribution['global_cases'] > 0,
        country_contribution['Value_numeric'] / country_contribution['global_cases'],
        0 # Assign 0 contribution if global cases is 0
    )

country_contribution.sample(10)


# In[20]:


# Sum contributions across all years for each country
country_total_contribution = country_contribution.groupby('Location')['contribution'].sum().reset_index()
country_total_contribution = country_total_contribution.sort_values('contribution', ascending=False)


# In[21]:


country_total_contribution.head(10)


# In[22]:


# Identify countries contributing to 75% of the total summed contribution
total_summed_contribution = country_total_contribution['contribution'].sum()
if total_summed_contribution > 0:
        country_total_contribution['cumulative_percentage'] = (country_total_contribution['contribution'].cumsum() / total_summed_contribution)
        top_countries = country_total_contribution[country_total_contribution['cumulative_percentage'] <= 0.75]['Location'].tolist()
        # Include the next country if the cumulative sum is just below 0.75
        if country_total_contribution['cumulative_percentage'].iloc[len(top_countries)-1] < 0.75 and len(country_total_contribution) > len(top_countries):
            top_countries.append(country_total_contribution['Location'].iloc[len(top_countries)])
else:
    top_countries = [] # If total contribution is zero, no countries can be identified

if top_countries:
    print(f"\nCountries cumulatively contributing to ~75% of total HIV burden contribution: {', '.join(top_countries)}")
else:
    print("\nCould not determine top contributing countries (total contribution might be zero).")


# In[23]:


# Plot HIV trends for top contributing countries
if top_countries:
    plt.figure(figsize=(14, 7)) # Increased figure size
    for country in top_countries:
        country_data = hiv_by_country[hiv_by_country['Location'] == country]
        if not country_data.empty:
                plt.plot(country_data['Period'], country_data['Value_numeric'], label=country, marker='o', linestyle='-')

    plt.title('HIV Cases in Top Contributing Countries (~75% of Total Burden Contribution)')
    plt.xlabel('Year')
    plt.ylabel('Number of Cases')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Countries") # Added title to legend
    plt.grid(True) # Add grid
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlap
    plt.show() # Show the plot
    plt.savefig('hiv_top_countries.png', bbox_inches='tight') # Ensure legend is saved
    plt.close()
    print("Saved plot: hiv_top_countries.png")
else:
    print("Skipping plot for top countries as none were identified.")


# In[ ]:


if top_countries:
    plot_data = hiv_by_country[hiv_by_country['Location'].isin(top_countries)].copy() 
  
    plt.figure(figsize=(14, 7)) # Increased figure size
    fig = px.bar(
        plot_data,
        x='Period',
        y='Value_numeric',
        color='Location',
        title='HIV Cases in Top Contributing Countries (~75% of Total Burden Contribution)',
        labels={'Period': 'Year', 'Value_numeric': 'Number of Cases', 'Location': 'Country'},
        barmode='group'
    )

    fig.update_layout(
        title_x=0.5,  # Center the title
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Countries"
        ),
        width=1400,
        height=500,
        xaxis=dict(type='category')
    )
    fig.show()

    try:
        pio.write_image(fig, 'hiv_top_countries_bar_plotly.png', scale=2) # Use scale=2 for higher resolution
        print("Saved plot: hiv_top_countries_bar_plotly.png")
    except ValueError as e:
        print(f"Could not save plot image. Ensure 'kaleido' is installed (`pip install -U kaleido`). Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the plot: {e}")


else:
    print("Skipping plot for top countries as none were identified.")


# In[ ]:


# Plot HIV trends for top contributing countries
if top_countries:
    fig = go.Figure()

    for country in top_countries:
        country_data = hiv_by_country[hiv_by_country['Location'] == country]
        if not country_data.empty:
            fig.add_trace(go.Scatter(
                x=country_data['Period'],
                y=country_data['Value_numeric'],
                mode='lines+markers',
                name=country
            ))

    # Update layout for better visualization
    fig.update_layout(
        title='HIV Cases in Top Contributing Countries (~75% of Total Burden Contribution)',
        xaxis_title='Year',
        yaxis_title='Number of Cases',
        legend_title='Countries',
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        template='plotly_white',
        width=1400,
        height=500,
        xaxis=dict(type='category')
    )

    # Show the plot
    fig.show()

    # Save the plot as an HTML file
    fig.write_html("hiv_top_countries.html")
    print("Saved plot: hiv_top_countries.html")
else:
    print("Skipping plot for top countries as none were identified.")


# In[26]:


# Plot HIV trends by WHO region
hiv_by_region = hiv_data.groupby(['ParentLocationCode', 'Period'])['Value_numeric'].sum().reset_index()

plt.figure(figsize=(12, 6))
for region in hiv_by_region['ParentLocationCode'].unique():
    region_data = hiv_by_region[hiv_by_region['ParentLocationCode'] == region]
    plt.plot(region_data['Period'], region_data['Value_numeric'], label=region, marker='o')

plt.title('HIV Cases by WHO Region')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig('hiv_by_region.png')
plt.close()

# HIV and Poverty Relationship Analysis
print("\nAnalyzing relationship between HIV and poverty...")


# In[ ]:


poverty_data['Reporting year',] = pd.to_numeric(poverty_data['Reporting year'], errors='coerce').astype('Int64')


# In[28]:


combined_data = hiv_data.merge(
    poverty_data,
    left_on=['SpatialDimValueCode', 'Period'],
    right_on=['Country code','Reporting year'],
    how='left'
)
combined_data.head()


# In[29]:


combined_data.isna().sum()


# In[30]:


combined_data.size


# In[31]:


combined_data.isna().sum()


# In[32]:


print(combined_data[['Value_numeric', 'Multidimensional poverty headcount ratio (%)']].isna().sum())
print(combined_data[['Value_numeric', 'Multidimensional poverty headcount ratio (%)']].describe())


# In[33]:


combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[59]:


combined_data = combined_data.dropna(subset=['Value_numeric', 'Multidimensional poverty headcount ratio (%)'])


# In[60]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
combined_data['Value_numeric_scaled'] = scaler.fit_transform(combined_data[['Value_numeric']])
combined_data['Poverty_ratio_scaled'] = scaler.fit_transform(combined_data[['Multidimensional poverty headcount ratio (%)']])


# In[61]:


# Use the correct column name for poverty index
model = MixedLM(
    combined_data['Value_numeric_scaled'], 
    combined_data[['Poverty_ratio_scaled']], 
    groups=combined_data['Location']
)
result = model.fit()
print(result.summary())


# In[62]:


from statsmodels.regression.linear_model import OLS

model = OLS(
    combined_data['Value_numeric_scaled'], 
    combined_data[['Poverty_ratio_scaled']]
)
result = model.fit()
print(result.summary())


# In[63]:


# Residual analysis
residuals = result.resid
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=20, color='blue')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of fitted values vs residuals
fitted_values = result.fittedvalues
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.7, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title('Fitted Values vs Residuals')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Plot the relationship
plt.figure(figsize=(10, 6))
sns.regplot(x=combined_data['Poverty_ratio_scaled'], 
            y=combined_data['Value_numeric_scaled'], 
            line_kws={"color": "red"})
plt.title('Relationship Between Poverty Ratio and HIV Cases')
plt.xlabel('Poverty Ratio (Scaled)')
plt.ylabel('HIV Cases (Scaled)')
plt.show()


# In[ ]:


# Residuals Distribution
fig = px.histogram(
    x=residuals,
    nbins=20,
    title="Residuals Distribution",
    labels={"x": "Residuals", "y": "Frequency"},
    marginal="box",
    opacity=0.7
)
fig.update_traces(marker_color="blue")
fig.update_layout(
    xaxis_title="Residuals",
    yaxis_title="Frequency",
    template="plotly_white"
)
fig.show()

# Scatter plot of fitted values vs residuals
fig = px.scatter(
    x=fitted_values,
    y=residuals,
    title="Fitted Values vs Residuals",
    labels={"x": "Fitted Values", "y": "Residuals"},
    opacity=0.7
)
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.update_traces(marker_color="green")
fig.update_layout(
    xaxis_title="Fitted Values",
    yaxis_title="Residuals",
    template="plotly_white"
)
fig.show()

# Relationship Between Poverty Ratio and HIV Cases
fig = px.scatter(
    x=combined_data['Poverty_ratio_scaled'],
    y=combined_data['Value_numeric_scaled'],
    title="Relationship Between Poverty Ratio and HIV Cases",
    labels={"x": "Poverty Ratio (Scaled)", "y": "HIV Cases (Scaled)"},
    opacity=0.7
)
fig.add_trace(
    go.Scatter(
        x=combined_data['Poverty_ratio_scaled'],
        y=result.predict(),
        mode="lines",
        line=dict(color="red"),
        name="Regression Line"
    )
)
fig.update_layout(
    xaxis_title="Poverty Ratio (Scaled)",
    yaxis_title="HIV Cases (Scaled)",
    template="plotly_white"
)
fig.show()


# In[78]:


combined_data.head()


# combined_data.columns

# # Question 2
# You have been provided with data on the under-five mortality rate and neonatal mortality rate for the African region, which has been downloaded from the UN Inter-agency Group for Child Mortality Estimation. Your task is to:
# 
# - Filter data for the eight countries belonging to the East African Community (list here: https://www.eac.int/overview-of-eac)
# - Visualize the latest estimate of each indicator at the country level using shapefiles, which can be downloaded from www.gadm.org.
# - Show the average trends in the mortality rates over time (plot the average trend line and add the points in the graphic for the country level estimates for each indicator. Expectation: two plots).
# - Based on your visualizations, identify the countries with the highest under-five mortality rates in East Africa and the highest neonatal mortality.

# In[ ]:


# load the dataset
df = pd.read_csv('dataset_datascience.csv')


# In[68]:


df.head()


# In[69]:


df.isna().sum()


# In[70]:


# drop columns with all NaN values
df = df.dropna(axis=1, how='all')


# In[71]:


df['Reference Date'] = df['Reference Date'].astype(str)

df['Year'] = df['Reference Date'].str.split('.').str[0].astype(int)

df[['Reference Date', 'Year']].head()


# In[72]:


df.Indicator.unique()


# In[73]:


def read_adm0_shapefiles(shapefiles_dir):
    """
    Reads administrative level 0 shapefiles and combines them into a single GeoDataFrame
    """
    adm0_shapefiles = []
    for file in os.listdir(shapefiles_dir):
        if file.endswith("_0.shp"):
            filepath = os.path.join(shapefiles_dir, file)
            gdf = gpd.read_file(filepath)
            country_code = file.split('_')[1]
            gdf['ISO3'] = country_code
            adm0_shapefiles.append(gdf)
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(adm0_shapefiles, ignore_index=True))
    return combined_gdf

# Load shapefiles
shapefiles_dir = "shapefiles"
adm0_gdf = read_adm0_shapefiles(shapefiles_dir)

# Ensure proper CRS (WGS 84)
adm0_gdf = adm0_gdf.to_crs(epsg=4326)



# In[74]:


# EAC countries and their ISO codes
eac_countries = {
    "BDI": "Burundi",
    "KEN": "Kenya",
    "RWA": "Rwanda",
    "SSD": "South Sudan",
    "TZA": "Tanzania",
    "UGA": "Uganda",
    "COD": "Democratic Republic of the Congo",
    "SOM": "Somalia"
}

# Filter shapefile data for EAC countries
eac_shapefiles = adm0_gdf[adm0_gdf['ISO3'].isin(eac_countries.keys())]

# Filter mortality data for EAC countries
eac_mortality = df[df['Geographic area'].isin(eac_countries.values())]


# In[75]:


def create_mortality_map(data, indicator, year, title):
    """
    Creates a choropleth map for the specified mortality indicator and year with country borders
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Filter data for the specific indicator and year
    plot_data = data[
        (data['Indicator'] == indicator) & 
        (data['Year'] == year)
    ]
    
    # Merge with shapefile data
    merged = eac_shapefiles.merge(
        plot_data, 
        left_on='ISO3', 
        right_on='REF_AREA'
    )
    
    # Create choropleth map
    merged.plot(
        column='Observation Value',
        ax=ax,
        legend=True,
        legend_kwds={'label': 'Deaths per 1,000 live births'},
        cmap='YlOrRd',
        edgecolor='black',  # Add black borders
        linewidth=1         # Set border width
    )
    
    # Add country labels
    for idx, row in merged.iterrows():
        # Get centroid for label placement
        centroid = row.geometry.centroid
        ax.annotate(
            row['ISO3'],  # Using ISO3 code as label
            xy=(centroid.x, centroid.y),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=8,
            fontweight='bold',
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
    
    # Add title and remove axes
    ax.set_title(f"{title} ({year})", pad=20, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add background color
    ax.set_facecolor('lightblue')  # Light blue background for water bodies
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Create maps for both indicators
neonatal_map = create_mortality_map(
    eac_mortality,
    'Neonatal mortality rate',
    2020,
    'Neonatal Mortality Rate in East Africa'
)

under_five_map = create_mortality_map(
    eac_mortality,
    'Under-five mortality rate',
    2020,
    'Under-Five Mortality Rate in East Africa'
)

# Save the maps
# neonatal_map.savefig('neonatal_mortality_map.png', dpi=300, bbox_inches='tight')
# under_five_map.savefig('under_five_mortality_map.png', dpi=300, bbox_inches='tight')

# plt.close('all')  # Close all figures to free memory


# In[76]:


def create_mortality_map_plotly(data, indicator, year, title):
    """
    Creates an interactive choropleth map using Plotly for the specified mortality indicator and year
    """
    # Filter data for the specific indicator and year
    plot_data = data[
        (data['Indicator'] == indicator) & 
        (data['Year'] == year)
    ]
    
    # Merge with shapefile data to get geometry
    merged = eac_shapefiles.merge(
        plot_data, 
        left_on='ISO3', 
        right_on='REF_AREA'
    )
    
    # Create a GeoJSON-like dictionary for Plotly
    geojson = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    # Map for country names to use in hover text
    country_names = {}
    for idx, row in merged.iterrows():
        geojson['features'].append({
            'type': 'Feature',
            'id': row['ISO3'],
            'properties': {'name': row['ISO3']},
            'geometry': row['geometry'].__geo_interface__
        })
        country_names[row['ISO3']] = row['ISO3']  # You can replace with full country name if available
    
    # Create choropleth map
    fig = px.choropleth(
        plot_data,
        geojson=geojson,
        locations='REF_AREA',
        featureidkey='id',
        color='Observation Value',
        color_continuous_scale='YlOrRd',
        range_color=[plot_data['Observation Value'].min(), plot_data['Observation Value'].max()],
        scope="africa",
        labels={'Observation Value': 'Deaths per 1,000 live births'},
        hover_name='REF_AREA',  # Country code on hover
        hover_data={
            'REF_AREA': False,  # Hide redundant country code
            'Observation Value': ':.1f'  # Format to one decimal place
        }
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title} ({year})",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': 'black', 'family': 'Arial, sans-serif'}
        },
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type='mercator',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='lightblue'
        ),
        coloraxis_colorbar=dict(
            title='Deaths per<br>1,000 live births',
            thicknessmode="pixels", 
            thickness=20,
            lenmode="pixels", 
            len=300
        ),
        margin={"r": 20, "t": 40, "l": 20, "b": 20},
        height=600,
        width=800
    )
    
    # Add country labels as annotations
    for idx, row in merged.iterrows():
        centroid = row.geometry.centroid
        fig.add_annotation(
            x=centroid.x,
            y=centroid.y,
            text=row['ISO3'],
            showarrow=False,
            font=dict(
                family="Arial, sans-serif",
                size=10,
                color="black"
            ),
            bgcolor="white",
            opacity=0.7,
            borderpad=2
        )
    
    return fig

# Create maps for both indicators
neonatal_map = create_mortality_map_plotly(
    eac_mortality,
    'Neonatal mortality rate',
    2020,
    'Neonatal Mortality Rate in East Africa'
)

under_five_map = create_mortality_map_plotly(
    eac_mortality,
    'Under-five mortality rate',
    2020,
    'Under-Five Mortality Rate in East Africa'
)

# Save the maps as interactive HTML files
# neonatal_map.write_html('neonatal_mortality_map.html')
# under_five_map.write_html('under_five_mortality_map.html')

# If you need static images
# neonatal_map.write_image('neonatal_mortality_map.png', width=1200, height=800, scale=2)
# under_five_map.write_image('under_five_mortality_map.png', width=1200, height=800, scale=2)

# Display the maps in a notebook
neonatal_map.show()
under_five_map.show()


# In[77]:


eac_countries = [
    "Burundi", "Kenya", "Rwanda", "South Sudan",
    "Tanzania", "Uganda", "Democratic Republic of the Congo", "Somalia"
]

eac_df = df[
    (df['Geographic area'].isin(eac_countries)) &
    (df['Indicator'].isin(["Neonatal mortality rate", "Under-five mortality rate"])) &
    (df['Sex'] == "Total") &
    (df['Wealth Quintile'] == "Total")
].copy()

avg_trends = eac_df.groupby(['Year', 'Indicator'])['Observation Value'].mean().reset_index()

fig_neonatal = px.line(
    eac_df[eac_df['Indicator'] == 'Neonatal mortality rate'],
    x='Year', y='Observation Value', color='Geographic area',
    labels={'Observation Value': 'Neonatal Mortality Rate'},
    title='Neonatal Mortality Rate Trends in EAC Countries'
)

avg_neonatal = avg_trends[avg_trends['Indicator'] == 'Neonatal mortality rate']
fig_neonatal.add_trace(
    go.Scatter(
        x=avg_neonatal['Year'],
        y=avg_neonatal['Observation Value'],
        mode='lines',
        name='EAC Average',
        line=dict(color='black', width=4, dash='dash')
    )
)

fig_neonatal.update_layout(legend_title_text='Country')
fig_neonatal.show()

fig_underfive = px.line(
    eac_df[eac_df['Indicator'] == 'Under-five mortality rate'],
    x='Year', y='Observation Value', color='Geographic area',
    labels={'Observation Value': 'Under-Five Mortality Rate'},
    title='Under-Five Mortality Rate Trends in EAC Countries'
)

# Add EAC average trend
avg_underfive = avg_trends[avg_trends['Indicator'] == 'Under-five mortality rate']
fig_underfive.add_trace(
    go.Scatter(
        x=avg_underfive['Year'],
        y=avg_underfive['Observation Value'],
        mode='lines',
        name='EAC Average',
        line=dict(color='black', width=4, dash='dash')
    )
)

fig_underfive.update_layout(legend_title_text='Country')
fig_underfive.show()

latest_df = eac_df.sort_values('Year').groupby(['Geographic area', 'Indicator']).tail(1)
latest_pivot = latest_df.pivot(index='Geographic area', columns='Indicator', values='Observation Value').reset_index()

# Bar charts
fig_bar_underfive = px.bar(
    latest_pivot.sort_values('Under-five mortality rate', ascending=False),
    x='Geographic area', y='Under-five mortality rate',
     color='Geographic area',
    title='Latest Under-Five Mortality Rate (EAC Countries)',
    labels={'Under-five mortality rate': 'Under-Five Mortality Rate'}
)
fig_bar_underfive.show()

fig_bar_neonatal = px.bar(
    latest_pivot.sort_values('Neonatal mortality rate', ascending=False),
    x='Geographic area', y='Neonatal mortality rate',
    color='Geographic area',
    title='Latest Neonatal Mortality Rate (EAC Countries)',
    labels={'Neonatal mortality rate': 'Neonatal Mortality Rate'}
)
fig_bar_neonatal.show()

# Identify countries with highest mortality
highest_underfive = latest_pivot.loc[latest_pivot['Under-five mortality rate'].idxmax(), 'Geographic area']
highest_neonatal = latest_pivot.loc[latest_pivot['Neonatal mortality rate'].idxmax(), 'Geographic area']

print(f"🚨 Highest Under-Five Mortality Rate: {highest_underfive}")
print(f"🚨 Highest Neonatal Mortality Rate: {highest_neonatal}")

