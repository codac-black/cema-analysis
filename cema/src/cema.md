# Task 1
You are provided with a dataset from the World Health Organization (WHO) Global Observatory, containing data on people living with HIV at the country level from 2000 to 2023.

Using this dataset, we would like you to:

- Create a visualization that shows the trend of HIV cases in the countries that contribute to 75% of the global burden
- Generate a visualization that displays the trend of HIV cases in the countries contributing to 75% of the burden within each WHO region (column called ParentLocationCode contains the WHO regions)
- You have also been provided with World Bank data on the multidimensional poverty headcount ratio, which includes factors such as income, educational attainment, school enrolment, electricity access, sanitation and drinking water.

We would like you to merge this dataset with the HIV data above and analyze the relationship between people living with HIV and multidimensional poverty, and the individual factors that contribute to the ratio. Remember to account for the random effects (country, year).

Write a paragraph on your findings.


```python
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
```


```python
plt.style.use('seaborn-v0_8-darkgrid')
```


```python
hiv_data = pd.read_csv('HIV data 2000-2023.csv', encoding='latin1')

```


```python
poverty_data = pd.read_excel('multidimensional_poverty.xlsx')

```


```python
hiv_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IndicatorCode</th>
      <th>Indicator</th>
      <th>ValueType</th>
      <th>ParentLocationCode</th>
      <th>ParentLocation</th>
      <th>Location type</th>
      <th>SpatialDimValueCode</th>
      <th>Location</th>
      <th>Period type</th>
      <th>Period</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2023</td>
      <td>320 000 [280 000 - 380 000]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2022</td>
      <td>320 000 [280 000 - 380 000]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2021</td>
      <td>320 000 [280 000 - 380 000]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2020</td>
      <td>320 000 [280 000 - 370 000]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2015</td>
      <td>300 000 [260 000 - 350 000]</td>
    </tr>
  </tbody>
</table>
</div>




```python
hiv_data.isna().sum()
```




    IndicatorCode          0
    Indicator              0
    ValueType              0
    ParentLocationCode     0
    ParentLocation         0
    Location type          0
    SpatialDimValueCode    0
    Location               0
    Period type            0
    Period                 0
    Value                  0
    dtype: int64




```python
poverty_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Country code</th>
      <th>Economy</th>
      <th>Reporting year</th>
      <th>Survey name</th>
      <th>Survey year</th>
      <th>Survey coverage</th>
      <th>Welfare type</th>
      <th>urvey comparability</th>
      <th>Monetary (%)</th>
      <th>Educational attainment (%)</th>
      <th>Educational enrollment (%</th>
      <th>Electricity (%)</th>
      <th>Sanitation (%)</th>
      <th>Drinking water (%)</th>
      <th>Multidimensional poverty headcount ratio (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SSA</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>2018</td>
      <td>IDREA</td>
      <td>2018</td>
      <td>N</td>
      <td>c</td>
      <td>2</td>
      <td>31.122005</td>
      <td>29.753423</td>
      <td>27.44306</td>
      <td>52.639532</td>
      <td>53.637516</td>
      <td>32.106507</td>
      <td>47.203606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ECA</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>2012</td>
      <td>HBS</td>
      <td>2018</td>
      <td>N</td>
      <td>c</td>
      <td>1</td>
      <td>0.048107</td>
      <td>0.19238</td>
      <td>-</td>
      <td>0.06025</td>
      <td>6.579772</td>
      <td>9.594966</td>
      <td>0.293161</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LAC</td>
      <td>ARG</td>
      <td>Argentina</td>
      <td>2010</td>
      <td>EPHC-S2</td>
      <td>2021</td>
      <td>U</td>
      <td>i</td>
      <td>3</td>
      <td>0.894218</td>
      <td>1.08532</td>
      <td>0.731351</td>
      <td>0</td>
      <td>0.257453</td>
      <td>0.364048</td>
      <td>0.906573</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ECA</td>
      <td>ARM</td>
      <td>Armenia</td>
      <td>2010</td>
      <td>ILCS</td>
      <td>2021</td>
      <td>N</td>
      <td>c</td>
      <td>1</td>
      <td>0.523521</td>
      <td>0</td>
      <td>1.793004</td>
      <td>0</td>
      <td>0.397725</td>
      <td>0.660082</td>
      <td>0.523521</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EAP</td>
      <td>AUS</td>
      <td>Australia</td>
      <td>2010</td>
      <td>SIH-LIS</td>
      <td>2018</td>
      <td>N</td>
      <td>I</td>
      <td>3</td>
      <td>0.516880</td>
      <td>1.71188</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
      <td>2.215770</td>
    </tr>
  </tbody>
</table>
</div>




```python
poverty_data.isna().sum()
```




    Region                                          0
    Country code                                    0
    Economy                                         0
    Reporting year                                  0
    Survey name                                     0
    Survey year                                     0
    Survey coverage                                 0
    Welfare type                                    0
    urvey comparability                             0
    Monetary (%)                                    0
    Educational attainment (%)                      0
    Educational enrollment (%                       0
    Electricity (%)                                 0
    Sanitation (%)                                  0
    Drinking water (%)                              0
    Multidimensional poverty headcount ratio (%)    0
    dtype: int64




```python
hiv_data['Value_cleaned'] = hiv_data['Value'].astype(str)
hiv_data['Value_cleaned'] = hiv_data['Value_cleaned'].str.split('[').str[0]
hiv_data['Value_cleaned'] = hiv_data['Value_cleaned'].str.replace(' ', '').str.strip()
hiv_data['Value_numeric'] = pd.to_numeric(hiv_data['Value_cleaned'], errors='coerce')
```


```python
hiv_data.isna().sum()
```




    IndicatorCode            0
    Indicator                0
    ValueType                0
    ParentLocationCode       0
    ParentLocation           0
    Location type            0
    SpatialDimValueCode      0
    Location                 0
    Period type              0
    Period                   0
    Value                    0
    Value_cleaned            0
    Value_numeric          468
    dtype: int64




```python
# Group the HIV data by 'Location' and 'Period' to calculate the total 'Value_numeric' for each combination
hiv_by_country = hiv_data.groupby(['Location', 'Period'])['Value_numeric'].sum().reset_index()

```


```python
global_total = hiv_by_country.groupby('Period')['Value_numeric'].sum().reset_index()
global_total.columns = ['Year', 'global_cases']
```


```python
hiv_by_country['Period'] = hiv_by_country['Period'].astype(int)
global_total['Year'] = global_total['Year'].astype(int)
```


```python
hiv_by_country.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Period</th>
      <th>Value_numeric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2005</td>
      <td>2800.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>4100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2015</td>
      <td>6500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2020</td>
      <td>10000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
global_total.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>global_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>21348540.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005</td>
      <td>23487320.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>26054020.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>28922150.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020</td>
      <td>31014450.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
country_contribution = hiv_by_country.merge(
    global_total, 
    left_on='Period',
    right_on='Year' ,
    how='left'
    ) 

```


```python
country_contribution = country_contribution.drop('Period',axis=1)
```


```python
country_contribution.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Value_numeric</th>
      <th>Year</th>
      <th>global_cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1600.0</td>
      <td>2000</td>
      <td>21348540.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2800.0</td>
      <td>2005</td>
      <td>23487320.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>4100.0</td>
      <td>2010</td>
      <td>26054020.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>6500.0</td>
      <td>2015</td>
      <td>28922150.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>10000.0</td>
      <td>2020</td>
      <td>31014450.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
country_contribution['contribution'] = np.where(
        country_contribution['global_cases'] > 0,
        country_contribution['Value_numeric'] / country_contribution['global_cases'],
        0 # Assign 0 contribution if global cases is 0
    )

country_contribution.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>Value_numeric</th>
      <th>Year</th>
      <th>global_cases</th>
      <th>contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1178</th>
      <td>Saint Vincent and the Grenadines</td>
      <td>0.0</td>
      <td>2010</td>
      <td>26054020.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>465</th>
      <td>Eswatini</td>
      <td>170000.0</td>
      <td>2005</td>
      <td>23487320.0</td>
      <td>0.007238</td>
    </tr>
    <tr>
      <th>959</th>
      <td>Nauru</td>
      <td>0.0</td>
      <td>2023</td>
      <td>33932790.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>Saint Lucia</td>
      <td>0.0</td>
      <td>2023</td>
      <td>33932790.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>United Arab Emirates</td>
      <td>860.0</td>
      <td>2015</td>
      <td>28922150.0</td>
      <td>0.000030</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Albania</td>
      <td>0.0</td>
      <td>2005</td>
      <td>23487320.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>459</th>
      <td>Estonia</td>
      <td>6900.0</td>
      <td>2015</td>
      <td>28922150.0</td>
      <td>0.000239</td>
    </tr>
    <tr>
      <th>1048</th>
      <td>Palau</td>
      <td>0.0</td>
      <td>2000</td>
      <td>21348540.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>563</th>
      <td>Guatemala</td>
      <td>31000.0</td>
      <td>2015</td>
      <td>28922150.0</td>
      <td>0.001072</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>Sri Lanka</td>
      <td>3800.0</td>
      <td>2005</td>
      <td>23487320.0</td>
      <td>0.000162</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sum contributions across all years for each country
country_total_contribution = country_contribution.groupby('Location')['contribution'].sum().reset_index()
country_total_contribution = country_total_contribution.sort_values('contribution', ascending=False)
```


```python
country_total_contribution.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>contribution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>161</th>
      <td>South Africa</td>
      <td>1.787466</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Nigeria</td>
      <td>0.533513</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Mozambique</td>
      <td>0.519931</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Kenya</td>
      <td>0.436397</td>
    </tr>
    <tr>
      <th>184</th>
      <td>United Republic of Tanzania</td>
      <td>0.407978</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Zimbabwe</td>
      <td>0.396269</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Uganda</td>
      <td>0.368979</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Zambia</td>
      <td>0.310133</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Malawi</td>
      <td>0.278307</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Brazil</td>
      <td>0.208368</td>
    </tr>
  </tbody>
</table>
</div>




```python
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

```

    
    Countries cumulatively contributing to ~75% of total HIV burden contribution: South Africa, Nigeria, Mozambique, Kenya, United Republic of Tanzania, Zimbabwe, Uganda, Zambia, Malawi, Brazil, Thailand, Ethiopia, Democratic Republic of the Congo, Cote d'Ivoire, Indonesia
    


```python
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

```


    
![png](cema_files/cema_23_0.png)
    


    Saved plot: hiv_top_countries.png
    


```python
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
```



    Saved plot: hiv_top_countries_bar_plotly.png
    


    <Figure size 1400x700 with 0 Axes>



```python

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

```



    Saved plot: hiv_top_countries.html
    


```python
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
```


    
![png](cema_files/cema_26_0.png)
    


    
    Analyzing relationship between HIV and poverty...
    


```python
poverty_data['Reporting year',] = pd.to_numeric(poverty_data['Reporting year'], errors='coerce').astype('Int64')
```


```python
combined_data = hiv_data.merge(
    poverty_data,
    left_on=['SpatialDimValueCode', 'Period'],
    right_on=['Country code','Reporting year'],
    how='left'
)
combined_data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IndicatorCode</th>
      <th>Indicator</th>
      <th>ValueType</th>
      <th>ParentLocationCode</th>
      <th>ParentLocation</th>
      <th>Location type</th>
      <th>SpatialDimValueCode</th>
      <th>Location</th>
      <th>Period type</th>
      <th>Period</th>
      <th>...</th>
      <th>Welfare type</th>
      <th>urvey comparability</th>
      <th>Monetary (%)</th>
      <th>Educational attainment (%)</th>
      <th>Educational enrollment (%</th>
      <th>Electricity (%)</th>
      <th>Sanitation (%)</th>
      <th>Drinking water (%)</th>
      <th>Multidimensional poverty headcount ratio (%)</th>
      <th>(Reporting year,)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2023</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2022</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2021</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2020</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2015</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
combined_data.isna().sum()
```




    IndicatorCode                                      0
    Indicator                                          0
    ValueType                                          0
    ParentLocationCode                                 0
    ParentLocation                                     0
    Location type                                      0
    SpatialDimValueCode                                0
    Location                                           0
    Period type                                        0
    Period                                             0
    Value                                              0
    Value_cleaned                                      0
    Value_numeric                                    468
    Region                                          1504
    Country code                                    1504
    Economy                                         1504
    Reporting year                                  1504
    Survey name                                     1504
    Survey year                                     1504
    Survey coverage                                 1504
    Welfare type                                    1504
    urvey comparability                             1504
    Monetary (%)                                    1504
    Educational attainment (%)                      1504
    Educational enrollment (%                       1504
    Electricity (%)                                 1504
    Sanitation (%)                                  1504
    Drinking water (%)                              1504
    Multidimensional poverty headcount ratio (%)    1504
    (Reporting year,)                               1504
    dtype: int64




```python
combined_data.size

```




    46560




```python
combined_data.isna().sum()
```




    IndicatorCode                                      0
    Indicator                                          0
    ValueType                                          0
    ParentLocationCode                                 0
    ParentLocation                                     0
    Location type                                      0
    SpatialDimValueCode                                0
    Location                                           0
    Period type                                        0
    Period                                             0
    Value                                              0
    Value_cleaned                                      0
    Value_numeric                                    468
    Region                                          1504
    Country code                                    1504
    Economy                                         1504
    Reporting year                                  1504
    Survey name                                     1504
    Survey year                                     1504
    Survey coverage                                 1504
    Welfare type                                    1504
    urvey comparability                             1504
    Monetary (%)                                    1504
    Educational attainment (%)                      1504
    Educational enrollment (%                       1504
    Electricity (%)                                 1504
    Sanitation (%)                                  1504
    Drinking water (%)                              1504
    Multidimensional poverty headcount ratio (%)    1504
    (Reporting year,)                               1504
    dtype: int64




```python
print(combined_data[['Value_numeric', 'Multidimensional poverty headcount ratio (%)']].isna().sum())
print(combined_data[['Value_numeric', 'Multidimensional poverty headcount ratio (%)']].describe())
```

    Value_numeric                                    468
    Multidimensional poverty headcount ratio (%)    1504
    dtype: int64
           Value_numeric  Multidimensional poverty headcount ratio (%)
    count   1.084000e+03                                     48.000000
    mean    2.096581e+05                                      8.808809
    std     6.616030e+05                                     17.204667
    min     5.100000e+02                                      0.000000
    25%     6.200000e+03                                      0.468174
    50%     2.500000e+04                                      1.971630
    75%     1.200000e+05                                      5.683537
    max     7.700000e+06                                     78.252000
    


```python
combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
```


```python
combined_data = combined_data.dropna(subset=['Value_numeric', 'Multidimensional poverty headcount ratio (%)'])

```


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
combined_data['Value_numeric_scaled'] = scaler.fit_transform(combined_data[['Value_numeric']])
combined_data['Poverty_ratio_scaled'] = scaler.fit_transform(combined_data[['Multidimensional poverty headcount ratio (%)']])
```


```python
# Use the correct column name for poverty index
model = MixedLM(
    combined_data['Value_numeric_scaled'], 
    combined_data[['Poverty_ratio_scaled']], 
    groups=combined_data['Location']
)
result = model.fit()
print(result.summary())
```

                  Mixed Linear Model Regression Results
    ==================================================================
    Model:            MixedLM Dependent Variable: Value_numeric_scaled
    No. Observations: 1552    Method:             REML                
    No. Groups:       194     Scale:              0.0630              
    Min. group size:  8       Log-Likelihood:     -524.7527           
    Max. group size:  8       Converged:          Yes                 
    Mean group size:  8.0                                             
    ------------------------------------------------------------------
                            Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    ------------------------------------------------------------------
    Poverty_ratio_scaled    -0.006    0.007 -0.906 0.365 -0.019  0.007
    Group Var                0.938    0.409                           
    ==================================================================
    
    


```python
from statsmodels.regression.linear_model import OLS

model = OLS(
    combined_data['Value_numeric_scaled'], 
    combined_data[['Poverty_ratio_scaled']]
)
result = model.fit()
print(result.summary())
```

                                      OLS Regression Results                                 
    =========================================================================================
    Dep. Variable:     Value_numeric_scaled   R-squared (uncentered):                   0.003
    Model:                              OLS   Adj. R-squared (uncentered):              0.002
    Method:                   Least Squares   F-statistic:                              4.324
    Date:                  Mon, 28 Apr 2025   Prob (F-statistic):                      0.0378
    Time:                          16:01:52   Log-Likelihood:                         -2200.0
    No. Observations:                  1552   AIC:                                      4402.
    Df Residuals:                      1551   BIC:                                      4407.
    Df Model:                             1                                                  
    Covariance Type:              nonrobust                                                  
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    Poverty_ratio_scaled     0.0527      0.025      2.079      0.038       0.003       0.102
    ==============================================================================
    Omnibus:                     2320.360   Durbin-Watson:                   0.205
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           697516.484
    Skew:                           8.993   Prob(JB):                         0.00
    Kurtosis:                     105.288   Cond. No.                         1.00
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    


```python
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
```


    
![png](cema_files/cema_38_0.png)
    



    
![png](cema_files/cema_38_1.png)
    



    
![png](cema_files/cema_38_2.png)
    



```python


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
```








```python
combined_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IndicatorCode</th>
      <th>Indicator</th>
      <th>ValueType</th>
      <th>ParentLocationCode</th>
      <th>ParentLocation</th>
      <th>Location type</th>
      <th>SpatialDimValueCode</th>
      <th>Location</th>
      <th>Period type</th>
      <th>Period</th>
      <th>...</th>
      <th>Monetary (%)</th>
      <th>Educational attainment (%)</th>
      <th>Educational enrollment (%)</th>
      <th>Electricity (%)</th>
      <th>Sanitation (%)</th>
      <th>Drinking water (%)</th>
      <th>Multidimensional poverty headcount ratio (%)</th>
      <th>(Reporting year,)</th>
      <th>Value_numeric_scaled</th>
      <th>Poverty_ratio_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2023</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>&lt;NA&gt;</td>
      <td>0.309392</td>
      <td>-0.081083</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2022</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>&lt;NA&gt;</td>
      <td>0.309392</td>
      <td>-0.081083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2021</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>&lt;NA&gt;</td>
      <td>0.309392</td>
      <td>-0.081083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2020</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>&lt;NA&gt;</td>
      <td>0.309392</td>
      <td>-0.081083</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HIV_0000000001</td>
      <td>Estimated number of people (all ages) living w...</td>
      <td>numeric</td>
      <td>AFR</td>
      <td>Africa</td>
      <td>Country</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Year</td>
      <td>2015</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>&lt;NA&gt;</td>
      <td>0.273740</td>
      <td>-0.081083</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



combined_data.columns

# Question 2
You have been provided with data on the under-five mortality rate and neonatal mortality rate for the African region, which has been downloaded from the UN Inter-agency Group for Child Mortality Estimation. Your task is to:

- Filter data for the eight countries belonging to the East African Community (list here: https://www.eac.int/overview-of-eac)
- Visualize the latest estimate of each indicator at the country level using shapefiles, which can be downloaded from www.gadm.org.
- Show the average trends in the mortality rates over time (plot the average trend line and add the points in the graphic for the country level estimates for each indicator. Expectation: two plots).
- Based on your visualizations, identify the countries with the highest under-five mortality rates in East Africa and the highest neonatal mortality.


```python
# load the dataset
df = pd.read_csv('dataset_datascience.csv')
```

    C:\Users\hp\AppData\Local\Temp\ipykernel_17988\2578247156.py:1: DtypeWarning:
    
    Columns (2) have mixed types. Specify dtype option on import or set low_memory=False.
    
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REF_AREA</th>
      <th>Geographic area</th>
      <th>Regional group</th>
      <th>Indicator</th>
      <th>Sex</th>
      <th>Wealth Quintile</th>
      <th>Series Name</th>
      <th>Series Year</th>
      <th>Reference Date</th>
      <th>Observation Value</th>
      <th>...</th>
      <th>Country notes</th>
      <th>Observation Status</th>
      <th>Unit of measure</th>
      <th>Series Type</th>
      <th>Series Category</th>
      <th>Series Method</th>
      <th>Age Group of Women</th>
      <th>Time Since First Birth</th>
      <th>Definition</th>
      <th>Interval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>Neonatal mortality rate</td>
      <td>Total</td>
      <td>Total</td>
      <td>Afghanistan Health Survey 2018 (Direct)</td>
      <td>2018</td>
      <td>1995.5</td>
      <td>47.869030</td>
      <td>...</td>
      <td>NaN</td>
      <td>Excluded from IGME</td>
      <td>Deaths per 1,000 live births</td>
      <td>Direct</td>
      <td>Others</td>
      <td>Survey/Census with Full Birth Histories</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>Neonatal mortality rate</td>
      <td>Total</td>
      <td>Total</td>
      <td>Afghanistan Health Survey 2018 (Direct)</td>
      <td>2018</td>
      <td>2000.5</td>
      <td>35.349317</td>
      <td>...</td>
      <td>NaN</td>
      <td>Excluded from IGME</td>
      <td>Deaths per 1,000 live births</td>
      <td>Direct</td>
      <td>Others</td>
      <td>Survey/Census with Full Birth Histories</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>Neonatal mortality rate</td>
      <td>Total</td>
      <td>Total</td>
      <td>Afghanistan Health Survey 2018 (Direct)</td>
      <td>2018</td>
      <td>2005.5</td>
      <td>27.699219</td>
      <td>...</td>
      <td>NaN</td>
      <td>Excluded from IGME</td>
      <td>Deaths per 1,000 live births</td>
      <td>Direct</td>
      <td>Others</td>
      <td>Survey/Census with Full Birth Histories</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>Neonatal mortality rate</td>
      <td>Total</td>
      <td>Total</td>
      <td>Afghanistan Health Survey 2018 (Direct)</td>
      <td>2018</td>
      <td>2010.5</td>
      <td>21.056003</td>
      <td>...</td>
      <td>NaN</td>
      <td>Excluded from IGME</td>
      <td>Deaths per 1,000 live births</td>
      <td>Direct</td>
      <td>Others</td>
      <td>Survey/Census with Full Birth Histories</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>Neonatal mortality rate</td>
      <td>Total</td>
      <td>Total</td>
      <td>Afghanistan Health Survey 2018 (Direct)</td>
      <td>2018</td>
      <td>2015.5</td>
      <td>20.167379</td>
      <td>...</td>
      <td>NaN</td>
      <td>Excluded from IGME</td>
      <td>Deaths per 1,000 live births</td>
      <td>Direct</td>
      <td>Others</td>
      <td>Survey/Census with Full Birth Histories</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df.isna().sum()
```




    REF_AREA                       0
    Geographic area                0
    Regional group            122426
    Indicator                      0
    Sex                            0
    Wealth Quintile                0
    Series Name                    0
    Series Year                    0
    Reference Date                 0
    Observation Value             36
    Lower Bound                55615
    Upper Bound                55615
    Standard Error             98771
    Country notes             124854
    Observation Status             0
    Unit of measure                0
    Series Type                73949
    Series Category            73949
    Series Method              73949
    Age Group of Women        123269
    Time Since First Birth    128804
    Definition                129564
    Interval                   10795
    dtype: int64




```python
# drop columns with all NaN values
df = df.dropna(axis=1, how='all')
```


```python
df['Reference Date'] = df['Reference Date'].astype(str)

df['Year'] = df['Reference Date'].str.split('.').str[0].astype(int)

df[['Reference Date', 'Year']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reference Date</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1995.5</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000.5</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005.5</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010.5</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015.5</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.Indicator.unique()
```




    array(['Neonatal mortality rate', 'Under-five mortality rate'],
          dtype=object)




```python
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


```


```python
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
```


```python
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
```


    
![png](cema_files/cema_51_0.png)
    



    
![png](cema_files/cema_51_1.png)
    



```python
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
```






```python
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

```









    🚨 Highest Under-Five Mortality Rate: Somalia
    🚨 Highest Neonatal Mortality Rate: South Sudan
    
