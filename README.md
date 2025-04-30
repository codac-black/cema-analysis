# CEMA HIV and Mortality Analysis Project

This project analyzes two key public health aspects:
1. The relationship between HIV cases and multidimensional poverty using WHO and World Bank datasets
2. Under-five and neonatal mortality rates in East African Community (EAC) countries

## Project Structure

```
cema/
├── src/
│   ├── cema.py          # Python implementation of analysis
│   ├── cema.md          # Markdown documentation of analysis
│   ├── cema.ipynb       # Jupyter notebook version
│   ├── analysis.rmd     # R Markdown implementation
│   └── data/
│       ├── HIV data 2000-2023.csv              # WHO HIV data
│       ├── dataset_datascience.csv             # Mortality data
│       └── multidimensional_poverty.xlsx       # World Bank poverty data
└── README.md
```

## Analysis Components

### Task 1: HIV and Poverty Analysis
- Visualization of HIV trends in countries contributing to 75% of global burden
- Analysis of HIV trends by WHO regions
- Statistical analysis of relationship between HIV cases and multidimensional poverty
- Mixed effects modeling accounting for country-level random effects

### Task 2: EAC Mortality Analysis
- Analysis of under-five mortality rates
- Analysis of neonatal mortality rates
- Temporal trends and geographic patterns
- Comparative analysis across EAC member states

## Implementation Options

The analysis is implemented in both R and Python, offering flexibility in choice of tools:

### Python Implementation
Required packages:
```
pandas
numpy
matplotlib
seaborn
statsmodels
plotly
```

### R Implementation
Required packages:
```
tidyverse
ggplot2
sf
lme4
plotly
readxl
viridis
```

## Running the Analysis

### Python Users
1. Install required packages: `pip install pandas numpy matplotlib seaborn statsmodels plotly`
2. Run either:
   - `python src/cema.py`
   - Open `src/cema.ipynb` in Jupyter Notebook

### R Users
1. Install required packages: `install.packages(c("tidyverse", "ggplot2", "sf", "lme4", "plotly", "readxl", "viridis"))`
2. Open and knit `src/analysis.rmd` in RStudio

## Data Sources

- WHO Global Health Observatory: HIV data (2000-2023)
- World Bank: Multidimensional poverty data
- UN Inter-agency Group: Child Mortality Estimation data

## Key Outputs

- Interactive visualizations of HIV trends
- Statistical analysis of poverty-HIV relationship
- Choropleth maps of mortality rates
- Time series analysis of mortality trends
- Policy recommendations based on findings

## Dependencies

- Python 3.7+ or R 4.0+
- IDE: VS Code, RStudio, or Jupyter Notebook
- See implementation-specific package lists above

## License

This project is licensed under the MIT License. See the LICENSE file for details.git commit -m "Initial project setup

