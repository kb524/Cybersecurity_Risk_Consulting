import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# path to csv files
data_folder = './Data/Cyber_News'

# get list of all files in the Data folder
all_files = os.listdir(data_folder)

# empty list for data frames per company
dataframes = []

# load csv files
for file in all_files:
    file_path = os.path.join(data_folder, file)

    # extract company isin (first word before spacing)
    company_isin = os.path.basename(file).split('.')[0].split(' ')[0]

    # load csv file
    df = pd.read_csv(file_path)

    # add information about company
    df['Company'] = company_isin

    # add dataframe to list of dataframes
    dataframes.append(df)

    # merge dataframes on date and company
    data = dataframes[0]
    for df in dataframes[1:]:
        data = pd.merge(data, df, on=['Date', 'Company'], how='outer')


# Show statistics and columns
pd.set_option('display.max_columns', None)
#reset set_option pd.reset_option('display.max_columns')
print(data.head())
print(data.describe())
print(data.columns)

# Histogramm static with matplotlib
df_short = data.loc[data['Company'] == 'US0378331005']

categories = ['Volume of News', 'Cyber Attack', 'Data Security Management','Cyber Security', 'Data Breach']

plt.hist(df_short[categories], label= categories, range=(0,15000))
plt.legend()
plt.xlabel('Quantity per Day')
plt.ylabel('Frequency')
plt.title('Histogram of US0378331005')
plt.show()

# Histogramm interactive with plotly
fig = px.histogram(df_short[categories], title='Interactive Histogram of US0378331005')
fig.update_layout(
    xaxis_title='Quantity per Day',
    yaxis_title='Frequency',
    legend_title='Legend'
)
#fig.show()

#Correlations
categories = ['Volume of News', 'Cyber Attack', 'Data Security Management','Cyber Security', 'Data Breach','Perc. of Positive Sentiment']
corr = df_short[categories].corr()
plt.figure()
sns.heatmap(corr, annot=True, cmap='viridis')
plt.tight_layout()
plt.show()

#boxplot
df_short[categories].plot(kind='box', figsize=(10, 8), subplots=True, sharex=False, sharey=False)
plt.tight_layout()
plt.show()

# Histogramm Perc. of Positive Sentiment

category = ['Perc. of Positive Sentiment']

plt.hist(df_short[category], label= category)
plt.legend()
plt.xlabel('Perc. of Positive Sentiment')
plt.ylabel('Frequency')
plt.title('Histogram Perc. of Positive Sentiment')
plt.show()

##Load Files with finance data

# Define the file path for file MarketCap
file_path = './Data/Finance/MarketCap.xlsx'

# Load the Excel file, specifically the sheet "Output"
MarketCap = pd.read_excel(file_path, sheet_name='Output')

# Display the loaded DataFrame
print(MarketCap.head())

plt.plot(MarketCap['Date'],MarketCap['US0378331005'])
plt.xlabel('Date')
plt.ylabel('MarketCap in millions')
plt.title('Development of Apple MarketCap')
plt.show()


# Define the file path for file Shareprice
file_path = './Data/Finance/Shareprice.xlsx'
Shareprice = pd.read_excel(file_path, sheet_name='Output')
#replace missing values with 0
Shareprice.replace(".", 0, inplace=True)

ISINs = Shareprice.columns.tolist()[1:len(Shareprice.columns)]

#calculate performance in a new column
for ISIN in ISINs:
    new_name= 'Perf_' + ISIN
    Shareprice[new_name] = Shareprice[ISIN].pct_change()


plt.plot(Shareprice['Date'],Shareprice['Perf_US0378331005'])
plt.xlabel('Date')
plt.ylabel('% Change')
plt.title('Share performance of Apple MarketCap')
plt.show()
