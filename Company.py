import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import seaborn as sns

##Testing for Company

def perfcalc(Start_date, Period_Days, ISIN):

    while True:
        try:
            # Try if a shareprice exists on selected end_date
            Shareprice_Start = Shareprice.loc[Shareprice['Date'] == Start_date, ISIN].values[0]
            break
        except IndexError:
            # if end_date doesn't exist, use the next available shareprice
            Start_date += pd.Timedelta(days=1)

    End_Date = Start_date + pd.Timedelta(days=Period_Days)


    while True:
        try:
            # Try if a shareprice exists on selected end_date
            Shareprice_End = Shareprice.loc[Shareprice['Date'] == End_Date, ISIN].values[0]
            break
        except IndexError:
            # if end_date doesn't exist, use the next available shareprice
            End_Date += pd.Timedelta(days=1)

    Shareprice_End = Shareprice.loc[Shareprice['Date'] == End_Date, ISIN].values[0]
    Performace = (Shareprice_End / Shareprice_Start) - 1

    return Performace


# path to csv files
data_folder = './Data/Company_files'

# get list of all files in the Data folder
all_files = os.listdir(data_folder)

# empty list for data frames per company
dataframes = []

# load csv files
for file in all_files:
    file_path = os.path.join(data_folder, file)

    # load csv file
    df = pd.read_csv(file_path)

    # convert date format
    df['Date'] = pd.to_datetime(df['Date'])

    # extract company isin (first word before spacing)
    company_isin = os.path.basename(file).split('.')[0].split(' ')[0]

    # add information about company
    df['Company'] = company_isin


    # add dataframe to list of dataframes
    dataframes.append(df)

    # merge dataframes on date and company
    data = dataframes[0]
    for df in dataframes[1:]:
        data = pd.merge(data, df, on=['Date', 'Company'], how='outer')

#Load Financial Data
file_path = './Data/Finance/Shareprice.xlsx'
Shareprice = pd.read_excel(file_path, sheet_name='Output')
Shareprice.replace(".", 0, inplace=True)

#Select Date for analysis
#pd.set_option('display.max_columns', None)
pd.reset_option('display.max_columns')
df = data[(data['Date'] >= '2020-05-01') & (data['Date'] <= '2023-05-31')]

print(df.head())

#Calculate Share Performance for x Days
period_days= [1,2,3]

for date in df['Date']:
    for period in period_days:
        new_column_name = 'Perf_' + str(period) + '_Days'
        df.loc[df['Date'] == date, new_column_name] = perfcalc(date, period, df.loc[df['Date'] == date, 'Company'].values[0])

print(df)
#Regression

# Define independent (Features) and dependent variables (Target)
df = df.fillna(0) #replace NaN by 0
X = df[['Cyber Attack', 'Data Security Management', 'Cyber Security', 'Data Breach']]  # independent variables
y = df['Perf_3_Days']      # dependent variable

# Split test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create & train model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# print results
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R²-Score:", r2_score(y_test, y_pred))
print("Koeffizienten:", model.coef_)
print("Intercept:", model.intercept_)



# Chart for mid-term presentation
fig, ax1 = plt.subplots()

# first chart for performance
ax1.plot(df['Date'], df['Perf_3_Days'] * 100, label='Performance in %', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('Performance in %', color='blue')
ax1.yaxis.set_major_formatter(PercentFormatter())
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('1 Day performance and cyber attack news data ')


# shorten date format as it was too long
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.tick_params(axis='x', rotation=45)


# second chart for cyber attack data
ax2 = ax1.twinx()
ax2.scatter(df['Date'], df['Cyber Attack'], label='Cyber attack news', color='orange')
ax2.set_ylabel('Cyber attack news', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')


plt.show()

#Correlation Chart
categories = ['Cyber Attack','Data Breach','Perf_3_Days','Perf_2_Days', 'Perf_1_Days']
corr = df[categories].corr()
plt.figure()
sns.heatmap(corr, annot=True, cmap='viridis')
plt.tight_layout()
plt.title('Correlation analysis')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

