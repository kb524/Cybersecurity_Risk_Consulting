import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from pandasgui import show
from tabulate import tabulate


#ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#Define function for performance calculation
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
data_folder = './Data/Cyber_News'

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

print(Shareprice.head())

#Select Date for analysis
pd.set_option('display.max_columns', None)
#pd.reset_option('display.max_columns')
df = data[(data['Date'] >= '2020-01-01') & (data['Date'] <= '2023-05-31')]
df.fillna(0, inplace=True)

show(df)

##Weekend fix
#identify trading days
df.loc[:, 'Trading_day'] = 0
for i in df.index:
    if df.loc[i, 'Date'] in Shareprice['Date'].values:
        df.loc[i, 'Trading_day'] = 1


#show(df)


# copy values from not-trading days to trading days
columns_to_copy = [col for col in df.columns if col not in ['Date', 'Company','Trading_day','Perc. of Positive Sentiment']]
for idx in df[df['Trading_day'] == 0].index:
    # find next date with Trading_day = 1
    next_idx = df[(df.index > idx) & (df['Trading_day'] == 1)].index.min()
    if not pd.isna(next_idx):
        #copy all values exept date, company, Trading day, sentiment
        df.loc[next_idx, columns_to_copy] += df.loc[idx, columns_to_copy]

# Delete non-trading days
df = df[df['Trading_day'] != 0]


#Calculate Share Performance for x Days
period_days= [1,2,3]

for date in df['Date']:
    for period in period_days:
        new_column_name = 'Perf_' + str(period) + '_Days'
        df.loc[df['Date'] == date, new_column_name] = perfcalc(date, period, df.loc[df['Date'] == date, 'Company'].values[0])


#Linear Regression for one y

# Define independent (Features) and dependent variables (Target)
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

#Linear Regression to compare different y
# Define independent variables
X = df[['Cyber Attack', 'Data Security Management', 'Cyber Security', 'Data Breach']]  # independent variables

# Define different dependent variables
targets = ['Perf_1_Days', 'Perf_2_Days','Perf_3_Days']

# Initialize results list
results = []

# Loop through each target variable
for target in targets:
    y = df[target]  # dependent variable

    # Split test and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create & train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Collect results
    results.append({
        'Target': target,
        'MSE': mean_squared_error(y_test, y_pred),
        'R2_Score': r2_score(y_test, y_pred),
        'Coefficients': [float(round(c, 4)) for c in model.coef_],
        'Intercept': float(round(model.intercept_, 4))
    })

# prepare results table
table_data = [
    [
        result['Target'],
        result['MSE'],
        result['R2_Score'],
        result['Coefficients'],
        result['Intercept']
    ]
    for result in results
]
headers = ["Target", "MSE", "R2_Score", "Coefficients", "Intercept"]

# show table
print(tabulate(table_data, headers=headers, tablefmt="grid"))

