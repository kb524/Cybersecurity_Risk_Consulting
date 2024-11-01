import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
data_folder = './Data/Apple'

# get list of all files in the Data folder
all_files = os.listdir(data_folder)

apple_files = [file for file in all_files if file.startswith("US0378331005")]


# empty list for data frames per company
dataframes = []

# load csv files
for file in apple_files:
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
file_path = './Data/Apple/Shareprice.xlsx'
Shareprice = pd.read_excel(file_path, sheet_name='Output')
Shareprice.replace(".", 0, inplace=True)

#Select Date for analysis
pd.set_option('display.max_columns', None)
data_mai = data[(data['Date'] >= '2023-05-01') & (data['Date'] <= '2023-05-31')]


#Calculate Share Performance for 3 Days
period_days= [3]

for date in data_mai['Date']:
    for period in period_days:
        new_column_name = 'Perf_' + str(period) + '_Days'
        data_mai.loc[data_mai['Date'] == date, new_column_name] = perfcalc(date, period, data_mai.loc[data_mai['Date'] == date, 'Company'].values[0])

print(data_mai)
#Regression

# Define independent (Features) and dependent variables (Target)
data_mai = data_mai.fillna(0) #replace NaN by 0
X = data_mai[['Cyber Attack', 'Data Security Management', 'Cyber Security', 'Data Breach']]  # independent variables
y = data_mai['Perf_3_Days']      # dependent variable

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


