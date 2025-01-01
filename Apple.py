import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from pandasgui import show
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from sklearn.preprocessing import MinMaxScaler


#ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


###Load & adjustment of cyber news data
##Pipline for cyber news data
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



#Statistical description of data
pd.set_option('display.max_columns', None)
#pd.reset_option('display.max_columns')
#print(data[["Volume of News","Perc. of Positive Sentiment","Cyber Attack","Cyber Security","Data Breach","Data Security Management"]].describe())

#replace nan by 0
df= data.copy()
df.fillna(0, inplace=True)


#Create visulalization of cyber news total data
df['Total_Cyber_News'] = df['Cyber Attack'] + df['Data Security Management']+ df['Cyber Security'] + df['Data Breach']

plt.plot(df['Date'], df['Total_Cyber_News'], label = 'Total Cyber News')
plt.xlabel('Date')
plt.ylabel('Total Cyber News')
plt.title('Development of total cyber news data')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

###Load Financial Data
file_path = './Data/Finance/Shareprice.xlsx'
Shareprice = pd.read_excel(file_path, sheet_name='Output')
Shareprice.replace(".", 0, inplace=True)

#Statistical description of data
#print(Shareprice['US0378331005'].describe())


#Create visulalization of shareprice
plt.plot(Shareprice['Date'], Shareprice['US0378331005'], label = 'Apple Inc. stock price')
plt.xlabel('Date')
plt.ylabel('Stock price in USD')
plt.title('Development of Apple Inc. stock price')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

##Weekend fix
#identify trading days according to Nasdaq calendar
nasdaq_calendar = mcal.get_calendar('NASDAQ')

# Fetch the schedule for the specified date range
start_date = df['Date'].min()
end_date = df['Date'].max()
schedule = nasdaq_calendar.schedule(start_date=start_date, end_date=end_date)

# Extract the list of valid trading dates
trading_days = schedule.index

##Fix for cyber news set
# Initialize the 'Trading_day' column to 0
df['Trading_day'] = 0

# Update 'Trading_day' column based on trading days
for i in df.index:
    if df.loc[i, 'Date'] in trading_days:
        df.loc[i, 'Trading_day'] = 1


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

##fix for shareprice dataset
Shareprice['Trading_day'] = 0

# Update 'Trading_day' column based on trading days
for i in Shareprice.index:
    if Shareprice.loc[i, 'Date'] in trading_days:
        Shareprice.loc[i, 'Trading_day'] = 1

# Delete non-trading days
Shareprice = Shareprice[Shareprice['Trading_day'] != 0]

print(Shareprice['US0378331005'].describe())
##Find dates for analysis
#first_date = df.loc[df['Total_Cyber_News'] >= 600, 'Date'].iloc[0]
#print(first_date)
df= df[(df['Date'] <= '2024-05-28')]

###Calculate Share Performance for x Days
#Define function for performance calculation
def perfcalc(Start_date, Period_Days, ISIN):
    # Define valid date range to prevent infinite loops
    min_date = Shareprice['Date'].min()
    max_date = Shareprice['Date'].max()

    while True:
        if Start_date < min_date or Start_date > max_date:
            raise ValueError("Start_date is out of range.")
        try:
            # Try if a shareprice exists on selected Start_date
            Shareprice_Start = Shareprice.loc[Shareprice['Date'] == Start_date, ISIN].values[0]
            break
        except IndexError:
            # Move to the next day if Start_date is not found
            Start_date += pd.Timedelta(days=1)

    # Calculate End_Date based on the period
    End_Date = Start_date + pd.Timedelta(days=Period_Days)

    while True:
        if End_Date < min_date or End_Date > max_date:
            raise ValueError("End_Date is out of range.")
        try:
            # Try if a shareprice exists on selected End_Date
            Shareprice_End = Shareprice.loc[Shareprice['Date'] == End_Date, ISIN].values[0]
            break
        except IndexError:
            # Move to the next day if End_Date is not found
            End_Date += pd.Timedelta(days=1)

    # Calculate performance
    Performance = (Shareprice_End / Shareprice_Start) - 1

    return Performance

period_days= [1,2,3]

for date in df['Date']:
    for period in period_days:
        new_column_name = 'Perf_' + str(period) + '_Days'
        df.loc[df['Date'] == date, new_column_name] = perfcalc(date, period, df.loc[df['Date'] == date, 'Company'].values[0])

###Date Selection
df_max= df[(df['Date'] <= '2024-05-28')]
df_2015 = df[(df['Date'] >= '2015-09-21') & (df['Date'] <= '2024-05-28')]
df_2021 = df[(df['Date'] >= '2021-01-28') & (df['Date'] <= '2024-05-28')]

###Linear Regression

datasets = {
    "df_max": df_max,
    "df_2015": df_2015,
    "df_2021": df_2021
}

# Linear Regression Analysis
# Define independent variables
independent_vars = ['Volume of News','Cyber Attack', 'Data Security Management', 'Cyber Security', 'Data Breach','Perc. of Positive Sentiment']

# Define dependent variables
targets = ['Perf_1_Days', 'Perf_2_Days', 'Perf_3_Days']

# Initialize overall results list
overall_results = []

# Min-Max Normalization for each dataset
scaler = MinMaxScaler()

for dataset_name, dataset in datasets.items():
    # Extract independent and dependent variables for the current dataset
    X = dataset[independent_vars]
    X_normalized = scaler.fit_transform(X)

    for target in targets:
        y = dataset[target]# dependent variable

        # Split test and training data
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

        # Create & train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Save data for visualization if dataset is df_max and target is Perf_1_Days
        if dataset_name == "df_max" and target == "Perf_3_Days":
            visualization_data = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })

        # Collect results
        overall_results.append({
            'Dataset': dataset_name,
            'Target': target,
            'MSE': mean_squared_error(y_test, y_pred),
            'R2_Score': r2_score(y_test, y_pred),
            'Coefficients': [float(round(c, 4)) for c in model.coef_]
        })

# Prepare results table
table_data = [
    [
        result['Dataset'],
        result['Target'],
        result['MSE'],
        result['R2_Score'],
        result['Coefficients']
    ]
    for result in overall_results
]
headers = ["Dataset", "Target", "MSE", "R2_Score", "Coefficients"]

# Show table
print(tabulate(table_data, headers=headers, tablefmt="grid"))

###Visualization of prediction vs. acutal performance
plt.figure(figsize=(8, 6))
x_min, x_max = -0.5, 0.2
y_min, y_max = -0.5, 0.2
plt.scatter(visualization_data['Actual'], visualization_data['Predicted'], alpha=0.6, label='Predicted vs actual')
plt.plot([x_min, x_max], [x_min, x_max], color='red', linestyle='--', linewidth=2, label='Ideal fit')
plt.xlabel('Actual performance', fontsize=14)
plt.ylabel('Predicted performance', fontsize=14)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
#plt.title('Apple non-linear actual vs predicted performance', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
#plt.show()

file_path = os.path.join("Figures", "Apple_linear_predvsact.png")
plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


##Non-linear model

datasets = {
    "df_max": df_max,
    "df_2015": df_2015,
    "df_2021": df_2021
}

# Define independent variables
independent_vars = ['Volume of News', 'Cyber Attack', 'Data Security Management', 'Cyber Security', 'Data Breach', 'Perc. of Positive Sentiment']

# Define dependent variables
targets = ['Perf_1_Days', 'Perf_2_Days', 'Perf_3_Days']

# Initialize overall results list
overall_results = []

# Min-Max Normalization for each dataset
scaler = MinMaxScaler()


for dataset_name, dataset in datasets.items():
    # Extract independent and dependent variables for the current dataset
    X = dataset[independent_vars]
    X_normalized = scaler.fit_transform(X)

    for target in targets:
        y = dataset[target]  # dependent variable

        # Split test and training data
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

        # Create & train the Random Forest Regressor
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Save data for visualization if dataset is df_max and target is Perf_1_Days
        if dataset_name == "df_max" and target == "Perf_3_Days":
            visualization_data = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })

        # Collect results
        overall_results.append({
            'Dataset': dataset_name,
            'Target': target,
            'MSE': mean_squared_error(y_test, y_pred),
            'R2_Score': r2_score(y_test, y_pred),
            'Feature Importances': [float(round(imp, 4)) for imp in model.feature_importances_]
        })

# Prepare results table
table_data = [
    [
        result['Dataset'],
        result['Target'],
        result['MSE'],
        result['R2_Score'],
        result['Feature Importances']
    ]
    for result in overall_results
]
headers = ["Dataset", "Target", "MSE", "R2_Score", "Feature Importances"]

# Print results table
print(tabulate(table_data, headers=headers, tablefmt="grid"))

###Visualization of prediction vs. acutal performance
plt.figure(figsize=(8, 6))
x_min, x_max = -0.5, 0.2
y_min, y_max = -0.5, 0.2
plt.scatter(visualization_data['Actual'], visualization_data['Predicted'], alpha=0.6, label='Predicted vs actual')
plt.plot([x_min, x_max], [x_min, x_max], color='red', linestyle='--', linewidth=2, label='Ideal fit')
plt.xlabel('Actual performance', fontsize=14)
plt.ylabel('Predicted performance', fontsize=14)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
#plt.title('Apple non-linear actual vs predicted performance', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
#plt.show()

file_path = os.path.join("Figures", "Apple_Nonlinear_predvsact.png")
plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
