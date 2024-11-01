import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


file_path = './Data/Finance/Shareprice.xlsx'
Shareprice = pd.read_excel(file_path, sheet_name='Output')
Shareprice.replace(".", 0, inplace=True)

pd.set_option('display.max_columns', None)
#reset set_option pd.reset_option('display.max_columns')

ISINs = Shareprice.columns.tolist()[1:len(Shareprice.columns)]


def perfcalc(Start_date, Period_Days, ISIN):
    Shareprice_Start= Shareprice.loc[Shareprice['Date'] == Start_date, ISIN].values[0]
    Start_date = pd.to_datetime(Start_date)

    End_Date = Start_date + pd.Timedelta(days=Period_Days)
    End_Date = pd.to_datetime(End_Date)

    while True:
        try:
            # Try if a shareprice exists on selected end_date
            Shareprice_End = Shareprice.loc[Shareprice['Date'] == End_Date, ISIN].values[0]
            break  # Wenn erfolgreich, die Schleife beenden
        except IndexError:
            # if end_date doesn't exist, use the next available shareprice
            End_Date += pd.Timedelta(days=1)

    Shareprice_End = Shareprice.loc[Shareprice['Date'] == End_Date, ISIN].values[0]
    Performace = (Shareprice_End/Shareprice_Start)-1
    
    return Performace

print(Shareprice.loc[0,'US0378331005'])

x= '1999-12-31'
datum = pd.to_datetime(x)

y= str((datum + pd.Timedelta(days=3)).date())
print(y)
y= str((datum + pd.Timedelta(days=-3)).date())


#print(Shareprice.loc[Shareprice['Date'] == x, 'US0378331005'].values[0])


#print(perfcalc('2005-01-10',3,'US0378331005'))

#print(Shareprice.loc[0,'Date'])

for i in range(0,5):
    print(i)
    print(Shareprice.loc[i,'Date'])
    print(perfcalc(Shareprice.loc[i,'Date'],3,'US0378331005'))


