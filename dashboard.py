import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Accounts in germand accounting for clubs
accounts_areas = {
    'Ideeller Bereich': [31, 32, 35, 36, 39, 41, 42, 44, 47],
    'Vermögensverwaltung': [51, 52, 53, 55, 56],
    'Zweckbetrieb': [601, 605, 607, 608, 701, 709, 61, 65, 67, 68, 71, 79],
    'Wirtschaftlicher Geschäftsbetrieb': [806, 807, 809, 87, 88]
}

# Base Configurations of the app
st.set_page_config(
    page_title="DLRG Reichenbach Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded")

# ------------- CSS Styles -------------
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #F5F7F8;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
"""
, unsafe_allow_html=True)

# ------------- Functions -------------

# Function that extracts the data from the file uploaded in the file picker
def get_data(file):
    # Reading the excel file
    data = pd.read_excel(file, [0,1,5,7])

    # Reading the excel file
    sheet0 =  data[0] # Overview over all accounts
    sheet1 =  data[1] # bills currently booked
    sheet5 =  data[5] # forecast of spendings, incomes for this year
    sheet7 =  data[7] # plan as budgeted

    # Dropping not filled rows from the bills dataframe
    sheet1 = sheet1[sheet1['Datum'].notna()]

    return sheet0, sheet1, sheet5, sheet7


# Function to extract several metrics fromt the bills dataframe, depending on the colums
def get_metric(df_bills, col):
    # Reading the profit from last row of the dataframe
    current_profit = df_bills.iloc[len(df_bills)-1,col]
    return round(current_profit,2)


# Function that determines the profit / loss for the selected area out of the 4 areas in germand accounting for clubs
def area_sum(df_bills, accounts):
    # Filtering the dataframe after area and summing up the bills
    sum = np.sum(df_bills[df_bills['Konto'].isin(accounts)]['Betrag+/-'])
    return round(sum, 2)


# Function that determines the profit / loss for a specific account
def account_sum(df_bills, acc):
    # Filtering the dataframe after account and summing up the bills
    acc_sum = np.sum(df_bills[df_bills['Konto'] == acc]['Betrag+/-'])

    # Determining a label if the sum is positive or negative
    if acc_sum >= 0:
        status = f'Einnahmen Konto {acc}'
    else:
        status = f'Ausgaben Konto {acc}'

    return round(acc_sum,2), status


# Function to generate a chart with all spendings and incomes for each account in the selected area
def generate_spendings_chart(df_bills, df_planned, accounts):
    # Setting up a dataframe
    df = pd.DataFrame()

    # Merging spendings / incomes and plan values on accounts
    df['Konto'] = accounts
    df = pd.merge(df, df_planned, on='Konto', how='left')
    df = pd.merge(df, df_bills.loc[:,['Konto','Betrag+/-']].dropna().groupby(['Konto'], as_index=False).sum(), on='Konto', how='left')

    df.drop(['Vorzeichen', 'Konto'], axis=1, inplace=True) # Dropping unncessary columns
    df = df.rename(columns={'Betrag+/-': 'Ist'}) # Renaming columns
    df.index = accounts # Set index
    df = df.fillna(0) # Replace nan values with 0

    # Adding columns to the dataframe comparing planned with actual values
    df['Diff'] = (df.Plan - df.Ist) * (-1)
    df['Prozent'] = round(df.Ist / df.Plan * 100, 2)

    # Setting up the plot
    fig, ax = plt.subplots(figsize=(5,2))

    # Sorting the dataframe for plotting
    df = df.sort_index(ascending=False)

    # Determining the colors: Red for spending and green for incomes
    colors = ['#CCE0AC' if df.loc[idx,'Plan'] > 0 else '#FF8A8A' for idx in df.index]

    # Plotting the data
    ax.barh(np.arange(len(df)), df.Ist, alpha=0.9, color=colors, height=0.5)
    ax.barh(np.arange(len(df)), df.Plan, alpha=0.2, color='grey', height=0.8)

    # Setting lables
    ax.set_yticks(np.arange(len(df)))
    labels1 = [account for account in df.index]
    ax.set_yticklabels(labels1)

    # Setting annotation abount budget utilization
    labels2 = [f'{round(df.loc[account,"Ist"], 2)}€ / {round(df.loc[account,"Plan"], 2)}€' for account in df.index]
    for i in np.arange(len(df)):
        plt.annotate(labels2[i], (0,i), xytext=(150,i-0.12), fontsize=4)

    # visual modifications
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.grid(axis='x')
    plt.tick_params(axis='both', which='major', labelsize=6)

    return fig


# Funtion to plot the development for proft and wealth over the year
def profit_chart (df_bills, df_forecast):
    fig, ax = plt.subplots()

    # Extracting necessary columns from bills dataframe
    df = df_bills[['Datum', 'Geldvermögen', 'Gewinn 2024']]
    df = df.sort_values(by='Datum')

    # Plotting the data
    ax.plot(df['Datum'], df['Gewinn 2024'], color='green')
    ax.plot(df['Datum'], df['Geldvermögen'], color='blue')

    # Creating a second dataframe and saving the current date, wealth, and profit in the first row
    df2 = pd.DataFrame(columns=['Datum', 'Geldvermögen', 'Gewinn'])
    df2.loc[0,'Datum'] = df.loc[len(df)-1,'Datum']
    df2.loc[0,'Geldvermögen'] = df.loc[len(df)-1,'Geldvermögen']
    df2.loc[0,'Gewinn'] = df.loc[len(df)-1,'Gewinn 2024']

    df_forecast = df_forecast.sort_values(by='Datum')

    # Adding the forecast to the second dataframe and calculating wealth and profits
    new_row = 1
    for idx in range(len(df_forecast)):
        df2.loc[new_row,'Datum'] = df_forecast.iloc[idx,2]
        df2.loc[new_row,'Gewinn'] = df2.loc[new_row-1,'Gewinn'] + df_forecast.iloc[idx,3]
        df2.loc[new_row,'Geldvermögen'] = df2.loc[new_row-1,'Geldvermögen'] + df_forecast.iloc[idx,3]
        new_row += 1

    # Plotting the forecase
    ax.plot(df2['Datum'], df2['Gewinn'], linestyle='--', color='green')
    ax.plot(df2['Datum'], df2['Geldvermögen'], linestyle='--', color='blue')

    # Visual adjustments
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.grid(axis='y')

    return fig


# Function that extracts a table with all transactions for the selected account
def transactions(df_bills, account):
    # Filtering the bills dataframe for the selected account and required columns
    df = df_bills[df_bills['Konto'] == account]
    df = df[['Beleg Nr.', 'Datum', 'Bezeichnung', 'Betrag+/-']]

    # Alternations for displaying outupts
    df.index = df['Beleg Nr.']
    df.drop(['Beleg Nr.'], inplace=True, axis=1)
    df['Datum'] = df['Datum'].dt.strftime('%d/%m/%Y')

    return df


# Funtion to diplay forecasted expenses and incomes in a table
def forecast_table(df_forecast):
    # Extracting data from the forecase dataframe and modifying it slightly
    df_forecast = df_forecast.sort_values(by='Datum')
    df = df_forecast[['Einnahme / Ausgabe', 'Datum', 'Betrag']]
    df['Datum'] = df['Datum'].dt.strftime('%d/%m/%Y')
    df.index = df_forecast['Konto']

    return df


# Funtion that prepares the data to extract top 5 lists
def top5_prep (df_bills, df_planned):
    accounts = [31, 32, 35, 36, 41, 42, 44, 47, 51, 52, 56, 601, 605, 607, 608, 701, 709, 61, 65, 67, 68, 71, 79, 806, 807, 809, 87, 88]

    # Setting up a dataframe
    df = pd.DataFrame()

    # Merging spendings / incomes and plan values on accounts
    df['Konto'] = accounts
    df = pd.merge(df, df_planned, on='Konto', how='left')
    df = pd.merge(df, df_bills.loc[:,['Konto','Betrag+/-']].dropna().groupby(['Konto'], as_index=False).sum(), on='Konto', how='left')

    df.drop(['Vorzeichen'], axis=1, inplace=True) # Dropping unncessary columns
    df = df.rename(columns={'Betrag+/-': 'Ist'}) # Renaming columns
    df = df.fillna(0) # Replace nan values with 0

    df['Diff'] = (df.Plan - df.Ist) * (-1)

    return df


# Function to display top 5 accounts with lower incomes as planned
def lower_income(df_prep):
    accounts_income = [31, 32, 35, 36, 39, 51, 52, 601, 605, 607, 608, 701, 709, 806, 807, 809]

    # Filtering the dataframe
    df = df_prep[df_prep['Diff'] < 0]
    df = df[df['Konto'].isin(accounts_income)]

    # Sorting and extracting top 5
    df = df.sort_values(by='Diff', ascending=True)
    df.index = df['Konto']
    df = df[['Bezeichnung', 'Diff']][0:5]

    return df


# Function to display top 5 accounts with lower expanditures as planned
def lower_exp(df_prep):
    accounts_exp = [41, 42, 44, 47, 56, 61, 65, 67, 68, 71, 79, 87, 88]

    # Filtering the dataframe
    df = df_prep[df_prep['Diff'] > 0]
    df = df[df['Konto'].isin(accounts_exp)]

    # Sorting and extracting top 5
    df = df.sort_values(by='Diff', ascending=False)
    df.index = df['Konto']
    df = df[['Bezeichnung', 'Diff']][0:5]

    return df


# Function to display top 5 accounts with higher incomes as planned
def higher_income(df_prep):
    accounts_income = [31, 32, 35, 36, 39, 51, 52, 601, 605, 607, 608, 701, 709, 806, 807, 809]

    # Filtering the dataframe
    df = df_prep[df_prep['Diff'] > 0]
    df = df[df['Konto'].isin(accounts_income)]

    # Sorting and extracting top 5
    df = df.sort_values(by='Diff', ascending=False)
    df.index = df['Konto']
    df = df[['Bezeichnung', 'Diff']][0:5]

    return df


# Function to display top 5 accounts with higher expanditures as planned
def higher_exp(df_prep):
    accounts_exp = [41, 42, 44, 47, 56, 61, 65, 67, 68, 71, 79, 87, 88]

    # Filtering the dataframe
    df = df_prep[df_prep['Diff'] < 0]
    df = df[df['Konto'].isin(accounts_exp)]

    # Sorting and extracting top 5
    df = df.sort_values(by='Diff', ascending=True)
    df.index = df['Konto']
    df = df[['Bezeichnung', 'Diff']][0:5]


# ------------- Sidebar elements -------------
with st.sidebar:

    # Display File Uploader
    uploaded_file = st.file_uploader('Konto-Excel auswählen')
    if uploaded_file is not None:
        accounts, bills, forecast, planned = get_data(uploaded_file)

    st.markdown('## Filter Options')

    # Selectbox for the area
    area = st.selectbox('Bereich auswählen', ['Ideeller Bereich', 'Vermögensverwaltung', 'Zweckbetrieb', 'Wirtschaftlicher Geschäftsbetrieb'], index=3)

    # Selectbox for the account
    account = st.selectbox('Kono auswählen', planned.Konto, index=len(planned)-1)

    # Displaying the names for the different accounts
    st.markdown('##### Kontenplan')
    df_accounts = pd.DataFrame()
    df_accounts['Bezeichnung'] = planned.Bezeichnung
    df_accounts.index = planned.Konto
    df_accounts = df_accounts.sort_index(ascending=True)
    st.table(df_accounts)

# ------------- Main Body -------------
st.markdown('# DLRG Reichenbach Finanzen')
cols = st.columns((1.5, 4, 2.5), gap='small')
accounts_selected = accounts_areas[area] # Saving a subsets of all accounts based on the selected area

# ------------- First column -------------
with cols[0]:

    st.markdown('##### Kennzahlen')

    # Current profit
    profit = f'{get_metric(bills, 9)} €'
    st.metric(label='Aktueller Gewinn', value=profit)

    # Forecasted profit
    profit_forecast = f'{get_metric(forecast, 4)} €'
    st.metric(label='Prognistizierter Gewinn', value=profit_forecast)

    # Current wealth
    wealth = f'{get_metric(bills, 8)} €'
    st.metric(label='Aktuelles Vermögen', value=wealth)

    # Forecasted wealth
    wealth_forecast = f'{get_metric(forecast, 5)} €'
    st.metric(label='Prognistizierters Vermögen', value=wealth_forecast)

    # Sum over all incomes and expenses per account
    sum_area = area_sum(bills, accounts_selected)
    st.metric(label=f'Bilanz im Berich {area}', value=f'{sum_area} €')

    # Displaying the status of selected accont
    sum, account_status = account_sum(bills, account)
    st.metric(label=account_status, value=f'{sum} €')


# ------------- Second column -------------

# Generate spendings plot
spendings = generate_spendings_chart(bills, planned, accounts_selected)

# Generate profits plot
profits = profit_chart(bills, forecast)

with cols[1]:
    # Plot spendings plot
    st.markdown(f'##### Soll-Ist-Vergleich')
    st.pyplot(spendings)

    # Plot profits plot
    st.markdown(f'##### Gewinn/- Vermögensentwicklung')
    st.pyplot(profits)

# ------------- Third column -------------
with cols[2]:
    # Display table with transactions of selected account
    st.markdown(f'##### Buchungen Konto {account}')
    st.dataframe(transactions(bills, account), use_container_width=True)

    # Display table with forecast
    st.markdown(f'##### Prognose')
    st.dataframe(forecast_table(forecast), use_container_width=True)


# ------------- Second Page -------------
st.markdown('### Top 5')
cols = st.columns(4, gap='small')
prep = top5_prep(bills, planned)# Prepare data for top 5 lists

# ------------- First column -------------
# top 5 lower incomes as planned
with cols[0]:
    st.markdown('##### Unter Plan Einnahmen')
    lower_income_accounts = lower_income(prep)
    st.dataframe(lower_income_accounts, use_container_width=True)

# ------------- Second column -------------
# top 5 higher incomes as planned
with cols[1]:
    st.markdown('##### Über Plan Einnahmen')
    higher_income_accounts = higher_income(prep)
    st.dataframe(higher_income_accounts, use_container_width=True)

# ------------- Third column -------------
# top 5 lower expenditures as planned
with cols[2]:
    st.markdown('##### Unter Plan Ausgaben')
    lower_exp_accounts = lower_exp(prep)
    st.dataframe(lower_exp_accounts, use_container_width=True)

# ------------- Fourth column -------------
# top 5 higher expenditures as planned
with cols[3]:
    st.markdown('##### Über Plan Ausgaben')
    higher_exp_accounts = higher_exp(prep)
    st.dataframe(higher_exp_accounts, use_container_width=True)
