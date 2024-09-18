import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('Hello')


PATH = '/Users/fredericweiss/Documents/Konto_2024.xlsm'
accounts_areas = {
    'Ideeller Bereich': [31, 32, 35, 36, 39, 41, 42, 44, 47],
    'Vermögensverwaltung': [51, 52, 53, 55, 56],
    'Zweckbetrieb': [601, 605, 607, 608, 701, 709, 61, 65, 67, 68, 71, 79],
    'Wirtschaftlicher Geschäftsbetrieb': [806, 807, 809, 87, 88]
}

st.set_page_config(
    page_title="DLRG Reichenbach Finance Dashboard",
    page_icon="🏂",
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

def get_data(path):
    # Reading the excel file
    data = pd.read_excel(path, [0,1,5,7])
    sheet0 =  data[0] # accounts
    sheet1 =  data[1] # bills
    sheet5 =  data[5] # forecast
    sheet7 =  data[7] # planned

    # Dropping not filled rows from the bills dataframe
    sheet1 = sheet1[sheet1['Datum'].notna()]

    return sheet0, sheet1, sheet5, sheet7


def get_metric(df_bills, col):
    # Reading the profit from last row of the dataframe
    current_profit = df_bills.iloc[len(df_bills)-1,col]
    return round(current_profit,2)


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

    # Adding columns to the dataframe
    df['Diff'] = (df.Plan - df.Ist) * (-1)
    df['Prozent'] = round(df.Ist / df.Plan * 100, 2)

    # Setting up the plot
    fig, ax = plt.subplots(figsize=(5,2))

    colors = ['#CCE0AC' if df.loc[idx,'Plan'] < 0 else '#FF8A8A' for idx in df.index]

    # Plotting the data
    ax.barh(np.arange(len(df)), df.Ist.iloc[::-1], alpha=0.9, color=colors, height=0.5)
    ax.barh(np.arange(len(df)), df.Plan.iloc[::-1], alpha=0.2, color='grey', height=0.8)

    # Setting lables
    ax.set_yticks(np.arange(len(df)))
    labels1 = [account for account in accounts[::-1]]
    ax.set_yticklabels(labels1)

    # Setting annotation abount budget utilization
    labels2 = [f'{round(df.loc[account,"Ist"], 2)}€ / {round(df.loc[account,"Plan"], 2)}€' for account in accounts[::-1]]
    for i in np.arange(len(df)):
        plt.annotate(labels2[i], (0,i), xytext=(150,i-0.12), fontsize=4)

    # visual modifications
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.grid(axis='x')
    plt.tick_params(axis='both', which='major', labelsize=6)

    # Returning the figure
    return fig


def transactions(df_bills, account):
    df = df_bills[df_bills['Konto'] == account]
    df = df[['Beleg Nr.', 'Datum', 'Bezeichnung', 'Betrag+/-']]
    df.index = df['Beleg Nr.']
    df.drop(['Beleg Nr.'], inplace=True, axis=1)
    df['Betrag+/-'] = df['Betrag+/-']
    return df



# ------------- Preparation -------------
accounts, bills, forecast, planned = get_data(PATH)

# ------------- Sidebar elements -------------
st.sidebar.title('Filter options')

with st.sidebar:
    # Selectbox for the area
    area = st.selectbox('Bereich auswählen', ['Ideeller Bereich', 'Vermögensverwaltung', 'Zweckbetrieb', 'Wirtschaftlicher Geschäftsbetrieb'], index=3)

    # Selectbox for the account
    account = st.selectbox('Kono auswählen', planned.Konto, index=len(planned)-1)

    # Displaying the names for the different accounts
    st.markdown('##### Kontenplan')
    df_accounts = pd.DataFrame()
    df_accounts['Bezeichnung'] = planned.Bezeichnung
    df_accounts.index = planned.Konto
    st.table(df_accounts)

# ------------- Main Body -------------
cols = st.columns((1.5, 4.5, 2), gap='medium')

# ------------- First column -------------
with cols[0]:
    # Current profit
    profit = f'{get_metric(bills, 9)} €'
    st.metric(label='Aktueller Gewinn', value=profit)

    # Current wealth
    wealth = f'{get_metric(bills, 8)} €'
    st.metric(label='Aktuelles Vermögen', value=wealth)

    # Forecasted profit
    profit_forecast = f'{get_metric(forecast, 4)} €'
    st.metric(label='Aktueller Gewinn', value=profit_forecast)

# ------------- Second column -------------

# generate spending plots
accounts_selected = accounts_areas[area]
spendings_ideell = generate_spendings_chart(bills, planned, accounts_selected)

with cols[1]:
    st.pyplot(spendings_ideell)

# ------------- Third column -------------
with cols[2]:
    st.table(transactions(bills, account))
