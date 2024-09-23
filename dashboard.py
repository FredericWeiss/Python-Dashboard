import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PATH = '/Users/fredericweiss/Documents/Konto_2024.xlsm'
accounts_areas = {
    'Ideeller Bereich': [31, 32, 35, 36, 39, 41, 42, 44, 47],
    'Vermögensverwaltung': [51, 52, 53, 55, 56],
    'Zweckbetrieb': [601, 605, 607, 608, 701, 709, 61, 65, 67, 68, 71, 79],
    'Wirtschaftlicher Geschäftsbetrieb': [806, 807, 809, 87, 88]
}

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


def area_sum(df_bills, accounts):
    sum = np.sum(df_bills[df_bills['Konto'].isin(accounts)]['Betrag+/-'])
    return round(sum, 2)


def account_sum(df_bills, acc):
    acc_sum = np.sum(df_bills[df_bills['Konto'] == acc]['Betrag+/-'])
    if acc_sum >= 0:
        status = f'Einnahmen Konto {acc}'
    else:
        status = f'Ausgaben Konto {acc}'

    return round(acc_sum,2), status


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


def profit_chart (df_bills, df_forecast):
    fig, ax = plt.subplots()

    df = df_bills[['Datum', 'Geldvermögen', 'Gewinn 2024']]
    df = df.sort_values(by='Datum')

    ax.plot(df['Datum'], df['Gewinn 2024'], color='green')
    ax.plot(df['Datum'], df['Geldvermögen'], color='blue')

    df2 = pd.DataFrame(columns=['Datum', 'Geldvermögen', 'Gewinn'])
    df2.loc[0,'Datum'] = df.loc[len(df)-1,'Datum']
    df2.loc[0,'Geldvermögen'] = df.loc[len(df)-1,'Geldvermögen']
    df2.loc[0,'Gewinn'] = df.loc[len(df)-1,'Gewinn 2024']

    df_forecast = df_forecast.sort_values(by='Datum')

    new_row = 1
    for idx in range(len(df_forecast)):
        df2.loc[new_row,'Datum'] = df_forecast.iloc[idx,2]
        df2.loc[new_row,'Gewinn'] = df2.loc[new_row-1,'Gewinn'] + df_forecast.iloc[idx,3]
        df2.loc[new_row,'Geldvermögen'] = df2.loc[new_row-1,'Geldvermögen'] + df_forecast.iloc[idx,3]
        new_row += 1

    ax.plot(df2['Datum'], df2['Gewinn'], linestyle='--', color='green')
    ax.plot(df2['Datum'], df2['Geldvermögen'], linestyle='--', color='blue')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.grid(axis='y')

    return fig


def transactions(df_bills, account):
    df = df_bills[df_bills['Konto'] == account]
    df = df[['Beleg Nr.', 'Datum', 'Bezeichnung', 'Betrag+/-']]
    df.index = df['Beleg Nr.']
    df.drop(['Beleg Nr.'], inplace=True, axis=1)
    df['Betrag+/-'] = df['Betrag+/-']
    df['Datum'] = df['Datum'].dt.strftime('%d/%m/%Y')
    return df


def forecast_table(df_forecast):
    df_forecast = df_forecast.sort_values(by='Datum')
    df = df_forecast[['Einnahme / Ausgabe', 'Datum', 'Betrag']]
    df['Datum'] = df['Datum'].dt.strftime('%d/%m/%Y')
    df.index = df_forecast['Konto']

    return df


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


def lower_income(df_prep):
    accounts_income = [31, 32, 35, 36, 39, 51, 52, 601, 605, 607, 608, 701, 709, 806, 807, 809]
    df = df_prep[df_prep['Diff'] < 0]
    df = df[df['Konto'].isin(accounts_income)]
    df = df.sort_values(by='Diff', ascending=True)
    df = df[['Konto', 'Bezeichnung', 'Diff']][0:5]

    return df


def lower_exp(df_prep):
    accounts_exp = [41, 42, 44, 47, 56, 61, 65, 67, 68, 71, 79, 87, 88]
    df = df_prep[df_prep['Diff'] > 0]
    df = df[df['Konto'].isin(accounts_exp)]
    df = df.sort_values(by='Diff', ascending=False)
    df = df[['Konto', 'Bezeichnung', 'Diff']][0:5]

    return df


def higher_income(df_prep):
    accounts_income = [31, 32, 35, 36, 39, 51, 52, 601, 605, 607, 608, 701, 709, 806, 807, 809]
    df = df_prep[df_prep['Diff'] > 0]
    df = df[df['Konto'].isin(accounts_income)]
    df = df.sort_values(by='Diff', ascending=False)
    df = df[['Konto', 'Bezeichnung', 'Diff']][0:5]

    return df


def higher_exp(df_prep):
    accounts_exp = [41, 42, 44, 47, 56, 61, 65, 67, 68, 71, 79, 87, 88]
    df = df_prep[df_prep['Diff'] < 0]
    df = df[df['Konto'].isin(accounts_exp)]
    df = df.sort_values(by='Diff', ascending=True)
    df = df[['Konto', 'Bezeichnung', 'Diff']][0:5]

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
st.markdown('# DLRG Reichenbach Finanzen')
cols = st.columns((1.5, 4, 2.5), gap='small')
accounts_selected = accounts_areas[area]

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

# generate spending plots
spendings = generate_spendings_chart(bills, planned, accounts_selected)
profits = profit_chart(bills, forecast)

with cols[1]:
    st.markdown(f'##### Soll-Ist-Vergleich')
    st.pyplot(spendings)
    st.markdown(f'##### Gewinn/- Vermögensentwicklung')
    st.pyplot(profits)

# ------------- Third column -------------
with cols[2]:

    st.markdown(f'##### Buchungen Konto {account}')
    st.dataframe(transactions(bills, account), use_container_width=True)

    st.markdown(f'##### Prognose')
    st.dataframe(forecast_table(forecast), use_container_width=True)


# ------------- Second Page -------------
st.markdown('### Top 5')
cols = st.columns(4, gap='small')
prep = top5_prep(bills, planned)

# ------------- First column -------------
with cols[0]:
    st.markdown('##### Unter Plan Einnahmen')
    lower_income_accounts = lower_income(prep)
    st.dataframe(lower_income_accounts, use_container_width=True)

# ------------- Second column -------------
with cols[1]:
    st.markdown('##### Über Plan Einnahmen')
    higher_income_accounts = higher_income(prep)
    st.dataframe(higher_income_accounts, use_container_width=True)

# ------------- Third column -------------
with cols[2]:
    st.markdown('##### Unter Plan Ausgaben')
    lower_exp_accounts = lower_exp(prep)
    st.dataframe(lower_exp_accounts, use_container_width=True)
# ------------- Fourth column -------------
with cols[3]:
    st.markdown('##### Über Plan Ausgaben')
    higher_exp_accounts = higher_exp(prep)
    st.dataframe(higher_exp_accounts, use_container_width=True)
