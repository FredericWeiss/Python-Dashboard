import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

print('Hello')


PATH = '/Users/fredericweiss/Documents/Konto_2024.xlsm'
accounts_ideell = [31, 32, 35, 36, 39, 41, 42, 44, 47]
accounts_vermögen = [51, 52, 53, 55, 56]
accounts_zweck = [601, 605, 607, 608, 701, 709, 61, 65, 67, 68, 71, 79]
accounts_wirtschaftlich = [806, 807, 809, 87, 88]


st.set_page_config(
    page_title="DLRG Reichenbach Finance Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


def get_data(path):
    data = pd.read_excel(path, [0,1,5,7])
    sheet0 =  data[0] # accounts
    sheet1 =  data[1] # bills
    sheet5 =  data[5] # forecast
    sheet7 =  data[7] # planned

    # Dropping not filled rows from the bills dataframe
    sheet1 = sheet1[sheet1['Datum'].notna()]

    return sheet0, sheet1, sheet5, sheet7


def get_profit(df_bills):
    current_profit = df_bills.iloc[len(df_bills)-1,9]
    return round(current_profit,2)


def generate_spendings_chart(df_bills, accounts):
    df_transformed = df_bills.loc[:,['Konto','Betrag+/-']].dropna().groupby(['Konto'], as_index=False).sum()
    df_transformed.Konto = df_transformed.Konto.astype('string')
    df_transformed = df_transformed[df_transformed['Konto'].isin(accounts)]

    chart = alt.Chart(df_transformed).mark_bar().encode(
        x='Konto',
        y='Betrag+/-'
        )  

    return chart


accounts, bills, forecast, planned = get_data(PATH)
profit = get_profit(bills)
spendings_chart = generate_spendings_chart(bills, accounts_ideell)

# ------------- Sidebar elements -------------
st.sidebar.title('Filter options')

# ------------- Main Body -------------
cols = st.columns((1.5, 4.5, 2), gap='medium')

# ------------- First column -------------
with cols[0]:
    st.write(profit)

# ------------- Second column -------------
with cols[1]:
    st.altair_chart(spendings_chart, use_container_width=True)

