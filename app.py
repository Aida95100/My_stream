import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

st.title("This is my first streamlit app")

if st.button("Click me please"):
    st.balloons()


lower_year, higher_year = st.slider('Pick a year', min_value=1960, max_value=2010, value=(1965, 1970))


nb_displayed = st.selectbox("Number of country to display", [3, 5, 10, 20, 30])

df = pd.read_csv('data/CO2_per_capita.csv', sep=';')


def top_n_emitters(df, start_year=2008, end_year=2011, nb_displayed=10):
    # Filtrer les années
    df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    # Moyenne des émissions par pays
    df_mean = df.groupby('Country Name')['CO2 Per Capita (metric tons)'].mean().reset_index()
    # Trier les valeurs
    df_mean = df_mean.sort_values(by='CO2 Per Capita (metric tons)', ascending=False).head(nb_displayed)
    # Créer la figure avec Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=df_mean['Country Name'],
            y=df_mean['CO2 Per Capita (metric tons)']
        )
    ])
    # Mettre en forme
    fig.update_layout(
        title=f'Top {nb_displayed} CO2 emitters from {start_year} to {end_year}',
        xaxis_title='Country',
        yaxis_title='CO2 per capita (metric tons)'
    )
    return fig
   


fig = top_n_emitters(df, lower_year, higher_year, nb_displayed)

st.plotly_chart(fig)

df = df.dropna(subset= ['CO2 Per Capita (metric tons)'])
df = df.sort_values(by="Year")

fig = px.scatter_geo(df,
                     animation_frame="Year",
                     locations="Country Code",
                     color='CO2 Per Capita (metric tons)',
                     size='CO2 Per Capita (metric tons)',
                     projection="natural earth")

st.plotly_chart(fig)



dft = pd.read_csv('data/geo_data.csv', sep=',')

dft = dft[['Continent_Name', 'Three_Letter_Country_Code']]

dft_merge = pd.merge(dft, df, left_on ='Three_Letter_Country_Code', right_on = 'Country Code' )

dft_merge.head(5)

dft_merge = dft_merge.dropna(subset= ['CO2 Per Capita (metric tons)'])

dft_merge = dft_merge.sort_values(by="Year")

fig = px.scatter_geo(dft_merge,
                     animation_frame="Year",
                     locations="Three_Letter_Country_Code",
                     color='Continent_Name',
                     size= 'CO2 Per Capita (metric tons)',
                     projection="natural earth")

st.plotly_chart(fig)

def top_n_emitters_v2(dft_merge, start_year, end_year, top_n):
      
    dft_merge = dft_merge[(dft_merge['Year'] >= start_year) & (dft_merge['Year'] <= end_year)]

    dft_merge = dft_merge.groupby(['Country Name','Continent_Name'])['CO2 Per Capita (metric tons)'].mean().reset_index()

    dft_merge = dft_merge.sort_values(by='CO2 Per Capita (metric tons)', ascending=False).head(top_n)


    fig = px.bar(dft_merge,
                 x='Country Name',
                 y='CO2 Per Capita (metric tons)',
                 color ='Continent_Name',
                 labels={'x':'Country Name', 'y':'Co2_per_capita'})
    
    st.plotly_chart(fig)




dft_merge = dft_merge.dropna(subset= ['CO2 Per Capita (metric tons)'])

dft_merge = dft_merge.sort_values(by="Year")

fig = px.choropleth(dft_merge, 
                    locations="Three_Letter_Country_Code", 
                    color ='Continent_Name', 
                    animation_frame='Year', 
                    projection ='natural earth')

st.plotly_chart(fig)





