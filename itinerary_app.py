# write everything in english

import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
import streamlit as st
from datetime import datetime, timedelta
# import ipywidgets as widgets
# from ipywidgets import interactive
# from IPython.display import display

# --- Load Data ---
def load_data():
    df = pd.read_csv("C:/Users/mateo/Documents/SYNTRA/2TIM/disney_csv/df_after2021.csv")
    entities = pd.read_csv("C:/Users/mateo/Documents/SYNTRA/2TIM/disney_csv/entities_extra.csv")
    metadata = pd.read_csv("C:/Users/mateo/Documents/SYNTRA/2TIM/disney_csv/metadata.csv")
    return df, entities, metadata  # edited absolute paths !!

df, entities_extra, metadata = load_data()

# --- Prepare ride names ---
entities_extra['short_name_snake'] = (
    entities_extra['short_name']
    .str.lower()
    .str.replace(' ', '_')
    .str.replace("'", "")
    .str.replace("&", "and")
    .str.replace("-", "_")
    .str.replace("__", "_")
    .str.strip()
)
ride_data = entities_extra[entities_extra['category_code'] == 'ride']
ride_names = ride_data['short_name_snake'].dropna().unique()
ride_names

# --- Clean waiting time data ---
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['date'] = df['datetime'].dt.date
df['weekday'] = df['datetime'].dt.day_name()

df_rides = df[
    df['attraction'].isin(ride_names) &
    df['SPOSTMIN'].notna() &
    (df['SPOSTMIN'] >= 0) &
    (df['SPOSTMIN'] < 300)
].copy()
# df_rides.info()
# df_rides  # 762991 rows × 6 columns

# --- Merge holiday info ---
metadata['DATE'] = pd.to_datetime(metadata['DATE'], errors='coerce').dt.date
metadata
df_rides = df_rides.merge(metadata[['DATE', 'HOLIDAYM']], left_on='date', right_on='DATE', how='left')
df_rides['HOLIDAYM'] = df_rides['HOLIDAYM'].fillna(0)
# df_rides.info()
# df_rides  # 762991 rows × 8 columns

# Function to create itinerary (same as before)
def make_itinerary(interesting, day_string, hourref, df_rides):
    day = datetime.strptime(day_string, '%Y-%m-%d').date()
    hourref2 = pd.to_datetime(hourref)

    # Filter the data based on conditions
    cond1 = df_rides['date'] == day
    cond2 = df_rides['datetime'] > hourref2
    cond3 = df_rides['attraction'].isin(interesting)

    df_filtered = df_rides[cond1 & cond2 & cond3]

    # Group by attraction and calculate the average wait time
    df5 = df_filtered.groupby('attraction')['SPOSTMIN'].agg('mean')
    df7 = df5.sort_values(ascending=True)

    # Sort the attractions by the average wait time
    interesting_sorted = df7.index.tolist()

    itinerary = []  # To store the final itinerary
    used_attractions = set()  # To keep track of which attractions have been added to the itinerary

    # Create hourlist based on the user's arrival time
    hourlist = [str((hourref2 + timedelta(hours=i)).hour) for i in range(7)]

    # Iterate through each hour and select the attraction with the lowest wait time
    for hour in hourlist:
        for att in interesting_sorted:
            if att not in used_attractions:
                # Filter for the selected attraction and hour
                df_temp = df_filtered[(df_filtered['attraction'] == att) & (df_filtered['datetime'].dt.hour == int(hour))]

                # If entries exist for the hour
                if not df_temp.empty:
                    # Calculate the average wait time for this hour
                    mean_wait_time = df_temp['SPOSTMIN'].mean()

                    # Add the attraction to the itinerary if its wait time is lower than the global average
                    if mean_wait_time < df_filtered['SPOSTMIN'].mean():
                        itinerary.append((att, hour, mean_wait_time))
                        used_attractions.add(att)
                        break  # Stop checking other attractions for this hour

    # After processing the optimal attractions, add remaining ones
    for att in interesting_sorted:
        if att not in used_attractions:
            df_temp = df_filtered[df_filtered['attraction'] == att]
            mean_wait_time = df_temp['SPOSTMIN'].mean()
            itinerary.append((att, "Remaining", mean_wait_time))

    return itinerary

# Streamlit App Interface
st.title("Theme Park Itinerary Generator")

# Date picker for arrival date
day_picker = st.date_input("Pick a Date", datetime(2021, 1, 14))

# Time picker for arrival time (from 7am to 11pm)
time_picker = st.time_input("Pick Arrival Time", datetime(2021, 1, 14, 10, 0).time())

# Multi-select for interesting attractions
attraction_picker = st.multiselect(
    "Select Interesting Attractions", options=ride_names, default=['haunted_mansion', 'dumbo']
)

# Button to generate itinerary
if st.button("Generate Itinerary"):
    day_string = day_picker.strftime('%Y-%m-%d')
    hourref = f"{day_string} {time_picker.strftime('%H:%M:%S')}"
    interesting = list(attraction_picker)

    itinerary = make_itinerary(interesting, day_string, hourref, df_rides)

    # Convert itinerary to DataFrame and display it
    itinerary_df = pd.DataFrame(itinerary, columns=["Attraction", "Hour", "Wait Time (minutes)"])

    # Display the itinerary DataFrame
    st.write(itinerary_df)
