
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import numpy as np


# --- Title ---
st.title("🎢 Disney World Wait Time Dashboard")
st.markdown("Analyze average posted wait times for ride attractions based on weekday and holiday status.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Disney_Waiting_Times/disney-waiting-times/all_waiting_times_extracted/all_waiting_times.csv")
    entities = pd.read_csv("C:/Disney_Waiting_Times/disney-waiting-times/data/overview data/entities_extra.csv")
    metadata = pd.read_csv("C:/Disney_Waiting_Times/disney-waiting-times/data/overview data/metadata.csv")
    return df, entities, metadata

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


# --- Merge holiday info ---
metadata['DATE'] = pd.to_datetime(metadata['DATE'], errors='coerce').dt.date
df_rides = df_rides.merge(metadata[['DATE', 'HOLIDAYM']], left_on='date', right_on='DATE', how='left')
df_rides['HOLIDAYM'] = df_rides['HOLIDAYM'].fillna(0)

# --- Sidebar Filters ---
st.sidebar.header("📌 Filters")
selected_day = st.sidebar.selectbox("Select a weekday", df_rides['weekday'].unique())
selected_holiday = st.sidebar.selectbox("Select HOLIDAYM value", sorted(df_rides['HOLIDAYM'].unique()))
selected_chart = st.sidebar.selectbox("Select chart type", ["Heatmap", "Bar chart", "Line chart"])
selected_attraction = st.sidebar.selectbox("Select a ride attraction", sorted(df_rides['attraction'].unique()))
date_range = st.sidebar.date_input("Select date range", value=(df_rides['datetime'].min(), df_rides['datetime'].max()))

# --- Apply filters ---
df_rides = df_rides[(df_rides['datetime'] >= pd.to_datetime(date_range[0])) & (df_rides['datetime'] <= pd.to_datetime(date_range[1]))]

# --- Filtered Data ---
df_filtered = df_rides[(df_rides['weekday'] == selected_day) & (df_rides['HOLIDAYM'] == selected_holiday)]
df_attraction = df_rides[df_rides['attraction'] == selected_attraction]

# --- Show Average Wait Time ---
avg_wait = df_filtered['SPOSTMIN'].mean()
st.metric(label=f"Average Wait Time on {selected_day} (HOLIDAYM={selected_holiday})", value=f"{avg_wait:.2f} min")

# --- Chart Selection ---
st.subheader(f"📊 {selected_chart} of Average Wait Time by Weekday & HOLIDAYM")
heatmap_data = df_rides.groupby(['weekday', 'HOLIDAYM'])['SPOSTMIN'].mean().unstack()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(weekday_order)

if selected_chart == "Heatmap":
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.5, ax=ax)
    plt.title("Avg Posted Wait Time (min)")
    st.pyplot(fig)

elif selected_chart == "Bar chart":
    avg_by_day = df_rides.groupby('weekday')['SPOSTMIN'].mean().reindex(weekday_order)
    fig, ax = plt.subplots()
    avg_by_day.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel('Avg Wait Time (min)')
    ax.set_title('Average Posted Wait Time by Weekday')
    st.pyplot(fig)

elif selected_chart == "Line chart":
    avg_by_day = df_rides.groupby('weekday')['SPOSTMIN'].mean().reindex(weekday_order)
    fig, ax = plt.subplots()
    avg_by_day.plot(kind='line', marker='o', ax=ax, color='orange')
    ax.set_ylabel('Avg Wait Time (min)')
    ax.set_title('Average Posted Wait Time by Weekday')
    st.pyplot(fig)

# --- Attraction-specific chart ---
st.subheader(f"🎡 Wait Time Trend for {selected_attraction}")
df_attraction_day_avg = df_attraction.groupby('date')['SPOSTMIN'].mean()
fig2, ax2 = plt.subplots(figsize=(10, 4))
df_attraction_day_avg.plot(ax=ax2, color='green')
ax2.set_title(f"Daily Avg Posted Wait Time - {selected_attraction}")
ax2.set_xlabel("Date")
ax2.set_ylabel("Avg Wait Time (min)")
ax2.grid(True)
st.pyplot(fig2)

# --- Holiday-Specific Analysis for Selected Attraction ---
st.subheader(f"📈 Holiday Impact on Wait Times for {selected_attraction}")
df_attraction_holiday = df_attraction.groupby('HOLIDAYM')['SPOSTMIN'].mean()
fig3, ax3 = plt.subplots(figsize=(8, 4))
df_attraction_holiday.plot(kind='bar', color='coral', ax=ax3)
ax3.set_title(f"Avg Wait Time by HOLIDAYM for {selected_attraction}")
ax3.set_xlabel("HOLIDAYM")
ax3.set_ylabel("Avg Wait Time (min)")
ax3.grid(True)
st.pyplot(fig3)

# --- Least Waited Attractions ---
st.subheader("📊 Attractions with the Lowest Average Wait Time")
avg_by_attraction = df_rides.groupby('attraction')['SPOSTMIN'].mean().sort_values().head(10)
fig4, ax4 = plt.subplots(figsize=(8, 5))
avg_by_attraction.plot(kind='barh', color='lightseagreen', ax=ax4)
ax4.set_title("Top 10 Least Waited Ride Attractions")
ax4.set_xlabel("Avg Wait Time (min)")
ax4.set_ylabel("Attraction")
ax4.invert_yaxis()
st.pyplot(fig4)

# --- Recommendation System ---
st.subheader("🔎 Recommended Attractions Based on Low Wait Times")
user_wait_pref = st.slider("Maximum preferred wait time (minutes)", min_value=0, max_value=60, value=30)
avg_waits = df_rides.groupby('attraction')['SPOSTMIN'].mean()
recommended = avg_waits[avg_waits <= user_wait_pref].sort_values().head(10)

if not recommended.empty:
    st.write("Here are some attractions with low average wait times:")
    st.dataframe(recommended.reset_index().rename(columns={'SPOSTMIN': 'Avg Wait Time (min)'}))
else:
    st.write("No attractions found under your preferred wait time.")

# --- Interpolation Function ---
def interpolate_daily_posted(df, attractions, start="09:00", end="21:00", freq="30min"):
    time_range = pd.date_range(start=start, end=end, freq=freq).time
    records = []

    for attraction in attractions:
        df_attr = df[df['attraction'] == attraction]
        for date in df_attr['date'].unique():
            df_day = df_attr[df_attr['date'] == date].copy()
            df_day = df_day.sort_values('datetime')
            if len(df_day) < 2 or df_day['datetime'].isnull().any():
                continue

            df_day['minutes'] = df_day['datetime'].dt.hour * 60 + df_day['datetime'].dt.minute
            try:
                interpolated = np.interp(
                    [t.hour * 60 + t.minute for t in time_range],
                    df_day['minutes'],
                    df_day['SPOSTMIN']
                )
                for t, val in zip(time_range, interpolated):
                    records.append({
                        'date': date,
                        'attraction': attraction,
                        'time': t.strftime('%H:%M'),
                        'SPOSTMIN': val
                    })
            except Exception as e:
                print(f"⚠️ Interpolation error on {attraction} {date}: {e}")
    return pd.DataFrame(records)

# --- Interpolation usage ---
if st.checkbox("🧪 Generate Interpolated Table"):
    interesting = ['space_mountain', 'splash_mountain', 'seven_dwarfs_mine_train']
    df_interpolated = interpolate_daily_posted(df_rides, interesting)
    st.success("✅ Interpolation completed!")
    st.dataframe(df_interpolated.head(20))

# --- Greedy itinerary generation ---
def generate_itinerary(df_interp, date, start_time, attractions, max_rides=5, walk_time=10):
    visited = set()
    plan = []
    current_time = pd.to_datetime(f"{date} {start_time}")
    end_time = pd.to_datetime(f"{date} 21:00")
    ride_duration = pd.Timedelta(minutes=20)

    while len(visited) < max_rides and current_time <= end_time:
        time_str = current_time.time().strftime('%H:%M')
        options = df_interp[
            (df_interp['date'] == date) &
            (df_interp['time'] == time_str) &
            (~df_interp['attraction'].isin(visited))
        ]
        if options.empty:
            break
        best = options.sort_values('SPOSTMIN').iloc[0]
        plan.append({
            'time': time_str,
            'attraction': best['attraction'],
            'wait_time': int(best['SPOSTMIN'])
        })
        visited.add(best['attraction'])
        current_time += pd.Timedelta(minutes=best['SPOSTMIN']) + ride_duration + pd.Timedelta(minutes=walk_time)

    return pd.DataFrame(plan)

# --- Arrival time and single recommendation ---
st.subheader("📅 Select a Day & Arrival Time")

# Warn if interpolated data is not available
if 'df_interpolated' not in locals() or df_interpolated.empty:
    st.warning("⚠️ Please run the interpolation step first.")
else:
    chosen_date = st.date_input("Choose a day", value=pd.to_datetime(df_interpolated['date'].max()))
    chosen_time = st.time_input("Choose an arrival time", value=pd.to_datetime("10:00").time())

    # Filter records matching selected day and time
    matching = df_interpolated[
        (df_interpolated['date'] == chosen_date) &
        (df_interpolated['time'] == chosen_time.strftime('%H:%M'))
    ]

    if matching.empty:
        st.warning("🚫 No wait time data available for the selected day/time.")
    else:
        st.success("Here are the predicted wait times at your arrival:")
        st.dataframe(matching.sort_values('SPOSTMIN'))

        # Greedy: pick the attraction with the shortest wait time
        shortest = matching.sort_values('SPOSTMIN').iloc[0]
        st.markdown(f"🎯 **Recommended first ride:** `{shortest['attraction']}` with estimated wait time: `{int(shortest['SPOSTMIN'])} min`")

def generate_itinerary(df_interp, date, start_time, attractions, max_rides=5, walk_time=10):
    visited = set()
    plan = []
    current_time = pd.to_datetime(f"{date} {start_time}")
    end_time = pd.to_datetime(f"{date} 21:00")  # Park closing time
    ride_duration = pd.Timedelta(minutes=20)    # Average ride duration

    while len(visited) < max_rides and current_time <= end_time:
        time_str = current_time.time().strftime('%H:%M')
        options = df_interp[
            (df_interp['date'] == date) &
            (df_interp['time'] == time_str) &
            (~df_interp['attraction'].isin(visited))
        ]

        if options.empty:
            break

        # Select the attraction with the shortest wait time
        best = options.sort_values('SPOSTMIN').iloc[0]
        plan.append({
            'time': time_str,
            'attraction': best['attraction'],
            'wait_time': int(best['SPOSTMIN'])
        })

        visited.add(best['attraction'])
        # Next time = current time + wait + ride duration + walk time
        current_time += pd.Timedelta(minutes=best['SPOSTMIN']) + ride_duration + pd.Timedelta(minutes=walk_time)

    return pd.DataFrame(plan)

# --- Full itinerary generation ---
if st.checkbox("📍 Generate Full Day Itinerary"):
    if 'df_interpolated' not in locals() or df_interpolated.empty:
        st.warning("⚠️ Please run the interpolation step first.")
    else:
        selected_max = st.slider("Maximum number of attractions", 1, 10, 5)
        arrival_time = st.time_input("Arrival Time", value=pd.to_datetime("10:00").time())
        itinerary = generate_itinerary(df_interpolated, chosen_date, arrival_time, interesting, max_rides=selected_max)

        if not itinerary.empty:
            st.success("✅ Suggested Itinerary")
            st.dataframe(itinerary)
        else:
            st.warning("No suitable itinerary could be generated.")

# --- Raw data preview ---
with st.expander("📄 View raw data"):
    st.dataframe(df_filtered.head(50))
