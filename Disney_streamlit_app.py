import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Disney Wait Time Predictor", page_icon="🏰", layout="wide"
)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calendar
from datetime import datetime


# --- Title ---
st.title("🎢 Disney World Waiting Time Dashboard")
st.markdown("Analyze average posted wait times for ride attractions based on weekday and holiday status.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("../disney-world/data/all_waiting_times.csv")
    entities = pd.read_csv("../disney-world/data/overview data/entities_extra.csv")
    metadata = pd.read_csv("../disney-world/data/overview data/metadata.csv")

    # df = pd.read_csv("C:/Disney_Waiting_Times/disney-waiting-times/all_waiting_times_extracted/all_waiting_times.csv")
    # entities = pd.read_csv("C:/Disney_Waiting_Times/disney-waiting-times/data/overview data/entities_extra.csv")
    # metadata = pd.read_csv("C:/Disney_Waiting_Times/disney-waiting-times/data/overview data/metadata.csv")
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
df_rides = df_rides.merge(metadata[['DATE', 'HOLIDAYM', 'WDWSEASON']], left_on='date', right_on='DATE', how='left')
df_rides['HOLIDAYM'] = df_rides['HOLIDAYM'].fillna(0)
df_rides['WDWSEASON'] = df_rides['WDWSEASON'].fillna("No Holiday")

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
heatmap_data = df_rides.groupby(['weekday', 'HOLIDAYM'])['SPOSTMIN'].mean().unstack()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(weekday_order)

if selected_chart == "Bar chart":
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

# --- Most Waited Attractions ---
st.subheader("📊 Attractions with the most Average Wait Time")
avg_by_attraction = df_rides.groupby('attraction')['SPOSTMIN'].mean().sort_values(ascending=False).head(10)
fig4, ax4 = plt.subplots(figsize=(8, 5))
avg_by_attraction.plot(kind='barh', color='lightseagreen', ax=ax4)
ax4.set_title("Top 10 Most Waited Ride Attractions")
ax4.set_xlabel("Avg Wait Time (min)")
ax4.set_ylabel("Attraction")
ax4.invert_yaxis()
st.pyplot(fig4)

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

# Group by WDWSEASON instead of HOLIDAYM
df_attraction_holiday = (
    df_rides[df_rides['attraction'] == selected_attraction]
    .groupby('WDWSEASON')['SPOSTMIN']
    .mean()
    .sort_values(ascending=False)
)

fig3, ax3 = plt.subplots(figsize=(8, 4))
df_attraction_holiday.plot(kind='bar', color='coral', ax=ax3)
ax3.set_title(f"Avg Wait Time by HOLIDAYM for {selected_attraction}")
#ax3.set_xlabel("HOLIDAYM")
ax3.set_xlabel("Holiday Name")
ax3.set_ylabel("Avg Wait Time (min)")
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True)

# ✅ Make x-labels readable
ax3.tick_params(axis='x', rotation=30)
for label in ax3.get_xticklabels():
    label.set_horizontalalignment('right')

st.pyplot(fig3)

# --- Recommendation System ---
st.subheader("🔎 Recommended Attractions Based on Low Wait Times")
user_wait_pref = st.slider("Maximum preferred wait time (minutes)", min_value=0, max_value=85, value=30)
avg_waits = df_rides.groupby('attraction')['SPOSTMIN'].mean()
recommended = avg_waits[avg_waits <= user_wait_pref].sort_values().head(20)

if not recommended.empty:
    st.write("Here are some attractions with low average wait times:")
    st.dataframe(recommended.reset_index().rename(columns={'SPOSTMIN': 'Avg Wait Time (min)'}))
else:
    st.write("No attractions found under your preferred wait time.")


# Define popular Disney attractions
attractions = [
    "Space Mountain",
    "Big Thunder Mountain",
    "Splash Mountain",
    "Pirates Of Caribbean",
    "Haunted Mansion",
    "It's A Small World",
    "Peter Pan's Flight",
    "Seven Dwarfs Mine Train",
    "Dumbo",
    "Jungle Cruise",
    "Flight Of Passage",
    "Expedition Everest",
    "Toy Story Mania",
    "Slinky Dog Dash",
    "Rock 'n' Roller Coaster",
]

# Define colors for busyness levels (from light green to dark red)
color_map = {
    "Very Low": "#c1e1c1",  # Light green
    "Low": "#c1e7c1",  # Lighter green
    "Below Average": "#c8e8ae",  # Light yellow-green
    "Average": "#f7e29c",  # Light yellow
    "Above Average": "#f6c592",  # Light orange
    "Busy": "#f0a27a",  # Darker orange
    "Very Busy": "#e57b65",  # Light red
    "Extremely Busy": "#d65f5f",  # Dark red
}


def get_busyness_category(wait_time):
    """Convert predicted wait time to busyness category"""
    if wait_time < 10:
        return "Very Low"
    elif wait_time < 20:
        return "Low"
    elif wait_time < 30:
        return "Below Average"
    elif wait_time < 40:
        return "Average"
    elif wait_time < 50:
        return "Above Average"
    elif wait_time < 60:
        return "Busy"
    elif wait_time < 75:
        return "Very Busy"
    else:
        return "Extremely Busy"


def get_color_for_wait_time(wait_time):
    """Get color for wait time"""
    category = get_busyness_category(wait_time)
    return color_map[category]


def predict_wait_time(date, attraction, hour=12):
    """Generate a simulated wait time based on realistic factors"""
    # Base wait time on month (busier in summer and holidays)
    month_factor = 1.0
    if date.month in [6, 7, 8]:  # Summer
        month_factor = 1.5
    elif date.month in [11, 12]:  # Holidays
        month_factor = 1.4
    elif date.month in [3, 4]:  # Spring break
        month_factor = 1.3

    # Weekend adjustment
    weekend_factor = 1.4 if date.weekday() >= 5 else 1.0

    # Hour adjustment (busier midday)
    hour_factor = 1.0
    if 11 <= hour <= 14:  # Midday
        hour_factor = 1.4
    elif 15 <= hour <= 17:  # Afternoon
        hour_factor = 1.3
    elif hour < 10 or hour > 19:  # Early morning or late evening
        hour_factor = 0.6

    # Attraction popularity (based on name)
    attraction_factor = 1.0
    very_popular = ["Seven Dwarfs Mine Train", "Flight Of Passage", "Slinky Dog Dash"]
    popular = ["Space Mountain", "Peter Pan", "Rock", "Expedition Everest"]
    less_popular = ["Small World", "Dumbo", "Carousel"]

    if any(pop in attraction for pop in very_popular):
        attraction_factor = 1.8
    elif any(pop in attraction for pop in popular):
        attraction_factor = 1.4
    elif any(pop in attraction for pop in less_popular):
        attraction_factor = 0.7

    # Base wait time between 15-45 minutes
    base_wait = 25

    # Calculate wait time
    wait_time = (
        base_wait * month_factor * weekend_factor * hour_factor * attraction_factor
    )

    # Add some randomness (±10%)
    # We use a deterministic seed to get consistent results for the same inputs
    seed = int(
        str(date.year)
        + str(date.month).zfill(2)
        + str(date.day).zfill(2)
        + str(hour).zfill(2)
    )
    np.random.seed(seed)
    random_factor = 0.9 + 0.2 * np.random.random()

    return wait_time * random_factor


def generate_calendar_data(year, month, selected_attraction):
    """Generate calendar data for the specified month"""
    # Get the number of days in the month
    _, num_days = calendar.monthrange(year, month)

    # Generate data for each day
    days_data = []
    for day in range(1, num_days + 1):
        date = datetime(year, month, day)
        wait_time = predict_wait_time(date, selected_attraction)
        category = get_busyness_category(wait_time)
        color = get_color_for_wait_time(wait_time)

        days_data.append(
            {
                "day": day,
                "wait_time": wait_time,
                "category": category,
                "color": color,
                "date": date,
            }
        )

    return days_data


def main():
    st.title("🏰 Disney Wait Time Predictor")

    # Display an informative introduction
    st.markdown("""
    This app predicts how busy Disney attractions will be on different days throughout the year.
    The simulation is based on realistic factors like:
    
    - 🗓️ Time of year (summer, holidays)
    - 📅 Day of week (weekends vs. weekdays)
    - 🕒 Time of day (morning, midday, evening)
    - 🎢 Attraction popularity
    
    The colors indicate expected wait times:
    - 🟢 Green: Low wait times
    - 🟡 Yellow: Average wait times
    - 🟠 Orange: High wait times
    - 🔴 Red: Very high wait times
    """)

    # Create sidebar for inputs
    st.sidebar.header("Settings")

    # Current year and available attractions
    current_year = datetime.now().year
    years = list(range(current_year, current_year + 2))

    # Input selectors
    selected_year = st.sidebar.selectbox("Select Year:", years)
    selected_month = st.sidebar.selectbox(
        "Select Month:", range(1, 13), format_func=lambda x: calendar.month_name[x]
    )
    selected_attraction = st.sidebar.selectbox(
        "Select Attraction:", sorted(attractions)
    )

    # Generate the calendar data
    calendar_data = generate_calendar_data(
        selected_year, selected_month, selected_attraction
    )

    # Display the calendar
    st.subheader(
        f"{calendar.month_name[selected_month]} {selected_year} - {selected_attraction}"
    )

    # Create Streamlit columns for days of week header
    weekdays = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]

    # Display the weekday headers using Streamlit columns
    cols = st.columns(7)
    for i, day in enumerate(weekdays):
        with cols[i]:
            st.markdown(
                f"<div style='text-align: center; font-weight: bold;'>{day}</div>",
                unsafe_allow_html=True,
            )

    # Calculate first day of month (0=Monday, 6=Sunday)
    first_day = datetime(selected_year, selected_month, 1).weekday()

    # Calculate number of days in the month
    _, days_in_month = calendar.monthrange(selected_year, selected_month)

    # Calculate number of weeks needed
    num_weeks = (first_day + days_in_month + 6) // 7

    # Create calendar grid
    day_counter = 1

    # Create each week row
    for week in range(num_weeks):
        # Create columns for each day in the week
        cols = st.columns(7)

        # Fill in the days
        for weekday in range(7):
            with cols[weekday]:
                if (week == 0 and weekday < first_day) or (day_counter > days_in_month):
                    # Empty cell
                    st.markdown(
                        "<div style='height: 100px; background-color: #f9f9f9; border-radius: 5px;'></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Get data for this day
                    day_data = calendar_data[day_counter - 1]
                    wait_time = day_data["wait_time"]
                    category = day_data["category"]
                    color = day_data["color"]

                    # Create day cell with explicit HTML closing tags
                    st.markdown(
                        f"""
                    <div style='height: 100px; background-color: {color}; border-radius: 5px; padding: 5px; display: flex; flex-direction: column;'>
                        <div style='font-weight: bold; text-align: center;'>{day_counter}</div>
                        <div style='font-size: 12px; text-align: center;'>{category}</div>
                        <div style='font-size: 14px; text-align: center; margin-top: auto;'>{wait_time:.1f} min</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    day_counter += 1

    # Add a legend
    st.subheader("Crowding Levels")

    # Use columns for the legend
    legend_cols = st.columns(4)
    i = 0
    for category, color in color_map.items():
        col_idx = i % 4
        with legend_cols[col_idx]:
            st.markdown(
                f"""
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border-radius: 3px;'></div>
                <div>{category}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        i += 1

    # Add additional analysis section
    st.subheader("Daily Wait Time Analysis")

    # Allow user to select a specific date
    selected_day = st.slider("Select Day:", 1, days_in_month)

    # Create a date object for the selected date
    selected_date = datetime(selected_year, selected_month, selected_day)

    # Create a daily profile for the selected attraction
    hours = range(8, 23)  # 8 AM to 10 PM
    wait_times = [
        predict_wait_time(selected_date, selected_attraction, hour) for hour in hours
    ]

    # Create two columns for the detailed view
    col1, col2 = st.columns(2)

    with col1:
        # Display the daily wait time profile
        st.write(f"### Wait Time Profile for {selected_date.strftime('%A, %B %d, %Y')}")

        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.plot(hours, wait_times, marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Predicted Wait Time (minutes)")
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h}:00" for h in hours])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_ylim(bottom=0)

        # Color the area under the curve based on wait times
        for i in range(len(hours) - 1):
            plt.fill_between(
                [hours[i], hours[i + 1]],
                [wait_times[i], wait_times[i + 1]],
                color=get_color_for_wait_time(wait_times[i]),
                alpha=0.3,
            )

        st.pyplot(fig)

    with col2:
        # Compare with other attractions
        st.write("### Compare with Other Attractions")

        # Select top attractions for comparison (limit to 10 for readability)
        popular_attractions = sorted(attractions)[:10]

        # Select hour for comparison
        comparison_hour = st.slider("Select Hour for Comparison:", 8, 22, 12)

        # Calculate wait times for popular attractions
        comparison_wait_times = [
            predict_wait_time(selected_date, attr, comparison_hour)
            for attr in popular_attractions
        ]

        # Create a horizontal bar chart
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()

        # Create colorful bars based on wait times
        bars = ax.barh(popular_attractions, comparison_wait_times)

        # Color each bar based on the wait time
        for i, bar in enumerate(bars):
            bar.set_color(get_color_for_wait_time(comparison_wait_times[i]))

        ax.set_xlabel("Predicted Wait Time (minutes)")
        ax.set_title(f"Predicted Wait Times at {comparison_hour}:00")
        ax.grid(True, axis="x", linestyle="--", alpha=0.7)

        # Highlight the selected attraction
        for i, attr in enumerate(popular_attractions):
            if attr == selected_attraction:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2)

        st.pyplot(fig)

    # Add some tips based on the predictions
    st.subheader("Tips for Your Visit")

    # Calculate the average wait time for the day
    avg_wait_time = sum(wait_times) / len(wait_times)

    # Find the best time to visit
    best_hour_idx = wait_times.index(min(wait_times))
    best_hour = hours[best_hour_idx]

    # Find the worst time to visit
    worst_hour_idx = wait_times.index(max(wait_times))
    worst_hour = hours[worst_hour_idx]


    # Display tips
    st.markdown(f"""
    Based on our predictions for {selected_attraction} on {selected_date.strftime("%A, %B %d, %Y")}:
    
    - **Average Wait Time**: {avg_wait_time:.1f} minutes
    - **Best Time to Visit**: Around {best_hour}:00 ({wait_times[best_hour_idx]:.1f} min wait)
    - **Busiest Time**: Around {worst_hour}:00 ({wait_times[worst_hour_idx]:.1f} min wait)
    
    **General Tips:**
    - {"Consider visiting on a different day if possible, as this attraction is predicted to be quite busy." if avg_wait_time > 60 else "This seems like a good day to visit this attraction!" if avg_wait_time < 30 else "Expect moderate crowds throughout the day."}
    - Try to visit popular attractions early in the morning or during meal times.
    - Consider using Disney's Lightning Lane or Genie+ services if available.
    """)

    # Add footer
    st.markdown("---")
    st.markdown("© 2025 Disney Wait Time Predictor")


    # --- FastPass Optimizer ---
def assign_fastpass_manual(date, selected_rides):
    """
    Assign fixed FastPass times to selected attractions.
    """
    fastpass_times = ["10:00", "13:00", "16:00"]
    assignments = {}

    for i, attraction in enumerate(selected_rides):
        if i >= len(fastpass_times):
            break  # Limit to 3 slots
        assignments[attraction] = fastpass_times[i]

    return assignments

# --- FastPass UI Integration ---
st.sidebar.header("🎟️ FastPass Optimization")
use_fastpass = st.sidebar.checkbox("Use FastPass Optimization")

if use_fastpass:
    # Tarih seçimi
    selected_fp_year = st.sidebar.selectbox("📅 FP Year", list(range(datetime.now().year, datetime.now().year + 2)))
    selected_fp_month = st.sidebar.selectbox("📅 FP Month", range(1, 13), format_func=lambda x: calendar.month_name[x])
    _, fp_days_in_month = calendar.monthrange(selected_fp_year, selected_fp_month)
    selected_fp_day = st.sidebar.slider("📅 FP Day", 1, fp_days_in_month)
    selected_fp_date = datetime(selected_fp_year, selected_fp_month, selected_fp_day)

    # Atraksiyon seçimi (çoklu seçim)
    st.sidebar.markdown("🎢 Select up to 3 attractions for FastPass:")
    selected_fp_rides = st.sidebar.multiselect("Choose Attractions", options=attractions, max_selections=3)

    # Gösterim
    st.subheader(f"🎟️ Your FastPass Plan for {selected_fp_date.strftime('%A, %B %d, %Y')}")
    if selected_fp_rides:
        plan = assign_fastpass_manual(selected_fp_date, selected_fp_rides)
        st.table(pd.DataFrame(list(plan.items()), columns=["Attraction", "FastPass Time Slot"]))
    else:
        st.info("Please select up to 3 attractions to generate your FastPass plan.")


# --- FastPass Benefit Comparison ---
def compare_fastpass_benefit(selected_date, selected_rides, fastpass_times=["10:00", "13:00", "16:00"], fastpass_wait_time=5):
    normal_total = 0
    fastpass_total = 0

    comparison_data = []

    for ride, time_str in zip(selected_rides, fastpass_times):
        hour = int(time_str.split(":")[0])
        normal_wait = predict_wait_time(selected_date, ride, hour)
        fastpass_wait = fastpass_wait_time

        normal_total += normal_wait
        fastpass_total += fastpass_wait

        comparison_data.append({
            "Attraction": ride,
            "Time": time_str,
            "Normal Wait (min)": round(normal_wait, 1),
            "FastPass Wait (min)": fastpass_wait,
            "Time Saved (min)": round(normal_wait - fastpass_wait, 1)
        })

    return normal_total, fastpass_total, pd.DataFrame(comparison_data)


# --- UI for Comparison ---
st.subheader("⏱️ FastPass vs Normal Visit Comparison")

if use_fastpass and selected_fp_rides:
    total_normal, total_fastpass, compare_df = compare_fastpass_benefit(selected_fp_date, selected_fp_rides)

    st.write(f"### For {selected_fp_date.strftime('%A, %B %d, %Y')}")

    st.dataframe(compare_df)

    st.markdown(f"""
    - 🕒 **Total Normal Wait Time**: `{total_normal:.1f}` minutes  
    - 🎟️ **Total With FastPass**: `{total_fastpass}` minutes  
    - 💡 **Time Saved**: `{total_normal - total_fastpass:.1f}` minutes
    """)

    # Visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Normal", "FastPass"], [total_normal, total_fastpass], color=["orangered", "seagreen"])
    ax.set_ylabel("Total Wait Time (minutes)")
    ax.set_title("⏱️ Total Wait Time Comparison")
    ax.bar_label(bars, padding=3)
    st.pyplot(fig)


if __name__ == "__main__":
    main()

# ====================================================================

# Function to create itinerary2
def make_itinerary2(interesting, day_string, hourref):
    day = datetime.strptime(day_string, '%Y-%m-%d').date()
    hourref2 = pd.to_datetime(hourref)

    # Filter the data based on conditions
    cond1 = df_rides['date'] == day
    cond2 = df_rides['datetime'] > hourref2
    cond3 = df_rides['attraction'].isin(interesting)

    df_filtered2 = df_rides[cond1 & cond2 & cond3]

    # Group by attraction and calculate the average wait time
    df5 = df_filtered2.groupby('attraction')['SPOSTMIN'].agg('mean')
    df7 = df5.sort_values(ascending=True)

    # Sort the attractions by the average wait time
    interesting_sorted = df7.index.tolist()

    itinerary2 = []  # To store the final itinerary
    used_attractions = set()  # To keep track of which attractions have been added to the itinerary

    # Create hourlist based on the user's arrival time, ensuring it finishes at 23:59
    start_hour = hourref2.hour
    end_hour = 23  # End of the day (23:59)

    hourlist = [str(start_hour + i) for i in range(0, end_hour - start_hour + 1)]

    # Iterate through each hour and select the attraction with the lowest wait time
    for hour in hourlist:
        for att in interesting_sorted:
            if att not in used_attractions:
                # Filter for the selected attraction and hour
                df_temp = df_filtered2[
                    (df_filtered2['attraction'] == att) & (df_filtered2['datetime'].dt.hour == int(hour))]

                # If entries exist for the hour
                if not df_temp.empty:
                    # Calculate the average wait time for this hour
                    mean_wait_time = df_temp['SPOSTMIN'].mean().round()

                    # Add the attraction to the itinerary if its wait time is lower than the global average
                    if mean_wait_time < df_filtered2['SPOSTMIN'].mean():
                        itinerary2.append((att, hour, mean_wait_time))
                        used_attractions.add(att)
                        break  # Stop checking other attractions for this hour

    # After processing the optimal attractions, add remaining ones
    for att in interesting_sorted:
        if att not in used_attractions:
            df_temp = df_filtered2[df_filtered2['attraction'] == att]
            if not df_temp.empty:
                mean_wait_time = df_temp['SPOSTMIN'].mean().round()
                itinerary2.append((att, "Remaining", mean_wait_time))

    return itinerary2


# Streamlit App Interface
st.title("Itinerary Generator for 1 day in the past, maximum 5 favourite attractions")

# Date picker for arrival date
day_picker = st.date_input("Pick a Date",
    value=df_rides['date'].min(),  # Default value
    min_value=df_rides['date'].min(),  # Min possible date
    max_value=df_rides['date'].max() )

# Time picker for arrival time (only selecting hour, no minutes)
time_picker = st.selectbox("Pick Arrival Hour", options=[f"{i}:00" for i in range(8, 24)])

# Multi-select for interesting attractions
attraction_picker = st.multiselect(
    "Select Interesting Attractions", options=ride_names, default=['haunted_mansion', 'dumbo']
)

# Button to generate itinerary
if st.button("Generate Itinerary"):
    day_string = day_picker.strftime('%Y-%m-%d')
    # Correct the formatting of the hourref
    hourref = f"{day_string} {time_picker}"  # No additional ":00:00"

    interesting = list(attraction_picker)

    itinerary2 = make_itinerary2(interesting, day_string, hourref)

    # Convert itinerary to DataFrame and display it
    itinerary_df = pd.DataFrame(itinerary2, columns=["Attraction", "Hour", "Wait Time (minutes)"])

    # Display the itinerary DataFrame
    st.write(itinerary_df)

    # Check for missing attractions (those that are in 'interesting' but not in 'itinerary_df')
    included_attractions = set(itinerary_df['Attraction'].unique())
    missing_attractions = [att for att in interesting if att not in included_attractions]

    # Show message for missing attractions
    if missing_attractions:
        st.write(f"Following attractions have no data for the selected day: {', '.join(missing_attractions)}.")

# ===========================================================