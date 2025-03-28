import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Disney Wait Time Predictor", page_icon="🏰", layout="wide"
)

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


if __name__ == "__main__":
    main()
