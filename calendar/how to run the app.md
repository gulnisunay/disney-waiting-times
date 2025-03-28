# Calendar App

Comprehensive Streamlit app that uses a trained machine learning model to predict and visualize Disney attraction wait times. The app includes:

1.  **Interactive Monthly Calendar View**

    * Shows predicted crowd levels for each day using a color-coded system.
    * Colors range from green (low wait times) to red (high wait times).
    * Each day shows the predicted average wait time and crowding category.

2.  **Detailed Analysis Tools**

    * **Daily Wait Time Profile:** Shows how wait times vary throughout the day for your selected attraction.
    * **Attraction Comparison:** Compares wait times across different attractions at a specific hour.
    * **Personalized Tips:** Provides recommendations based on predictions, including best/worst times to visit.

3.  **User Controls**

    * Select year, month, and specific attraction.
    * Choose a specific day for detailed analysis.
    * Adjust time of day for comparison views.

4.  **Visualizations**

    * Color-coded calendar that mimics the style of your reference image.
    * Line chart showing wait time patterns throughout the day.
    * Bar chart comparing different attractions.

## How to Run the App

1.  Save the code to a file named `app.py`.
2.  Install required packages:

    ```bash
    pip install streamlit pandas numpy matplotlib joblib
    ```

3.  Make sure your model files are in the same directory:

    * `memory_efficient_model.pkl`
    * `scaler.pkl`
    * `model_features.pkl`

4.  Run the app with:

    ```bash
    streamlit run app.py
    ```

The app automatically extracts attraction names from your model features, so you don't need to manually specify them. It also implements intelligent caching to improve performance when users interact with the interface.
