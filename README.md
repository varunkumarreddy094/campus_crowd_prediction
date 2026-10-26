# Campus Crowd Navigator (Streamlit app)

1. Install: `pip install -r requirements.txt`
2. Keep `campus_crowd_dataset.csv` in this folder (a copy is included).
3. Run: `streamlit run app.py`  (opens http://localhost:8501)

The model trains on first start (about 5 seconds). Use the sidebar to pick a day and hour and to switch exam, event, class and weather
inputs; the Crowd forecast and Route planner tabs update. Replace the walking distances in `EDGES` (top of app.py) with real campus measurements.
