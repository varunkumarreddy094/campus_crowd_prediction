"""Campus Crowd Navigator - Streamlit app.

Run:  streamlit run app.py
Needs campus_crowd_dataset.csv in the same folder.
Trains the XGBoost model from the project notebook on first start (about 5 seconds), then answers what-if questions.
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Campus Crowd Navigator", page_icon=":world_map:", layout="wide")

CSV = "campus_crowd_dataset.csv"
LEVELS = ["Low", "Medium", "High"]
COLORS = ["#1E9E69", "#E9A91A", "#D8443B"]
T_LOW, T_HIGH = 0.20, 0.42
SEED = 42
# tuned hyper-parameters found in the notebook (forward-chaining CV)
XGB_PARAMS = dict(subsample=0.9, n_estimators=300, min_child_weight=3, max_depth=6, learning_rate=0.05, colsample_bytree=0.7)
WALK_M_PER_MIN = 78

EDGES = [  # illustrative walking distances in metres - replace with real campus measurements
    ("Main_Gate", "Junction_A", 100), ("Junction_A", "Library", 80), ("Junction_A", "Canteen", 90),
    ("Junction_A", "Admin_Block", 120), ("Library", "Canteen", 60), ("Library", "Main_Lecture_Block", 110),
    ("Canteen", "Main_Lecture_Block", 100), ("Admin_Block", "Junction_B", 70),
    ("Junction_B", "Main_Lecture_Block", 130), ("Junction_B", "Gymnasium", 90),
    ("Gymnasium", "Bus_Shuttle_Stop", 140), ("Main_Lecture_Block", "Bus_Shuttle_Stop", 120),
    ("Bus_Shuttle_Stop", "Hostel", 150), ("Junction_B", "Hostel", 160), ("Canteen", "Bus_Shuttle_Stop", 200),
]
POS = {"Main_Gate": (0, 3), "Junction_A": (1.2, 3), "Library": (2.2, 4.2), "Canteen": (2.2, 1.8),
       "Admin_Block": (2.9, 2.9), "Main_Lecture_Block": (3.8, 3), "Junction_B": (3.3, 4.7),
       "Gymnasium": (4.6, 5.0), "Bus_Shuttle_Stop": (5.6, 3.0), "Hostel": (5.4, 4.6)}
ZONES = ["Library", "Canteen", "Main_Lecture_Block", "Gymnasium", "Bus_Shuttle_Stop"]
PLACES = ["Main_Gate", "Hostel", "Admin_Block"] + ZONES
pretty = lambda n: n.replace("_", " ")


def to_level(r):
    return np.digitize(np.round(np.asarray(r, dtype=float), 4), [T_LOW, T_HIGH])


@st.cache_resource(show_spinner="Training the crowd model on the dataset...")
def load_everything():
    raw = pd.read_csv(CSV, parse_dates=["timestamp"])
    d = raw.sort_values(["zone_id", "timestamp"]).reset_index(drop=True).copy()
    d["date"] = d.timestamp.dt.normalize()
    d["ratio"] = d.crowd_count / d.zone_capacity
    d["level"] = to_level(d.ratio)
    g = d.groupby("zone_id")
    # history from past slots only (H = 1 hour). rolling_3slot_avg in the CSV contains the target, so it is not used.
    d["lag_0"] = g["ratio"].shift(1); d["lag_1"] = g["ratio"].shift(2); d["lag_2"] = g["ratio"].shift(3)
    d["lag_roll_mean"] = d[["lag_0", "lag_1", "lag_2"]].mean(axis=1)
    d["lag_trend"] = d.lag_0 - d.lag_1
    d["lag_yday"] = g["ratio"].shift(12)
    d["class_prev"] = g["is_class_in_session"].shift(1).fillna(0).astype(int)
    d["class_next"] = g["is_class_in_session"].shift(-1).fillna(0).astype(int)
    d["class_changeover"] = ((d.class_prev != d.is_class_in_session) | (d.class_next != d.is_class_in_session)).astype(int)
    d["day_type_holiday"] = (d.day_type == "Holiday").astype(int)
    d["dow_num"] = d.timestamp.dt.dayofweek
    d["is_weekend"] = (d.dow_num >= 5).astype(int)
    d["slot_idx"] = d.timestamp.dt.hour - 8
    d["hour_sin"] = np.sin(2 * np.pi * d.timestamp.dt.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.timestamp.dt.hour / 24)
    d["weather_Rain"] = (d.weather_condition == "Rain").astype(int)
    d["weather_Cloudy"] = (d.weather_condition == "Cloudy").astype(int)
    zd = pd.get_dummies(d.zone_name, prefix="zone").astype(int)
    d = pd.concat([d, zd], axis=1)
    feats = (["lag_0", "lag_1", "lag_2", "lag_roll_mean", "lag_trend", "lag_yday",
              "is_class_in_session", "class_prev", "class_next", "class_changeover", "is_exam_period", "is_event_day",
              "day_type_holiday", "is_weekend", "slot_idx", "hour_sin", "hour_cos", "dow_num", "weather_Rain", "weather_Cloudy"]
             + list(zd.columns))
    ok = d.dropna(subset=["lag_0", "lag_1", "lag_2", "lag_yday"]).copy()
    dates = np.sort(ok.date.unique())
    n_tr, n_va = int(len(dates) * 0.70), int(len(dates) * 0.15)
    train_dates = dates[:n_tr]; test_dates = dates[n_tr + n_va:]
    tr = ok[ok.date.isin(train_dates)]
    model = XGBClassifier(objective="multi:softprob", random_state=SEED, n_jobs=2, eval_metric="mlogloss",
                          tree_method="hist", **XGB_PARAMS).fit(tr[feats], tr.level)
    te = ok[ok.date.isin(test_dates)]
    f1 = None
    from sklearn.metrics import f1_score, accuracy_score
    p = model.predict(te[feats]); f1 = (f1_score(te.level, p, average="macro"), accuracy_score(te.level, p))
    level_mid = tr.groupby("level").ratio.mean().reindex([0, 1, 2]).values
    return dict(d=ok, feats=feats, model=model, level_mid=level_mid, train_dates=set(pd.to_datetime(train_dates)),
                test_dates=set(pd.to_datetime(test_dates)), scores=f1, dates=[pd.Timestamp(x) for x in dates])


@st.cache_data
def graph_edges():
    G = nx.Graph()
    for u, v, w in EDGES:
        G.add_edge(u, v, dist=w)
    return G


def set_weights(G, cong, lam):
    for u, v, dct in G.edges(data=True):
        dct["w"] = dct["dist"] * (1 + lam * (cong.get(u, 0) + cong.get(v, 0)) / 2)


def path_stats(G, path, occ):
    dist = expo = 0.0
    for u, v in zip(path[:-1], path[1:]):
        w = G[u][v]["dist"]; dist += w; expo += w * (occ.get(u, 0) + occ.get(v, 0)) / 2
    return dist, expo


def draw_map(G, levels, occ, paths=None, title=""):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    nx.draw_networkx_edges(G, POS, ax=ax, edge_color="#c9d0dc", width=2.5)
    colors, labels = [], {}
    for n in G.nodes:
        if n in ZONES:
            colors.append(COLORS[levels[n]]); labels[n] = f"{pretty(n)}\n{occ[n]*100:.0f}%"
        else:
            colors.append("#8d98ab"); labels[n] = pretty(n)
    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=colors, node_size=1500, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(G, POS, ax=ax, labels=labels, font_size=7, font_color="black")
    for (p, col, ls, w) in (paths or []):
        nx.draw_networkx_edges(G, POS, ax=ax, edgelist=list(zip(p[:-1], p[1:])), edge_color=col, width=w, style=ls)
    ax.set_title(title, fontsize=10); ax.axis("off"); fig.tight_layout()
    return fig


M = load_everything()
d, feats, model = M["d"], M["feats"], M["model"]

st.title("Campus Crowd Navigator")
st.caption("Predicts the crowd level of five campus zones for the next hour and recommends less crowded walking routes. "
           "Simulated campus data; walking distances are illustrative.")

# ---- sidebar: choose the moment and edit what is known about it ----
with st.sidebar:
    st.header("When")
    pool = sorted(M["dates"])
    day = st.selectbox("Day", pool, index=len(pool) - 8,
                       format_func=lambda x: f"{x:%a %d %b %Y}" + ("  (unseen by model)" if x in M["test_dates"] else "  (training day)"))
    slots = sorted(d.time_slot.unique())
    slot = st.select_slider("Time slot", options=slots, value="18:00-19:00" if "18:00-19:00" in slots else slots[0])
    st.header("What-if")
    st.caption("Change what the model is told about this hour. The prediction updates.")
    base = d[(d.date == day) & (d.time_slot == slot)].set_index("zone_name")
    over_exam = st.toggle("Exam period", bool(base.is_exam_period.iloc[0]))
    over_event = st.toggle("Event on campus", bool(base.is_event_day.iloc[0]))
    over_class = st.toggle("Classes in session", bool(base.is_class_in_session.iloc[0]))
    weather = st.radio("Weather", ["Clear", "Cloudy", "Rain"], index=["Clear", "Cloudy", "Rain"].index(base.weather_condition.iloc[0]), horizontal=True)
    holiday = st.toggle("Holiday", bool(base.day_type_holiday.iloc[0]))

X = base[feats].copy().astype(float)
X["is_exam_period"] = float(over_exam); X["is_event_day"] = float(over_event); X["is_class_in_session"] = float(over_class)
X["weather_Rain"] = float(weather == "Rain"); X["weather_Cloudy"] = float(weather == "Cloudy"); X["day_type_holiday"] = float(holiday)
X["class_changeover"] = ((X.class_prev != X.is_class_in_session) | (X.class_next != X.is_class_in_session)).astype(float)
proba = model.predict_proba(X[feats])
pred_level = {z: int(proba[i].argmax()) for i, z in enumerate(X.index)}
cong = {z: float(proba[i] @ M["level_mid"]) for i, z in enumerate(X.index)}
cap = base.zone_capacity.to_dict()
actual_level = {z: int(base.loc[z, "level"]) for z in X.index}
actual_occ = {z: float(base.loc[z, "ratio"]) for z in X.index}
G = graph_edges()

tab1, tab2, tab3 = st.tabs(["Crowd forecast", "Route planner", "About"])

with tab1:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.pyplot(draw_map(G, pred_level, cong, title=f"Predicted crowd, {day:%a %d %b}, {slot}"))
    with c2:
        rows = []
        for i, z in enumerate(X.index):
            rows.append({"Zone": pretty(z), "Predicted": LEVELS[pred_level[z]], "P(Low)": proba[i][0], "P(Med)": proba[i][1], "P(High)": proba[i][2],
                         "Expected people": round(cong[z] * cap[z]), "Capacity": cap[z], "Actual": LEVELS[actual_level[z]] if day in M["dates"] else "-"})
        st.dataframe(pd.DataFrame(rows).set_index("Zone"),
                     column_config={k: st.column_config.ProgressColumn(k, min_value=0, max_value=1, format="%.0f%%") for k in ["P(Low)", "P(Med)", "P(High)"]})
        changed = (over_exam != bool(base.is_exam_period.iloc[0]) or over_event != bool(base.is_event_day.iloc[0]) or over_class != bool(base.is_class_in_session.iloc[0])
                   or weather != base.weather_condition.iloc[0] or holiday != bool(base.day_type_holiday.iloc[0]))
        st.caption("What-if switches changed: the 'Actual' column is the original record and may no longer apply." if changed else "Switches match the original record, so Actual can be compared directly.")
    st.subheader("The whole day")
    day_df = d[d.date == day]
    Xd = day_df[feats].astype(float); pdl = model.predict(Xd)
    grid = pd.DataFrame({"zone": day_df.zone_name.values, "slot": day_df.time_slot.values, "pred": pdl, "act": day_df.level.values})
    hm = grid.pivot(index="zone", columns="slot", values="pred").reindex(ZONES)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    from matplotlib.colors import ListedColormap
    ax.imshow(hm.values.astype(int), cmap=ListedColormap(COLORS), vmin=0, vmax=2, aspect="auto")
    ax.set_yticks(range(5)); ax.set_yticklabels([pretty(z) for z in ZONES], fontsize=8)
    ax.set_xticks(range(12)); ax.set_xticklabels([s[:2] for s in hm.columns], fontsize=8)
    ax.set_title(f"Predicted level per hour (green low, amber medium, red high). Correct in {(grid.pred == grid.act).mean()*100:.0f}% of hours", fontsize=9)
    fig.tight_layout(); st.pyplot(fig)

with tab2:
    a, b, c = st.columns([1, 1, 1.4])
    src = a.selectbox("From", PLACES, index=PLACES.index("Canteen"), format_func=pretty)
    dst = b.selectbox("To", PLACES, index=PLACES.index("Hostel"), format_func=pretty)
    lam = c.slider("Crowd avoidance", 0.0, 12.0, 3.0, 0.5, help="0 = shortest path only. Higher values accept longer walks to avoid crowds.")
    if src == dst:
        st.warning("Pick two different places.")
    else:
        set_weights(G, {}, 0); p_short = nx.shortest_path(G, src, dst, weight="w")
        set_weights(G, cong, lam); p_aware = nx.shortest_path(G, src, dst, weight="w")
        d1, e1 = path_stats(G, p_short, cong); d2, e2 = path_stats(G, p_aware, cong)
        red = (1 - e2 / e1) * 100 if e1 > 0 else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Crowd exposure vs shortest path", f"{-red:.0f}%", delta=None)
        m2.metric("Extra walking", f"{(d2 - d1) / WALK_M_PER_MIN:+.1f} min")
        m3.metric("Shortest / crowd-aware", f"{d1:.0f} m / {d2:.0f} m")
        l, r = st.columns([1.5, 1])
        l.pyplot(draw_map(G, pred_level, cong, paths=[(p_aware, "#1155D6", "solid", 5), (p_short, "#222222", "dashed", 2.5)],
                          title="Blue = crowd-aware route, black dashed = shortest route"))
        r.markdown("**Shortest route**  \n" + " → ".join(map(pretty, p_short)) + f"  \n{d1:.0f} m, {d1 / WALK_M_PER_MIN:.1f} min")
        r.markdown("**Crowd-aware route**  \n" + " → ".join(map(pretty, p_aware)) + f"  \n{d2:.0f} m, {d2 / WALK_M_PER_MIN:.1f} min")
        r.caption("Exposure uses the model's predicted crowding for this hour and your what-if settings.")
        # best departure hour for this walk (uses the model's predictions for every hour of the day)
        rows = []
        for sl in slots:
            bs = d[(d.date == day) & (d.time_slot == sl)].set_index("zone_name")
            Xs = bs[feats].astype(float)
            ps = model.predict_proba(Xs[feats]); cg = {z: float(ps[i] @ M["level_mid"]) for i, z in enumerate(Xs.index)}
            set_weights(G, cg, lam); pa = nx.shortest_path(G, src, dst, weight="w"); dd, ee = path_stats(G, pa, cg)
            rows.append((sl[:2], ee / dd if dd else 0))
        st.subheader("Best time to leave")
        st.bar_chart(pd.DataFrame(rows, columns=["hour", "average crowding on the route"]).set_index("hour"))

with tab3:
    f1, acc = M["scores"]
    st.markdown(f"""
**Model:** XGBoost classifier, tuned with forward-chaining cross-validation. On the 14 unseen test days it reaches macro-F1 **{f1:.3f}** and accuracy **{acc:.3f}**.

**Inputs the model sees:** the last three hours of occupancy and yesterday's level for the same hour (past data only), the timetable, exam and event flags, holiday and weekend, hour of day and day of week, weather, and the zone.

**Routing:** each zone's class probabilities become an expected occupancy. Edge cost is `distance x (1 + lambda x average congestion of its two ends)`, and Dijkstra's algorithm finds the cheapest path.

**Limits:** the data is simulated, the walking distances are illustrative, and the shortest forecast is one hour because the data is hourly. The history features come from the dataset, so the what-if switches change calendar and weather inputs only.
""")
