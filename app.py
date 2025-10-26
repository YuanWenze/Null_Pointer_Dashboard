import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from pymongo import MongoClient

load_dotenv()

# Postgres schema helper
PG_SCHEMA = os.getenv("PG_SCHEMA", "public")   # CHANGE: to your car schema name
def qualify(sql: str) -> str:
    # Replace occurrences of {S}.<table> with <schema>.<table>
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# CONFIG: Postgres and Mongo Queries
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:password@localhost:5432/Connected_Car_Platform"),  # Will read from your .env file
        "queries": {
            # User 1: FLEET MANAGER
            "Fleet Manager: Vehicle Overview": {
                "sql": """
                    SELECT vehicle_id, manufacturer, vehicle_model, engine_type, last_service_date
                    FROM {S}.vehicle
                    ORDER BY last_service_date ASC;
                """,
                "chart": {"type": "table"},
                "tags": ["fleet_manager"],
                "params": []
            },
            "Fleet Manager: Driver Performance": {
                "sql": """
                    SELECT d.driver_id, d.name, COUNT(t.trip_id) as trip_count, COUNT(a.alert_id) as alert_count
                    FROM {S}.driver d
                    LEFT JOIN {S}.trip t ON d.driver_id = t.driver_id
                    LEFT JOIN {S}.alert a ON d.driver_id = a.driver_id
                    GROUP BY d.driver_id, d.name
                    ORDER BY alert_count DESC;
                """,
                "chart": {"type": "bar", "x": "name", "y": "alert_count"},
                "tags": ["fleet_manager", "safety"],
                "params": []
            },
            "Fleet Manager: Recent Trips": {
                "sql": """
                    SELECT trip_id, vehicle_id, start_time, total_distance
                    FROM {S}.trip
                    ORDER BY start_time DESC
                    LIMIT 10;
                """,
                "chart": {"type": "table"},
                "tags": ["fleet_manager"],
                "params": []
            },
            "Fleet Manager: Vehicle Utilization": {
                "sql": """
                    SELECT v.vehicle_id, v.manufacturer, v.vehicle_model, 
                           COUNT(t.trip_id) as trip_count, 
                           COALESCE(SUM(t.total_distance), 0) as total_km
                    FROM {S}.vehicle v
                    LEFT JOIN {S}.trip t ON v.vehicle_id = t.vehicle_id
                    GROUP BY v.vehicle_id, v.manufacturer, v.vehicle_model
                    ORDER BY trip_count DESC;
                """,
                "chart": {"type": "bar", "x": "vehicle_id", "y": "total_km"},
                "tags": ["fleet_manager"],
                "params": []
            },

            # User 2: MAINTENANCE TECHNICIAN
            "Maintenance: Vehicles Due for Service": {
                "sql": """
                    SELECT vehicle_id, manufacturer, vehicle_model, last_service_date
                    FROM {S}.vehicle
                    WHERE last_service_date < CURRENT_DATE - INTERVAL '2 months'
                    ORDER BY last_service_date ASC;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance"],
                "params": []
            },
            "Maintenance: Recent Maintenance Records": {
                "sql": """
                    SELECT m.maintenance_id, v.manufacturer, v.vehicle_model, 
                           m.maintenance_type, m.maintenance_date, m.maintenance_cost
                    FROM {S}.maintenance m
                    JOIN {S}.vehicle v ON m.vehicle_id = v.vehicle_id
                    ORDER BY m.maintenance_date DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance"],
                "params": []
            },
            "Maintenance: Active Alerts": {
                "sql": """
                    SELECT a.alert_id, v.vehicle_id, v.manufacturer, v.vehicle_model,
                           a.alert_type, a.severity_level, a.alert_timestamp
                    FROM {S}.alert a
                    JOIN {S}.vehicle v ON a.vehicle_id = v.vehicle_id
                    WHERE a.alert_timestamp >= NOW() - INTERVAL '48 hours'
                    ORDER BY a.alert_timestamp DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance"],
                "params": []
            },
            "Maintenance: Sensor Overview": {
                "sql": """
                    SELECT v.vehicle_id, v.manufacturer, v.vehicle_model, 
                           s.sensor_type, COUNT(s.sensor_id) as sensor_count
                    FROM {S}.vehicle v
                    JOIN {S}.sensor s ON v.vehicle_id = s.vehicle_id
                    GROUP BY v.vehicle_id, v.manufacturer, v.vehicle_model, s.sensor_type
                    ORDER BY v.vehicle_id;
                """,
                "chart": {"type": "bar", "x": "vehicle_id", "y": "sensor_count"},
                "tags": ["maintenance"],
                "params": []
            },

            # User 3: SAFETY ANALYST
            "Safety: Alert Statistics": {
                "sql": """
                    SELECT alert_type, severity_level, COUNT(*) as count,
                           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {S}.alert), 2) as percentage
                    FROM {S}.alert
                    GROUP BY alert_type, severity_level
                    ORDER BY count DESC;
                """,
                "chart": {"type": "pie", "names": "alert_type", "values": "count"},
                "tags": ["safety"],
                "params": []
            },
            "Safety: High-Risk Drivers": {
                "sql": """
                    SELECT d.driver_id, d.name,
                           COUNT(CASE WHEN a.severity_level = 'High' THEN 1 END) as high_alert_count,
                           COUNT(a.alert_id) as total_alerts
                    FROM {S}.driver d
                    LEFT JOIN {S}.alert a ON d.driver_id = a.driver_id
                    GROUP BY d.driver_id, d.name
                    HAVING COUNT(CASE WHEN a.severity_level = 'High' THEN 1 END) > :risk_threshold
                    OR COUNT(a.alert_id) > :risk_threshold * 2
                    ORDER BY high_alert_count DESC;
                """,
                "chart": {"type": "bar", "x": "name", "y": "high_alert_count"},
                "tags": ["safety"],
                "params": ["risk_threshold"]
            },
            "Safety: Trip Distance Analysis": {
                "sql": """
                    SELECT COUNT(*) as total_trips, 
                           SUM(total_distance) as total_distance, 
                           ROUND(AVG(total_distance), 2) as avg_distance, 
                           MAX(total_distance) as longest_trip
                    FROM {S}.trip;
                """,
                "chart": {"type": "table"},
                "tags": ["safety"],
                "params": []
            },
            "Safety: Speed Analysis": {
                "sql": """
                    SELECT s.vehicle_id, v.manufacturer, v.vehicle_model,
                           ROUND(AVG(s.sensor_value), 2) as avg_speed,
                           MAX(s.sensor_value) as max_speed
                    FROM {S}.sensor s
                    JOIN {S}.vehicle v ON s.vehicle_id = v.vehicle_id
                    WHERE s.sensor_type = 'speed transmitter'
                    GROUP BY s.vehicle_id, v.manufacturer, v.vehicle_model
                    ORDER BY avg_speed DESC;
                """,
                "chart": {"type": "bar", "x": "vehicle_id", "y": "avg_speed"},
                "tags": ["safety"],
                "params": []
            }
        }
    },

    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),  # Will read from the .env file
        "db_name": os.getenv("MONGO_DB", "connected-car-platform"),  # Will read from the .env file
        
        "queries": {
            "Mongo: Hourly Average Speed (Vehicle, Last 24h)": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$match": {
                        "meta.sensor_type": "gps",
                        "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}
                    }},
                    {"$project": {
                        "hour": {"$dateTrunc": {"date": "$ts", "unit": "hour"}},
                        "vehicle_id": "$meta.vehicle_id",
                        "speed": "$speed"
                    }},
                    {"$group": {
                        "_id": {"vehicle": "$vehicle_id", "hour": "$hour"}, 
                        "avg_speed": {"$avg": "$speed"}
                    }},
                    {"$sort": {"_id.hour": 1}}
                ],
                "chart": {"type": "line", "x": "_id.hour", "y": "avg_speed"}
            },

            "Mongo: Harsh Braking Events by Driver (Last 30 days)": {
                "collection": "behavior",
                "aggregate": [
                    {"$match": {
                        "event_type": "harsh_braking",
                        "ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=30)}
                    }},
                    {"$group": {
                        "_id": "$meta.driver_id", 
                        "total_events": {"$count": {}},
                        "latest_event": {"$max": "$ts"}
                    }},
                    {"$sort": {"total_events": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "total_events"}
            },

            "Mongo: Latest Sensor Reading per Vehicle": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$sort": {"ts": -1}},
                    {"$group": {"_id": "$meta.vehicle_id", "doc": {"$first": "$$ROOT"}}},
                    {"$replaceRoot": {"newRoot": "$doc"}},
                    {"$project": {
                        "_id": 0, 
                        "vehicle_id": "$meta.vehicle_id", 
                        "ts": 1,
                        "speed": 1,
                        "fuel_level": 1,
                        "engine_temp": 1
                    }}
                ],
                "chart": {"type": "table"}
            },

            "Mongo: Battery Status Distribution": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$match": {"meta.sensor_type": "battery"}},
                    {"$project": {
                        "battery": "$readings.voltage",
                        "bucket": {
                            "$switch": {
                                "branches": [
                                    {"case": {"$gte": ["$readings.voltage", 13.0]}, "then": "Excellent (13.0+)"},
                                    {"case": {"$gte": ["$readings.voltage", 12.5]}, "then": "Good (12.5-12.9)"},
                                    {"case": {"$gte": ["$readings.voltage", 12.0]}, "then": "Fair (12.0-12.4)"},
                                    {"case": {"$gte": ["$readings.voltage", 11.5]}, "then": "Low (11.5-11.9)"},
                                ],
                                "default": "Critical (<11.5)"
                            }
                        }
                    }},
                    {"$group": {"_id": "$bucket", "cnt": {"$count": {}}}},
                    {"$sort": {"cnt": -1}}
                ],
                "chart": {"type": "pie", "names": "_id", "values": "cnt"}
            },

            "Mongo: Event Count by Type and Driver": {
                "collection": "behavior",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=7)}}},
                    {"$group": {
                        "_id": {"driver": "$meta.driver_id", "event_type": "$event_type"}, 
                        "cnt": {"$count": {}}
                    }},
                    {"$project": {
                        "driver": "$_id.driver", 
                        "event_type": "$_id.event_type", 
                        "count": "$cnt", 
                        "_id": 0
                    }}
                ],
                "chart": {"type": "treemap", "path": ["event_type", "driver"], "values": "count"}
            },

            "Mongo: Environmental Conditions Analysis": {
                "collection": "env_readings",
                "aggregate": [
                    {"$match": {"ts": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=1)}}},
                    {"$group": {
                        "_id": "$readings.road_condition", 
                        "avg_temperature": {"$avg": "$readings.temperature"},
                        "avg_humidity": {"$avg": "$readings.humidity"},
                        "reading_count": {"$count": {}}
                    }},
                    {"$sort": {"reading_count": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "reading_count"}
            }
        }
    }
}

# The following block of code will create a simple Streamlit dashboard page
st.set_page_config(page_title="Connected Car Platform Dashboard", layout="wide")
st.title("Connected Car Platform | Real-time Analytics Dashboard")

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri)

def mongo_overview(client: MongoClient, db_name: str):
    info = client.server_info()
    db = client[db_name]
    colls = db.list_collection_names()
    stats = db.command("dbstats")
    total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
    return {
        "DB": db_name,
        "Collections": f"{len(colls):,}",
        "Total docs (est.)": f"{total_docs:,}",
        "Storage": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
        "Version": info.get("version", "unknown")
    }

@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list):
    db = _client[db_name]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No data available.")
        return
    ctype = spec.get("type", "table")
    # light datetime parsing for x axes
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass

    if ctype == "table":
        st.dataframe(df, use_container_width=True)
    elif ctype == "line":
        st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "bar":
        st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "pie":
        st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
    elif ctype == "heatmap":
        pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                  labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                        use_container_width=True)
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# The following block of code is for the dashboard sidebar
with st.sidebar:
    st.header("Connections")
    # These fields are pre-filled from .env file
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])     
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])        
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"]) 
    st.divider()
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")

    st.header("Role & Parameters")
    # Updated roles for connected car platform
    role = st.selectbox("User role", ["fleet_manager", "maintenance", "safety", "all"], index=3)
    
    # Parameters for different users
    vehicle_id = st.text_input("vehicle_id", value="VH001")
    driver_id = st.number_input("driver_id", min_value=1, value=1, step=1)
    risk_threshold = st.number_input("risk_threshold", min_value=0, value=5, step=1)
    days = st.slider("last N days", 1, 90, 7)

    PARAMS_CTX = {
        "vehicle_id": vehicle_id,
        "driver_id": int(driver_id),
        "risk_threshold": int(risk_threshold),
        "days": int(days),
    }

# Postgres part of the dashboard
st.subheader("PostgreSQL Analytics")
try:
    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres Query", expanded=True):
        # Filter queries by role
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}

        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(no queries for this role)"]
        sel = st.selectbox("Choose a saved query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])   
            st.code(sql, language="sql")

            run  = auto_run or st.button("▶ Run Postgres", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                df = run_pg_query(eng, sql, params=params)
                render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# Mongo panel
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB Analytics")
    try:
        mongo_client = get_mongo_client(mongo_uri)   
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("Run Mongo Aggregation", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                dfm = run_mongo_aggregate(mongo_client, mongo_db, q["collection"], q["aggregate"])
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")
