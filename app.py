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
PG_SCHEMA = os.getenv("PG_SCHEMA", "car_schema")   # CHANGE: to your car schema name
def qualify(sql: str) -> str:
    # Replace occurrences of {S}.<table> with <schema>.<table>
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# CONFIG: Postgres and Mongo Queries
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:password@localhost:5432/connected_car"),  # Will read from your .env file
        "queries": {
            # User 1: FLEET MANAGER
            "Fleet Manager: Average Speed per Vehicle": {
                "sql": """
                    SELECT vehicle_id, AVG(speed) as avg_speed 
                    FROM {S}.sensor_readings 
                    WHERE ts >= NOW() - INTERVAL '7 days'
                    GROUP BY vehicle_id
                    ORDER BY avg_speed DESC;
                """,
                "chart": {"type": "bar", "x": "vehicle_id", "y": "avg_speed"},
                "tags": ["fleet_manager"],
                "params": []
            },
            "Fleet Manager: Harsh Braking Events by Driver": {
                "sql": """
                    SELECT driver_id, COUNT(*) as event_count
                    FROM {S}.behavior_events 
                    WHERE event_type = 'harsh_braking' 
                    AND ts >= NOW() - INTERVAL '30 days'
                    GROUP BY driver_id
                    ORDER BY event_count DESC;
                """,
                "chart": {"type": "bar", "x": "driver_id", "y": "event_count"},
                "tags": ["fleet_manager", "safety"],
                "params": []
            },
            "Fleet Manager: Vehicle Utilization (Hours Driven)": {
                "sql": """
                    SELECT vehicle_id, 
                           SUM(CASE WHEN ignition_status = true THEN 1 ELSE 0 END) as total_hours,
                           COUNT(DISTINCT DATE(ts)) as days_operated
                    FROM {S}.sensor_readings 
                    WHERE ts >= NOW() - INTERVAL '7 days'
                    AND sensor_type = 'ignition'
                    GROUP BY vehicle_id
                    ORDER BY total_hours DESC;
                """,
                "chart": {"type": "bar", "x": "vehicle_id", "y": "total_hours"},
                "tags": ["fleet_manager"],
                "params": []
            },
            "Fleet Manager: Speeding Incidents by Time of Day": {
                "sql": """
                    SELECT EXTRACT(HOUR FROM ts) as hour_of_day, 
                           COUNT(*) as incident_count
                    FROM {S}.behavior_events 
                    WHERE event_type = 'speeding'
                    AND ts >= NOW() - INTERVAL '30 days'
                    GROUP BY hour_of_day
                    ORDER BY hour_of_day;
                """,
                "chart": {"type": "line", "x": "hour_of_day", "y": "incident_count"},
                "tags": ["fleet_manager", "safety"],
                "params": []
            },

            # User 2: MAINTENANCE TECHNICIAN
            "Maintenance: Vehicles with Engine Warnings": {
                "sql": """
                    SELECT vehicle_id, warning_code, severity, timestamp
                    FROM {S}.diagnostic_alerts 
                    WHERE severity >= 2
                    AND timestamp >= NOW() - INTERVAL '48 hours'
                    ORDER BY severity DESC, timestamp DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance"],
                "params": []
            },
            "Maintenance: Battery Voltage Trends": {
                "sql": """
                    SELECT vehicle_id,
                           DATE_TRUNC('hour', ts) as hour,
                           AVG(voltage) as avg_voltage,
                           MIN(voltage) as min_voltage,
                           MAX(voltage) as max_voltage
                    FROM {S}.sensor_readings 
                    WHERE sensor_type = 'battery'
                    AND ts >= NOW() - INTERVAL '24 hours'
                    GROUP BY vehicle_id, hour
                    ORDER BY vehicle_id, hour;
                """,
                "chart": {"type": "line", "x": "hour", "y": "avg_voltage"},
                "tags": ["maintenance"],
                "params": ["vehicle_id"]
            },
            "Maintenance: Tire Pressure Anomalies": {
                "sql": """
                    SELECT vehicle_id, tire_position, pressure,
                           CASE 
                               WHEN pressure < 32 THEN 'Low'
                               WHEN pressure > 38 THEN 'High'
                               ELSE 'Normal'
                           END as status
                    FROM {S}.tire_readings 
                    WHERE (pressure < 32 OR pressure > 38)
                    AND ts >= NOW() - INTERVAL '2 hours'
                    ORDER BY vehicle_id, tire_position;
                """,
                "chart": {"type": "table"},
                "tags": ["maintenance"],
                "params": []
            },

            # User 3: SAFETY ANALYST
            "Safety: Driver Behavior Scores": {
                "sql": """
                    SELECT driver_id,
                           SUM(CASE 
                               WHEN event_type = 'speeding' THEN -2
                               WHEN event_type = 'harsh_braking' THEN -3
                               WHEN event_type = 'rapid_acceleration' THEN -2
                               WHEN event_type = 'sharp_turning' THEN -2
                               WHEN event_type = 'smooth_driving' THEN 1
                               ELSE 0
                           END) as behavior_score,
                           COUNT(*) as total_events
                    FROM {S}.behavior_events 
                    WHERE ts >= NOW() - INTERVAL '7 days'
                    GROUP BY driver_id
                    ORDER BY behavior_score DESC;
                """,
                "chart": {"type": "bar", "x": "driver_id", "y": "behavior_score"},
                "tags": ["safety"],
                "params": []
            },
            "Safety: Event Distribution by Type": {
                "sql": """
                    SELECT event_type, COUNT(*) as event_count
                    FROM {S}.behavior_events 
                    WHERE ts >= NOW() - INTERVAL '30 days'
                    GROUP BY event_type
                    ORDER BY event_count DESC;
                """,
                "chart": {"type": "pie", "names": "event_type", "values": "event_count"},
                "tags": ["safety"],
                "params": []
            },
            "Safety: High-Risk Drivers": {
                "sql": """
                    SELECT driver_id, 
                           COUNT(CASE WHEN event_type = 'harsh_braking' THEN 1 END) as harsh_braking_count,
                           COUNT(CASE WHEN event_type = 'speeding' THEN 1 END) as speeding_count,
                           COUNT(*) as total_events
                    FROM {S}.behavior_events 
                    WHERE ts >= NOW() - INTERVAL '30 days'
                    GROUP BY driver_id
                    HAVING COUNT(CASE WHEN event_type = 'harsh_braking' THEN 1 END) > :risk_threshold
                    OR COUNT(CASE WHEN event_type = 'speeding' THEN 1 END) > :risk_threshold
                    ORDER BY total_events DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["safety"],
                "params": ["risk_threshold"]
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
