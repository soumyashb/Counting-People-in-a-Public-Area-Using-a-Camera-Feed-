import sqlite3


conn = sqlite3.connect("crowd_counts.db")
cursor = conn.cursor()

# Crowd count table
cursor.execute("""
CREATE TABLE IF NOT EXISTS crowd_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    count INTEGER
)
""")

# Entry/Exit table
cursor.execute("""
CREATE TABLE IF NOT EXISTS entry_exit_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    entry_count INTEGER,
    exit_count INTEGER,
    total_count INTEGER
)
""")

# Users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

# Alerts table
cursor.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    threshold INTEGER
)
""")

# Default admin
cursor.execute(
    "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
    ("admin", "admin123", "admin")
)

# Default alert threshold
cursor.execute(
    "INSERT OR IGNORE INTO alerts (id, threshold) VALUES (1, 10)"
)

conn.commit()
conn.close()
print("✅ Database setup complete")