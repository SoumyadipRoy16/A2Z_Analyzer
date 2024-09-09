import streamlit as st
import sqlite3
from datetime import datetime
from uuid import uuid4
import hashlib
import pandas as pd
import io
from ExtraML import ExtraML  

def init_db():
    conn = sqlite3.connect('conversations.db', check_same_thread=False)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT, 
                  timestamp DATETIME, 
                  role TEXT, 
                  content TEXT, 
                  title TEXT, 
                  user_id INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')

    conn.commit()
    return conn, c

conn, c = init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    hashed_password = hash_password(password)
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    result = c.fetchone()
    return result[0] if result else None

def save_message(session_id, role, content, user_id, title=None):
    if title:
        c.execute("UPDATE conversations SET title = ? WHERE session_id = ? AND user_id = ?", (title, session_id, user_id))
    c.execute("INSERT INTO conversations (session_id, timestamp, role, content, user_id) VALUES (?, ?, ?, ?, ?)",
              (session_id, datetime.now(), role, content, user_id))
    conn.commit()

def get_session_list(user_id):
    c.execute("""
        SELECT session_id, 
               COALESCE(MIN(CASE WHEN role = 'system' THEN title END), MIN(title)) as title,
               MAX(timestamp) as last_update
        FROM conversations 
        WHERE user_id = ?
        GROUP BY session_id
        HAVING COUNT(*) > 0
        ORDER BY last_update DESC
    """, (user_id,))
    return [(row[0], row[1] or f"Conversation {row[0][:8]}") for row in c.fetchall()]

def load_conversation(session_id, user_id):
    c.execute("SELECT role, content FROM conversations WHERE session_id = ? AND user_id = ? ORDER BY timestamp",
              (session_id, user_id))
    return c.fetchall()

def process_csv(uploaded_file):
    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
    return df

st.set_page_config(page_title="Data Science Assistant", page_icon="📊", layout="centered")

if 'user_id' not in st.session_state:
    st.title("Data Science Assistant - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    with col2:
        if st.button("Create Account"):
            if create_user(username, password):
                st.success("Account created successfully. Please log in.")
            else:
                st.error("Username already exists")

if 'user_id' in st.session_state:
    st.title(f"Data Science Assistant - Welcome, {st.session_state.username}")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'extra_ml' not in st.session_state:
        st.session_state.extra_ml = ExtraML()

    with st.sidebar:
        st.header("About")
        st.write("Data Science Assistant is an AI-powered tool for data analysis tasks.")
        st.write("Version 1.0")

        session_list = get_session_list(st.session_state.user_id)

        if st.sidebar.button("Start New Conversation"):
            st.session_state.session_id = str(uuid4())
            st.session_state.messages = []
            st.session_state.uploaded_files = {}
            st.session_state.extra_ml = ExtraML()
            st.rerun()

        st.subheader("Past Conversations")
        for session_id, title in session_list:
            if st.button(title[:20] + '...' if len(title) > 20 else title):
                st.session_state.session_id = session_id
                st.session_state.messages = load_conversation(session_id, st.session_state.user_id)
                st.rerun()

        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = process_csv(uploaded_file)
        st.session_state.uploaded_files[uploaded_file.name] = df
        st.success(f"CSV '{uploaded_file.name}' uploaded and processed successfully!")

    if st.session_state.uploaded_files:
        st.write("Uploaded files:")
        for file_name, df in st.session_state.uploaded_files.items():
            st.write(f"- {file_name}")
            st.dataframe(df.head())

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)


if prompt := st.chat_input("What data science task can I help you with?"):
    st.session_state.messages.append(("user", prompt))
    save_message(st.session_state.session_id, "user", prompt, st.session_state.user_id)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        with st.spinner("Analyzing..."):
            if "correlation" in prompt.lower():
                for file_name, df in st.session_state.uploaded_files.items():
                    plots, html_content = st.session_state.extra_ml.correlation_analysis(df)
                    st.subheader(f"Correlation Analysis for {file_name}")
                    for i, plot in enumerate(plots):
                        st.plotly_chart(plot, use_container_width=True)
                        st.write(f"Plot {i+1} description: [Add a brief explanation here]")
                        st.markdown("---")  
                    for html in html_content:
                        st.components.v1.html(html, height=600)
                response = "Correlation analysis completed. The plots are displayed above."
            elif "missing values" in prompt.lower():
                for file_name, df in st.session_state.uploaded_files.items():
                    missing_info = st.session_state.extra_ml.check_missing_values(df)
                    st.write(missing_info)
                response = "Missing values analysis completed. The results are displayed above."
            elif "train model" in prompt.lower():
                if len(st.session_state.uploaded_files) >= 2:
                    train_file, test_file = list(st.session_state.uploaded_files.values())[:2]
                    predictions, plots, html_content = st.session_state.extra_ml.fit(train_file, test_file)
                    for plot in plots:
                        st.pyplot(plot)
                    for html in html_content:
                        st.components.v1.html(html, height=600)
                    response = "Model training completed. The results and plots are displayed above."
                else:
                    response = "Please upload both training and test datasets to train the model."
            else:
                response = "I'm sorry, I couldn't understand your request. Could you please rephrase or specify a data science task?"

        st.write(response)
    st.session_state.messages.append(("assistant", response))
    save_message(st.session_state.session_id, "assistant", response, st.session_state.user_id)

    if st.button("Generate Report"):
        if len(st.session_state.uploaded_files) >= 2:
            train_df, test_df = list(st.session_state.uploaded_files.values())[:2]
            report = st.session_state.extra_ml.run_analysis(train_df, test_df)
            st.text_area("Analysis Report", report, height=300)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="extraml_report.txt",
                mime="text/plain"
            )
        else:
            st.error("Please upload both training and test datasets to generate the report.")

    if st.button("Export Conversation"):
        conversation = "\n".join([f"{m[0]}: {m[1]}" for m in st.session_state.messages])
        st.download_button(
            label="Download Conversation",
            data=conversation,
            file_name=f"data_science_conversation_{st.session_state.session_id}.txt",
            mime="text/plain"
        )