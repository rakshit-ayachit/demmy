import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from PIL import Image
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import time
import base64
import pandas as pd
import sqlite3
from passlib.hash import bcrypt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

tesseract_dir = "./models/tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_dir

def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, is_admin INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

def update_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in c.fetchall()]
    if 'is_admin' not in columns:
        c.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0')
    conn.commit()
    conn.close()

def create_logs_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 username TEXT, 
                 event_type TEXT, 
                 event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def user_exists(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    result = c.fetchone()
    conn.close()
    return result is not None

def add_user(username, password, is_admin):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = bcrypt.hash(password)
    c.execute('INSERT INTO users VALUES (?, ?, ?)', (username, hashed_password, is_admin))
    conn.commit()
    conn.close()
    add_log(username, 'signup')

def verify_login(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    user = c.fetchone()
    conn.close()
    if user:
        stored_password = user[1]
        is_authenticated = bcrypt.verify(password, stored_password)
        if is_authenticated:
            is_admin = user[2]
            add_log(username, 'login')
            return True, is_admin
    return False, False

def logout(username):
    add_log(username, 'logout')

def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def ocr_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def remove_whitespace(text):
    return " ".join(text.split())

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = " ".join([word for word in word_tokens if word.lower() not in stop_words])
    return filtered_text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = " ".join([lemmatizer.lemmatize(word) for word in word_tokens])
    return lemmatized_text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def predict(model, tokenizer, texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors='tf')
    outputs = model(encodings)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_labels = np.argmax(logits, axis=-1)
    confidence_scores = np.max(probabilities, axis=-1)
    return predicted_labels, confidence_scores

def view_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    conn.close()
    return users

def view_logs():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM user_logs ORDER BY event_time DESC')
    logs = c.fetchall()
    conn.close()
    return logs

def add_log(username, event_type):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO user_logs (username, event_type) VALUES (?, ?)', (username, event_type))
    conn.commit()
    conn.close()

def login_signup():
    st.title('User Authentication')

    create_users_table()
    update_users_table()
    create_logs_table()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'login_mode' not in st.session_state:
        st.session_state.login_mode = 'Login'

    if st.session_state.login_mode == 'Login':
        st.subheader('Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            is_authenticated, is_admin = verify_login(username, password)
            if is_authenticated:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.is_admin = is_admin
                if st.session_state.is_admin:
                    st.session_state.admin_view = True
                st.experimental_rerun()
                st.empty()
            else:
                st.error('Invalid username or password')
        if st.button('Go to Signup'):
            st.session_state.login_mode = 'Signup'
            st.experimental_rerun()
            st.empty()
        st.empty()
    
    elif st.session_state.login_mode == 'Signup':
        st.subheader('Signup')
        new_username = st.text_input('New Username')
        new_password = st.text_input('New Password', type='password')
        admin_key = st.text_input('Admin Key (Optional)', type='password')
        
        if st.button('Signup'):
            if user_exists(new_username):
                st.warning('Username already exists')
            else:
                is_admin = False
                if admin_key == '0000':
                    is_admin = True
                add_user(new_username, new_password, is_admin)
                st.success(f'Registered new user: {new_username}')
        if st.button('Go to Login'):
            st.session_state.login_mode = 'Login'
            st.experimental_rerun()
            st.empty()
        st.empty()

def document_classification():
    st.title('Tally Document Classification App')

    with st.sidebar:
        st.markdown(f'<span style="font-family: Arial, sans-serif; font-size: 18px;">$ Logged in as <span style="font-family: monospace; color: #4CAF50;">{st.session_state.username}</span></span>', unsafe_allow_html=True)
        st.markdown("### Upload or Input Text")
        uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True)
        show_images = st.checkbox("Show Uploaded Images", value=False)
        manual_text = st.text_area("Or enter text manually")

        model_options = ["BERT", "ELECTRA"]
        selected_model = st.selectbox("Select Model", model_options)

        model_paths = {
            "BERT": "./models",
            "ELECTRA": "./models"
            }
        predict_button = st.button('Predict')

    label_dict = {'Budget': 0, 'Email': 1, 'Memo': 2, 'Form': 3, 'Invoice': 4}
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    texts = []
    filenames = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f'Uploading image from {uploaded_file.name}...'):
                time.sleep(1)
                extracted_text = ocr_image(uploaded_file)
                texts.append(extracted_text)
                filenames.append(uploaded_file.name)

            if show_images:
                st.image(uploaded_file, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)

    if manual_text:
        texts.append(manual_text)
        filenames.append("Manual Input")

    if predict_button and texts:
        with st.spinner('Preprocessing text...'):
            preprocessed_texts = [preprocess_text(text) for text in texts]

        model_path = model_paths[selected_model]
        model, tokenizer = load_model_and_tokenizer(model_path)

        with st.spinner('Loading model and making prediction...'):
            predicted_labels, confidence_scores = predict(model, tokenizer, preprocessed_texts)

        decoded_labels = [reverse_label_dict[label] for label in predicted_labels]

        results_df = pd.DataFrame({
            'Filename': filenames,
            'Predicted Label': decoded_labels,
            'Confidence Score (%)': confidence_scores * 100
        })

        st.subheader('Predicted Labels and Confidence Scores:')
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        st.markdown(f'<a href="{href}" download="document_classification_results.csv"><button>Download CSV File</button></a>', unsafe_allow_html=True)

    if st.session_state.is_admin:
        if st.button('Go back to Admin Page'):
            st.session_state.admin_view = True
            st.experimental_rerun()
            st.empty()
        st.empty()
    st.empty()

def view_users_page():

    users = view_users()
    user_data = {'Username': [], 'Password': [], 'Admin': []}
    for user in users:
        user_data['Username'].append(user[0])
        user_data['Password'].append(user[1])
        user_data['Admin'].append('Yes' if user[2] else 'No')

    st.subheader('List of All Users')
    st.dataframe(pd.DataFrame(user_data))

    csv = pd.DataFrame(user_data).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    st.markdown(f'<a href="{href}" download="users.csv"><button>Download Users CSV</button></a>', unsafe_allow_html=True)

def view_logs_page():

    logs = view_logs()
    log_data = {'ID': [], 'Username': [], 'Event Type': [], 'Event Time': []}
    for log in logs:
        log_data['ID'].append(log[0])
        log_data['Username'].append(log[1])
        log_data['Event Type'].append(log[2])
        log_data['Event Time'].append(log[3])

    st.subheader('User Login/Logout Logs:')
    if logs:
        st.dataframe(pd.DataFrame(log_data))
    else:
        st.write('No logs available')

    csv = pd.DataFrame(log_data).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    st.markdown(f'<a href="{href}" download="logs.csv"><button>Download Logs CSV</button></a>', unsafe_allow_html=True)

def main():
    create_users_table()
    update_users_table()
    create_logs_table()

    if st.session_state.get('logged_in', False):
        if st.session_state.is_admin:
            if st.session_state.get('admin_view', True):
                st.title('Admin Dashboard')
                st.write('Select an option from below:')
                if st.button('Go to Document Classifier'):
                    st.session_state.admin_view = False
                    st.experimental_rerun()
                    st.empty()
                st.empty()

                if st.button('View All Users'):
                    view_users_page()
                    st.empty()

                if st.button('View Logs'):
                    view_logs_page()
                    st.empty()
                
                if st.button('View GitHub Repository'):
                    st.markdown('Link to GitHub repo here')
                    st.empty()

                if st.button('Logout'):
                    logout(st.session_state.username)
                    st.session_state.logged_in = False
                    st.session_state.admin_view = True
                    st.experimental_rerun()
                    st.empty()

            else:
                document_classification()
                if st.button('Logout'):
                    logout(st.session_state.username)
                    st.session_state.logged_in = False
                    st.session_state.admin_view = True
                    st.experimental_rerun()
                    st.empty()
                st.empty()

        else:
            document_classification()
            if st.button('Logout'):
                logout(st.session_state.username)
                st.session_state.logged_in = False
                st.experimental_rerun()
                st.empty()
            st.empty()

    else:
        login_signup()
        st.empty()

if __name__ == '__main__':
    main()
