import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det.pkl")
news_model=joblib.load("news_cat.pkl")
review_model=joblib.load("review.pkl")


st.set_page_config(layout="wide")

st.markdown("""
    <div style='background-color: white; color: black; padding: 10px 15px; border-radius: 10px; text-align: center;'>
        <h1 style='margin: 0;'>
            🎯 LENS-X (Lens Expert - ML & NLP Suite)
        </h1>
        <p style='margin-top: 8px; font-size: 18px; font-weight: bold;'>
            (The Ultimate NLP Solution 🚀)
        </p>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    div[data-testid="stTabs"] label {
        font-size: 30px !important;
    }
    </style>
""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    "📩 Spam Classifier",
    "🔤 Language Detection",
    "🍽️ Food Review Sentiment",
    "📰 News Classification",
    "📊 Data Analyst Portfolio",
    "🧑‍💻 ML Playground"
])

with tab1:
    st.header("📩 Spam Classifier")

    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 10px; border-radius: 8px;'>
        <p style='font-size:16px; color:#333;'>
            📄 <b>Description:</b> This Spam Classifier detects whether a given message is <i>Spam</i> or <i>Not Spam</i> using a trained NLP model. Enter your message below or upload a file with multiple messages to check them in bulk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    msg=st.text_input("Enter Msg", key="spam_input")
    if st.button("Check for Spam"):
        pred=spam_model.predict([msg])
        if pred[0]==0:
            st.image("spam.jpg",width=300)
        else:
            st.image("not_spam.jpg",width=300)

    uploaded_file=st.file_uploader("upload file containing bulk msg",type=["csv","txt"])
    
    if uploaded_file:
        df_spam=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
        pred=spam_model.predict(df_spam.Msg)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam["Check for Spam"]=pred
        df_spam["Check for Spam"]=df_spam["Check for Spam"].map({0:'Spam',1:'Not Spam'})
        st.dataframe(df_spam)

with tab2:
    st.header("🔤 Language Detection")

    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 10px; border-radius: 8px;'>
        <p style='font-size:16px; color:#333;'>
            🌐 <b>Description:</b> This Language Detection tool identifies the language of your input text using an advanced NLP model. Enter a sentence below or upload a file with multiple lines of text to detect languages in bulk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    msg = st.text_input("Enter Msg", key="lang_input")
    if st.button("Detect Language"):
        pred = language_model.predict([msg])
        st.success(f"🔤 Detected Language: {pred[0]}")

    uploaded_file = st.file_uploader("Upload file containing multiple texts", type=["csv", "txt"], key="lang_file")

    if uploaded_file:
        df_lang = pd.read_csv(uploaded_file, header=None, names=['Text'])
        pred = language_model.predict(df_lang.Text)
        df_lang.index = range(1, df_lang.shape[0] + 1)
        df_lang["Detected Language"] = pred
        st.dataframe(df_lang)

with tab3:
    st.header("🍽️ Food Review Sentiment")

    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 10px; border-radius: 8px;'>
        <p style='font-size:16px; color:#333;'>
            ⭐ <b>Description:</b> Analyze customer sentiments from food reviews! This tool classifies reviews as Positive or Negative using a trained NLP sentiment model. Enter a single review or upload a file with multiple reviews to get instant sentiment analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    review_msg = st.text_input("Enter your food review", key="review_input")
    if st.button("Analyze Sentiment"):
        pred = review_model.predict([review_msg])
        if pred[0] == 1:
            st.image("like.png",width=300)
        else:
            st.image("dislike.png",width=300)

    # Bulk reviews prediction
    uploaded_review_file = st.file_uploader("Upload CSV file containing bulk reviews", type=["csv", "txt"], key="review_upload")
    if uploaded_review_file:
        df_reviews = pd.read_csv(uploaded_review_file, header=None, names=['Review'])
        pred_reviews = review_model.predict(df_reviews.Review)
        df_reviews.index = range(1, df_reviews.shape[0]+1)
        df_reviews["Prediction"] = pred_reviews
        df_reviews["Prediction"] = df_reviews["Prediction"].map({1: 'Positive 😊', 0: 'Negative 😞'})
        st.dataframe(df_reviews)
    
with tab4:
    st.header("📰 News Classification")

    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 10px; border-radius: 8px;'>
        <p style='font-size:16px; color:#333;'>
            🗞️ <b>Description:</b> Instantly categorize news headlines or articles into relevant topics like sports, politics, business, and more! Our NLP-based classifier helps you understand the theme of your news content. Enter a headline or upload a file with multiple news lines for bulk classification.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.image("under_construction.png", width=300)

with tab5:
    st.header("🏦 Canara Bank Automation Portfolio")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("canara_bank_logo_2.png", width=250) 

    with col2:
        st.markdown("""
        ## Project Overview

        This project automates key processes in Canara Bank, including:
        
        - 🏦 Open Account
        - ❌ Delete Account
        - 🔍 Check Details
        - 💰 Deposit
        - 💸 Withdraw
        - 🔄 Transfer
        
        **Technologies Used:** Python, Tkinter, Sqlite3
        
        ### 🔗 Links:
        <a href='https://github.com/raunakbharti/banking_automation' target='_blank'>GitHub</a> |
        <a href='https://linkedin.com/in/www.linkedin.com/in/raunak-bharti-636470276' target='_blank'>LinkedIn</a>
    """, unsafe_allow_html=True)
        
with tab6:
    st.header("🚀 Build Your Own ML Model Without Coding")
    st.write("Upload your dataset, select target, and train ML models interactively!")

    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 Preview of Data:", df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    # Encode categorical
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Decide problem type
    problem_type = "Regression" if y.dtype in ['int64', 'float64'] and len(set(y)) > 10 else "Classification"
    st.info(f"Detected Problem Type: **{problem_type}**")

    # Show relevant algorithms
    if problem_type == "Classification":
        algo_options = ["Logistic Regression", "KNN", "Random Forest"]
    else:
        algo_options = ["Linear Regression"]

    algo = st.selectbox("🤖 Choose Algorithm", algo_options)

    scaler_needed = algo in ["Logistic Regression", "KNN"]
    if scaler_needed:
        st.caption("⚠️ Scaling will be applied for this algorithm.")

    test_size = st.slider("🧪 Select test size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if scaler_needed:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if st.button("🏋️ Train Model"):
        if algo == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif algo == "KNN":
            model = KNeighborsClassifier()
        elif algo == "Random Forest":
            model = RandomForestClassifier()
        elif algo == "Linear Regression":
            model = LinearRegression()

        model.fit(X_train, y_train)
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.success("✅ Model trained successfully!")

    if st.button("📊 Test Model"):
        if "model" in st.session_state:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            y_pred = model.predict(X_test)

            if problem_type == "Regression":
                score = r2_score(y_test, y_pred)
                st.metric("📈 R² Score", f"{score:.4f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                st.metric("🎯 Accuracy", f"{acc:.4f}")

            if algo == "Random Forest":
                st.subheader("📊 Feature Importances")
                st.bar_chart(pd.Series(model.feature_importances_, index=features))
        else:
            st.warning("⚠️ Please train the model first.")


st.sidebar.image("image.png")
with st.sidebar.expander("🧑‍💼👩‍💼About us"):
    st.write("We are a group of students trying to understand the concept of NLP")
with st.sidebar.expander("📞Contact us"):
    st.write("999999999")
    st.write("aaaa@gmail.com")
with st.sidebar.expander("🆘 Help"):
    st.write("🔹 Use the tabs above to access different NLP tools.")
    st.write("🔹 Upload a .csv or .txt file for bulk predictions.")
    st.write("🔹 For any issues, please contact support@example.com.")


