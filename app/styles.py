import streamlit as st


def load_css():
    st.markdown(
        """
<style>

/* -----------------------------
Main App
----------------------------- */

.main {
    background-color: #F7F9FC;
}

/* -----------------------------
Hide Streamlit Branding
----------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* -----------------------------
Metric Cards
----------------------------- */

div[data-testid="metric-container"] {

    background: white;

    border-radius: 16px;

    padding: 18px;

    border: 1px solid #E5E7EB;

    box-shadow: 0px 8px 24px rgba(0,0,0,0.05);

    transition: all .25s ease;
}

div[data-testid="metric-container"]:hover {

    transform: translateY(-3px);

    box-shadow: 0px 15px 30px rgba(0,0,0,.12);
}

/* -----------------------------
Buttons
----------------------------- */

.stButton>button {

    width:100%;

    border-radius:12px;

    border:none;

    padding:12px;

    font-size:16px;

    font-weight:600;

    background:#534AB7;

    color:white;
}

.stButton>button:hover{

    background:#4338CA;
}

/* -----------------------------
Sidebar
----------------------------- */

section[data-testid="stSidebar"]{

    background:#FFFFFF;

    border-right:1px solid #E5E7EB;
}

/* -----------------------------
Tabs
----------------------------- */

button[data-baseweb="tab"]{

    font-size:16px;

    font-weight:600;

    border-radius:12px;

    padding:10px 18px;
}

/* -----------------------------
Dataframe
----------------------------- */

div[data-testid="stDataFrame"]{

    border-radius:16px;

    overflow:hidden;

    border:1px solid #E5E7EB;
}

/* -----------------------------
Plotly Charts
----------------------------- */

.js-plotly-plot{

    border-radius:16px;

    overflow:hidden;

    background:white;

    padding:10px;

    box-shadow:0 8px 20px rgba(0,0,0,.05);
}

/* -----------------------------
Info Boxes
----------------------------- */

.stAlert{

    border-radius:16px;
}

/* -----------------------------
Headers
----------------------------- */

h1{

    font-weight:800;

    color:#1F2937;
}

h2{

    font-weight:700;
}

h3{

    font-weight:700;
}

/* -----------------------------
Horizontal Line
----------------------------- */

hr{

    margin-top:30px;

    margin-bottom:30px;
}

</style>
        """,
        unsafe_allow_html=True,
    )