import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("todays_bets.csv")
st.table(df)