import streamlit as st
import pandas as pd
import numpy as np

st.title('Underdog Player Props Predictions')

st.markdown('Below is a table showcasing current Underdog Fantasy player props in the NBA, along with a prediction generated from a model. Here are the column names and their description:')
st.markdown('* **Name**: Name of the NBA player')
st.markdown('* **Underdog Points**: The predicted amount of points from Underdog Fantasy.')
st.markdown('* **Model Points**: The predicted amount of points from our model.')
st.markdown('* **Differential**: The absolute differential between Underdog Points and Model Points.')
st.markdown('* **Pos/Neg**: Determines the sign of our differential. + meaning our model predicts the player is going to score more than Underdog Fantasy’s prediction. – meaning our model predicts the player is going to score less than Underdog Fantasy’s prediction.')



st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

df = pd.read_csv("todays_bets.csv")
st.dataframe(df)

# :arrow_up_small:
# :arrow_down_small: