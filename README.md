# Monte Carlo CAT Risk Bot

This Streamlit app runs a Monte Carlo CAT loss simulation applying attachment points and limits.

## Features
- Simulate net losses using mock gross loss distribution
- Apply attachment and limit
- Calculate AAL, PML (99%)
- Plot EP curve + loss histogram

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push to GitHub → Deploy on [Streamlit Cloud](https://streamlit.io/cloud)