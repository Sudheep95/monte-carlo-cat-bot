import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MonteCarloCATBot:
    def __init__(self, attachment=0, limit=1e7, trials=10000):
        self.attachment = attachment
        self.limit = limit
        self.trials = trials
        self.losses = []

    def run_simulation(self):
        self.losses = []
        for _ in range(self.trials):
            gross_loss = np.random.exponential(scale=1e6)
            net_loss = max(0, gross_loss - self.attachment)
            net_loss = min(net_loss, self.limit)
            self.losses.append(net_loss)
        return self.losses

    def get_metrics(self):
        losses = np.array(self.losses)
        aal = np.mean(losses)
        pml_99 = np.percentile(losses, 99)
        return {"AAL": aal, "PML_99": pml_99}

    def get_ep_curve(self):
        df = pd.DataFrame({"Loss": sorted(self.losses, reverse=True)})
        df["ReturnPeriod"] = 1 / (np.arange(1, len(df) + 1) / len(df))
        return df

st.set_page_config(page_title="Monte Carlo CAT Bot", layout="wide")
st.title("🌪 Monte Carlo CAT Risk Bot")

st.sidebar.header("Simulation Parameters")
attachment = st.sidebar.number_input("Attachment Point", min_value=0.0, value=0.0, step=100000.0)
limit = st.sidebar.number_input("Limit", min_value=0.0, value=10000000.0, step=1000000.0)
trials = st.sidebar.number_input("Number of Trials", min_value=100, value=10000, step=1000)

if st.button("Run Simulation"):
    bot = MonteCarloCATBot(attachment=attachment, limit=limit, trials=trials)
    losses = bot.run_simulation()
    metrics = bot.get_metrics()
    ep_df = bot.get_ep_curve()

    st.subheader("Simulation Results")
    st.write(f"Average Annual Loss (AAL): ${metrics['AAL']:,.0f}")
    st.write(f"99% Probable Maximum Loss (PML): ${metrics['PML_99']:,.0f}")

    st.subheader("EP Curve")
    fig, ax = plt.subplots()
    ax.plot(ep_df["ReturnPeriod"], ep_df["Loss"])
    ax.set_xscale("log")
    ax.set_xlabel("Return Period (Years)")
    ax.set_ylabel("Loss")
    ax.set_title("EP Curve")
    st.pyplot(fig)

    st.subheader("Loss Histogram")
    fig2, ax2 = plt.subplots()
    ax2.hist(losses, bins=50)
    ax2.set_xlabel("Net Loss")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    st.subheader("EP Curve Data")
    st.dataframe(ep_df)