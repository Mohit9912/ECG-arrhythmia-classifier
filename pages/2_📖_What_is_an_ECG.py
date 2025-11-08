import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ECG Guide", layout="wide")

st.title("📖 What is an ECG?")
st.markdown("### A Crash Course in Heart Signals for Non-Experts")

st.info(
    "**The Core Concept:** Your heart is an electrically driven pump. "
    "An ECG (Electrocardiogram) is just a graph of that electrical voltage over time."
)

st.markdown("---")

# --- SECTION 1: ANATOMY OF A HEARTBEAT ---
st.header("1. Anatomy of a Healthy Heartbeat")

# A "textbook" normal beat for demonstration
# (This is a smoothed, idealized version of a real beat)
normal_beat = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.1, 0.08, 0.05, 0.02, 0.0, # P-wave
    0.0, 0.0, 0.0, -0.05, -0.1, # Q-wave dip
    0.3, 0.7, 1.0, 0.7, 0.3, # R-peak (big spike)
    -0.2, -0.1, 0.0, # S-wave dip
    0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.22, 0.2, 0.15, 0.1, 0.05, 0.0, # T-wave
    0.0, 0.0, 0.0
]

# Create layout columns
col1, col2 = st.columns([3, 2])

with col1:
    # --- DRAW THE INTERACTIVE PLOT ---
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the beat with a thicker line
    ax.plot(normal_beat, color='#1f77b4', linewidth=3)
    
    # Add colorful shaded regions to highlight waves
    # P-Wave Highlight (Green)
    ax.axvspan(5, 15, color='green', alpha=0.2, label='P-Wave')
    # QRS Highlight (Red)
    ax.axvspan(18, 28, color='red', alpha=0.2, label='QRS Complex')
    # T-Wave Highlight (Blue)
    ax.axvspan(33, 45, color='blue', alpha=0.2, label='T-Wave')

    # Annotate the R-Peak with an arrow
    ax.annotate('R-Peak', xy=(22, 1.0), xytext=(25, 0.99),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                fontsize=12, fontweight='bold')

    # Make the plot look professional
    ax.set_title("A 'Perfect' Textbook Heartbeat", fontsize=16)
    ax.set_xlabel("Time ->")
    ax.set_ylabel("Voltage (Electrical Signal)")
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

with col2:
    # --- USE TABS FOR CLEANER INFO ---
    st.subheader("Understanding the Waves")
    tab1, tab2, tab3 = st.tabs(["🟢 P-Wave", "🔴 QRS Complex", "🔵 T-Wave"])
    
    with tab1:
        st.markdown("### The 'Prep' Phase")
        st.write("The small bump at the start.")
        st.success("**What's happening?** The top chambers of the heart (atria) are squeezing to push blood into the main pumps.")
    
    with tab2:
        st.markdown("### The 'Main Event'")
        st.write("The big, sharp spike in the middle.")
        st.error("**What's happening?** The main pumps (ventricles) fire! This big electrical surge causes the powerful contraction that sends blood to your entire body.")
        st.write("*Note: The very tip of this spike is called the **R-Peak**.*")

    with tab3:
        st.markdown("### The 'Reset' Phase")
        st.write("The smoother bump at the end.")
        st.info("**What's happening?** The heart muscle is electrically 'recharging' and relaxing, getting ready for the next beat.")

st.markdown("---")

# --- SECTION 2: WHAT OUR AI LOOKS FOR ---
st.header("2. What does 'Arrhythmia' look like?")
st.write(
    "Our AI model doesn't know biology, but it's amazingly good at finding patterns. "
    "It learned that a healthy beat usually follows the neat pattern above."
)
st.write("When a beat looks 'messy'—too wide, weirdly shaped, or missing a wave—it flags it as **Abnormal**.")

# Create side-by-side comparison
comp1, comp2 = st.columns(2)

with comp1:
    st.markdown("### ✅ Normal Beat")
    # Re-plot normal beat purely for comparison
    fig_n, ax_n = plt.subplots(figsize=(6, 3))
    ax_n.plot(normal_beat, color='green', linewidth=2)
    ax_n.set_title("Clear, Sharp QRS")
    # Hide axes for a cleaner look
    ax_n.set_yticks([])
    ax_n.set_xticks([])
    st.pyplot(fig_n)
    st.caption("Notice the sharp, narrow spike and clear P and T waves.")

with comp2:
    st.markdown("### 🚨 Abnormal (Arrhythmia)")
    # A manually created "messed up" beat example
    abnormal_beat = [0,0,0.05,0.1,0.15,-0.1, -0.3, 0.6, 0.8, 0.9, 0.7, 0.5, 0.2, -0.1,-0.2,-0.1,0,0.1,0.2,0.1,0,0,0,0,0,0,0,0,0,0,0]
    
    fig_a, ax_a = plt.subplots(figsize=(6, 3))
    ax_a.plot(abnormal_beat, color='red', linewidth=2)
    ax_a.set_title("Wide, Bizarre Shape")
    ax_a.set_yticks([])
    ax_a.set_xticks([])
    st.pyplot(fig_a)
    st.caption("This is a Ventricular beat. It starts in the wrong place, making the signal slow, wide, and 'weird' looking.")