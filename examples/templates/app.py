import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Function Plotter')

# User inputs
function_str = st.text_input("Enter a function of 'x' (e.g., sin(x), x^2):", "sin(x)")
x_min = st.number_input("Minimum x value", value=-10.0)
x_max = st.number_input("Maximum x value", value=10.0)

# Convert user function into a usable function
def user_function(x):
    try:
        return eval(function_str)
    except Exception as e:
        st.error(f"Error in computing function: {e}")
        return x * 0  # Return an array of zeros

if st.button("Plot"):
    # Generate x values
    x = np.linspace(x_min, x_max, 400)
    y = user_function(x)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label=function_str)
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.legend()
    ax.grid(True, which='both')

    # Display plot
    st.pyplot(fig)