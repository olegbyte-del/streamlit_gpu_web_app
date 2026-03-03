# Download necessary prerequisites
# 1. Ubuntu 22.04 Cloud GPU Server
# 2. Cuda Toolkit and cuDNN installed
# 3. Root or Sudo 

import streamlit as st
import numpy as np
import pandas as pd
import torch

# Title
st.title("Streamlit App with GPU Computations")

# Sidebar for user input
st.sidebar.header("Choose an Operation")
operation = st.sidebar.selectbox("Select a computation:",
                                ["Matrix Multiplication (Numpy)", "DataFrame Operations (Pandas)",
                                "Tensor Computtaions (Pytorch)", "Tran and Test a Deep Learning Model"])

# Main app Functionality
if operation == "Matrix Multiplication (Numpy)":
    st.header("Matrix Multiplication with Numpy")
    
    # User inputs for matrix dimensions
    rows = st.number_input("Number of rows:", min_value = 1, max_value =  1000, value = 3)
    cols = st.number_input("Number of columns", min_value = 1, max_value= 1000, value = 3)
    