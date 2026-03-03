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
                                "Tensor Computations (Pytorch)", "Tran and Test a Deep Learning Model"])

# Main app Functionality
if operation == "Matrix Multiplication (Numpy)":
    st.header("Matrix Multiplication with Numpy")
    
    # User inputs for matrix dimensions
    rows = st.number_input("Number of rows:", min_value = 1, max_value =  1000, value = 3)
    cols = st.number_input("Number of columns", min_value = 1, max_value= 1000, value = 3)
    
    if st.button("Generate and Multiply Matrices"):
        # Generate random matrices
        matrix_a = np.random.rand(rows, cols)
        matrix_b = np.random.rand(cols, rows)
        
        # Perform matrix multiplication
        result = np.dot(matrix_a, matrix_b)
        
        st.write("Matrix A:")
        st.write(matrix_a)
        
        st.write("Matrix B:")
        st.wrtie(matrix_b)
        
        st.write("Resultant Matrix:")
        st.write(result)
        
elif operation == "DataFrme Operations (Pandas)":
    st.header("DataFrame Operations with Pandas")
    
    # Generate a random DataFrame
    rows = st.number_input("Number of rows:", min_value = 0, max_value = 1000, value = 10)
    
    if st.button("Generate DataFrame"):
        df = pd.DataFrame(np.random.rand(rows, 5))
        columns= ["A", "B", "C", "D", "E"]

        st.write("Randomly Generated DataFrame:")
        st.write(df)
        
        st.write("Column Summaries:")
        st.write(df.describe())