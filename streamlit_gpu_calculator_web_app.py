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
                                "Tensor Computations (Pytorch)", "Train and Test a Deep Learning Model"])

# Main app Functionality
if operation == "Matrix Multiplication (Numpy)":
    st.header("Matrix Multiplication with Numpy")
    
    # User inputs for matrix dimensionss
    rows = st.number_input("Number of rows:", min_value=1, max_value=1000, value=3)
    cols = st.number_input("Number of columns", min_value=1, max_value=1000, value=3)
    
    if st.button("Generate and Multiply Matrices"):
        # Generate random matrices
        matrix_a = np.random.rand(rows, cols)
        matrix_b = np.random.rand(cols, rows)
        
        # Perform matrix multiplication
        result = np.dot(matrix_a, matrix_b)
        
        st.write("Matrix A:")
        st.write(matrix_a)
        
        st.write("Matrix B:")
        st.write(matrix_b)
        
        st.write("Resultant Matrix:")
        st.write(result)
        
elif operation == "DataFrame Operations (Pandas)":
    st.header("DataFrame Operations with Pandas")
    
    # Generate a random DataFrame
    rows = st.number_input("Number of rows:", min_value=0, max_value=1000, value=10)
    
    if st.button("Generate DataFrame"):
        df = pd.DataFrame(np.random.rand(rows, 5),
        columns= ["A", "B", "C", "D", "E"])

        st.write("Randomly Generated DataFrame:")
        st.write(df)
        
        st.write("Column Summaries:")
        st.write(df.describe())
    
elif operation == "Tensor Computations (Pytorch)":
    st.header("Tensor Computations with Pytorch")
    
    #Tensor size input
    tensor_size = st.number_input("Tensor size:", min_value=1, max_value=10000, value=3)
    
    if st.button("Generate Tensor and Compute"):
        # Generate random tensor on GPU
        tensor_a = torch.rand(tensor_size, tensor_size, device="cuda")
        tensor_b = torch.rand(tensor_size, tensor_size, device="cuda")
        
        # Perform matrix multiplication on GPU
        result = torch.matmul(tensor_a, tensor_b)
        
        st.write("Tensor A:")
        st.write(tensor_a.cpu().numpy())
        
        st.write("Tensor B:")
        st.write(tensor_b.cpu().numpy())
        
        st.write("Resultant Tensor:")
        st.write(result.cpu().numpy())
        
elif operation == "Train and Test a Deep Learning Model":
    st.header("Train and Test a Deep Learning Model")
    
    # User input for dataset size
    num_samples = st.number_input("Number of Samples:", min_value=100, max_value=10000, value=1000)
    num_features = st.number_input("Number of Features:", min_value=1, max_value=100, value=10)
    
    if st.button("Train Model"):
        # Generate random data
        X = torch.rand(num_samples, num_features, device="cuda")
        y = torch.sum(X, dim=1) + torch.randn(num_samples, device="cuda") * 0.1
        
        # Define a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1)
        ).to("cuda")
        
        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train the model
        epochs = 50
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions.squeeze(), y)
            loss.backward()
            optimizer.step()
            
        st.success(f"Training complete! Final LOss:{loss.item():.4f}")
        
        # Test the model
        test_data = torch.rand(10, num_features, device="cuda")
        test_predication = model(test_data)
        st.write("Test Data:")
        st.write(test_data.cpu().numpy())
        
        st.write("Predictions:")
        st.write(test_predication.cpu().detach().numpy())