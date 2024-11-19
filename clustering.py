import tkinter as tk
from tkinter import messagebox
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample customer data (you can replace this with your actual dataset)
data = {
    'customer_id': range(1, 21),
    'total_transactions': [15, 20, 35, 40, 10, 12, 18, 25, 30, 28, 50, 45, 60, 20, 22, 18, 14, 38, 41, 55],
    'average_transaction_value': [50, 60, 55, 70, 40, 45, 48, 65, 68, 60, 90, 85, 95, 50, 52, 47, 43, 75, 80, 100],
    'tenure_months': [12, 24, 36, 48, 6, 8, 15, 30, 40, 35, 60, 50, 70, 20, 22, 18, 16, 45, 50, 65],
}

# Create DataFrame from sample data
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['total_transactions', 'average_transaction_value', 'tenure_months']])

# Fit KMeans clustering (using 3 clusters for this example)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Function to predict the cluster of the new customer based on input data
def predict_cluster():
    try:
        # Get input data from the user
        total_transactions = int(entry_transactions.get())
        avg_transaction_value = float(entry_avg_transaction_value.get())
        tenure_months = int(entry_tenure_months.get())
        
        # Prepare input for clustering (same scaling as the training data)
        new_customer = np.array([[total_transactions, avg_transaction_value, tenure_months]])
        new_customer_scaled = scaler.transform(new_customer)
        
        # Predict the cluster
        cluster = kmeans.predict(new_customer_scaled)[0]
        
        # Display the cluster in a message box
        messagebox.showinfo("Customer Cluster", f"The customer belongs to Cluster {cluster + 1}.")
        
    except ValueError:
        # Handle invalid inputs
        messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")

# Create the main window
root = tk.Tk()
root.title("Customer Segmentation")

# Create and place widgets in the window
label_transactions = tk.Label(root, text="Total Transactions:")
label_transactions.grid(row=0, column=0)
entry_transactions = tk.Entry(root)
entry_transactions.grid(row=0, column=1)

label_avg_transaction_value = tk.Label(root, text="Average Transaction Value:")
label_avg_transaction_value.grid(row=1, column=0)
entry_avg_transaction_value = tk.Entry(root)
entry_avg_transaction_value.grid(row=1, column=1)

label_tenure_months = tk.Label(root, text="Tenure (Months):")
label_tenure_months.grid(row=2, column=0)
entry_tenure_months = tk.Entry(root)
entry_tenure_months.grid(row=2, column=1)

# Button to predict the cluster
predict_button = tk.Button(root, text="Predict Cluster", command=predict_cluster)
predict_button.grid(row=3, column=0, columnspan=2)

# Start the Tkinter event loop
root.mainloop()
