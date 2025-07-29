# Stock-Market-analsyis

This notebook implements a Graph Neural Network (GNN)-based stock prediction model using historical stock data and correlation-based graph structures.

🔍 Main Components:
Libraries Used:

PyTorch and PyTorch Geometric for building the GCN model.

yfinance, pandas, numpy for data fetching and processing.

scikit-learn for scaling and splitting data.

matplotlib, seaborn for visualization.

Data Preprocessing:

Reads stock data (stock_data.csv) for companies like AAPL, MSFT, GOOGL, AMZN, META, NVDA.

Handles missing values and normalizes the dataset using MinMaxScaler.

Constructs a correlation matrix and builds a graph structure (edges between stocks) based on a correlation threshold (e.g., 0.7).

Graph Construction:

Nodes = Stocks.

Edges = High correlation (> 0.7) between their price movements.

Uses torch_geometric.data.Data to define graph with edge index and node features.

Model Architecture:

Implements a Graph Convolutional Network (GCN) using GCNConv.

The network predicts future stock prices or movement based on graph-aware features.

Training & Evaluation:

Trains the GCN using historical price data.

Evaluates using metrics like Mean Squared Error (MSE).

Outputs include predicted vs actual stock prices, plotted for visualization.

📈 Key Outputs:
Graph Plot showing correlation-based connections between stocks.

Prediction Charts comparing actual vs predicted stock trends.

Loss Curve over epochs, showing training progress.

MSE Score, indicating prediction accuracy
