# 📈 Sales Forecasting Using Machine Learning

A beginner-friendly Machine Learning project that analyzes historical sales data and predicts future sales using Linear Regression and Random Forest Regression.

## 📌 Project Overview

Sales forecasting helps businesses estimate future sales based on past sales patterns. This project uses Python, Pandas, Matplotlib, Scikit-learn, and Streamlit to analyze sales data, compare ML models, and forecast upcoming sales.

The project includes a Streamlit dashboard for interactive exploration and prediction.

## 🎯 Project Objectives

* Analyze historical sales data.
* Visualize sales trends using graphs.
* Prepare data for Machine Learning.
* Train and compare regression models.
* Forecast sales for future months.
* Display results through an interactive dashboard.

## 🛠️ Technologies Used

* **Python** — Programming language
* **Pandas** — Data handling and analysis
* **NumPy** — Numerical operations
* **Matplotlib** — Data visualization
* **Scikit-learn** — Machine Learning models and evaluation
* **Streamlit** — Interactive web dashboard
* **Jupyter Notebook** — Experimentation and development

## 🤖 Machine Learning Models

### 1. Linear Regression

A regression algorithm that learns a relationship between input features and sales.

### 2. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees to predict sales.

The models are compared using evaluation metrics such as MAE and RMSE.

## 📊 Project Features

* Historical sales visualization
* Sales data overview and statistics
* Feature engineering using year, month, previous sales, and rolling averages
* Chronological training and testing
* Model comparison
* Future sales forecasting
* Forecast table and graph
* Downloadable forecast CSV file

## 📁 Project Structure

```text
only-ml-project/
│
├── app.py
├── sales.csv
├── sales_forecasting.ipynb
├── README.md
├── requirements.txt
└── sales_forecast.csv
```

**Note:** `sales_forecast.csv` is the generated forecast output. It may appear after running the forecasting code or downloading the result. Your actual folder may differ slightly.

## ⚙️ Installation and Setup

### Step 1: Install Python

Install Python if it is not already installed on your computer.

### Step 2: Open the Project Folder

Open the project folder in VS Code.

### Step 3: Open the Terminal

In VS Code, select **Terminal → New Terminal**.

### Step 4: Install Dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Streamlit Dashboard

Run this command from the project folder:

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

## 📂 Dataset

The project uses a CSV dataset containing monthly sales records.

The notebook can generate sample sales data for practice. If you use generated data, the results are illustrative and do not represent actual business sales.

## 📈 Evaluation Metrics

* **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted sales.
* **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors.

Lower values generally indicate smaller prediction errors on the evaluated dataset.

## 🔮 Forecasting Output

The project can generate predictions for upcoming months and display them in a table and graph. The forecast can also be downloaded as a CSV file.

Predictions depend on the available data and model assumptions; they are estimates, not guaranteed future sales.

## 🚀 Future Improvements

* Add more historical sales data.
* Try additional forecasting algorithms.
* Improve feature engineering.
* Add interactive date filters.
* Deploy the dashboard online.
* Evaluate predictions on real-world sales data.

## 👩‍💻 Author

**Subuhi Khan**

B.Tech CSE (AI & ML) Student

GitHub: [@codewithsubuhi](https://github.com/codewithsubuhi)

## 📜 License

This project is intended for learning and educational purposes.
