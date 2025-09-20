<html>
<head>
  <title>Smart City Analytics Platform</title>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
    h1, h2, h3 { color: #333; }
    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
    code { font-family: monospace; }
    a { color: #0066cc; }
  </style>
</head>
<body>
  <h1>Smart City Analytics Platform</h1>
  <p>This project helps predict traffic congestion in cities using data from sensors and cameras. It’s built with two machine learning models: LSTM (good for time-series data) and Random Forest (great for quick predictions). There’s a dashboard to see traffic stats and a web API to get real-time predictions.</p>

  <h2>What’s Used</h2>
  <ul>
    <li><b>Models</b>: LSTM (for deep learning on time-series traffic data) and Random Forest (for fast, accurate predictions).</li>
    <li><b>Dataset</b>: <a href="https://www.kaggle.com/datasets/ziya07/smart-mobility-traffic-dataset">Kaggle Smart Mobility Traffic Dataset</a> (traffic data like vehicle count, speed, and time).</li>
    <li><b>Tools</b>: Python, Streamlit (dashboard), Flask (API), Docker (containers), MLflow (tracking experiments), DVC (data versioning).</li>
  </ul>

  <h2>Clone and Run Locally</h2>
  <p>Want to run this on your computer? Follow these steps:</p>
  <ol>
    <li><b>Clone the project</b>:
      <pre><code>git clone https://github.com/Karthik-099/smart-city-analytics.git
cd smart-city-analytics</code></pre>
    </li>
    <li><b>Set up a virtual environment</b>:
      <pre><code>python -m venv traffic
source traffic/bin/activate  # On Windows: traffic\Scripts\activate</code></pre>
    </li>
    <li><b>Install dependencies</b>:
      <pre><code>pip install -r requirements.txt  # Full project (includes LSTM)
# Or for dashboard only:
pip install -r requirements_dashboard.txt</code></pre>
    </li>
    <li><b>Get the dataset</b>:
      <p>Download <a href="https://www.kaggle.com/datasets/ziya07/smart-mobility-traffic-dataset">traffic_data.csv</a> from Kaggle and place it in <code>data/raw/traffic_data.csv</code>.</p>
    </li>
    <li><b>Run the data pipeline</b> (to process data and train models):
      <pre><code>dvc repro</code></pre>
    </li>
    <li><b>Run the dashboard</b>:
      <pre><code>streamlit run deployment/app.py</code></pre>
      <p>Visit <a href="http://localhost:8501">http://localhost:8501</a> to see traffic stats, charts, and predictions.</p>
    </li>
    
      <pre><code>python deployment/api.py</code></pre>
      <p>Test it:
      <pre><code>curl -X POST -H "Content-Type: application/json" -d '{"Hour": 8, "DayOfWeek": 1, "IsWeekend": 0, "RushHour": 1, "VehicleCount_lag1": 150, "AverageSpeed_lag1": 30, "VehicleCount_rolling_mean": 145}' http://localhost:5000/predict</code></pre>
      <p>You should see: <code>{"congestion": 1}</code> (means congestion).</p>
    </li>
  </ol>

  <h2>Pull and Run with Docker</h2>
  <p>Prefer running it with Docker? Here’s how:</p>
  <ol>
    <li><b>Pull the dashboard image</b>:
      <pre><code>docker pull karthik75/smart-city-analytics-dashboard:latest</code></pre>
    </li>
    <li><b>Run the dashboard</b>:
      <pre><code>docker run -p 8501:8501 karthik75/smart-city-analytics-dashboard:latest</code></pre>
      <p>Visit <a href="http://localhost:8501">http://localhost:8501</a> for the dashboard.</p>
    </li>

      <pre><code>docker pull karthik75/smart-city-analytics-api:latest</code></pre>
    </li>
    <li><b>Run the API</b>:
      <pre><code>docker run -p 5000:5000 karthik75/smart-city-analytics-api:latest</code></pre>
      <p>Test it:
      <pre><code>curl -X POST -H "Content-Type: application/json" -d '{"Hour": 8, "DayOfWeek": 1, "IsWeekend": 0, "RushHour": 1, "VehicleCount_lag1": 150, "AverageSpeed_lag1": 30, "VehicleCount_rolling_mean": 145}' http://localhost:5000/predict</code></pre>
      <p>Expect: <code>{"congestion": 1}</code>.</p>
    </li>
  </ol>

  <h2>Where to Access</h2>
  <ul>
    <li><b>Dashboard</b>: Open <a href="http://localhost:8501">http://localhost:8501</a> after running locally or via Docker.</li>
  
    <li><b>Experiments</b>: Run <code>mlflow ui</code> and visit <a href="http://localhost:5000">http://localhost:5000</a> to see model logs.</li>
  </ul>

</body>
</html>
