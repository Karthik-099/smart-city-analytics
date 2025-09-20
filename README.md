<html>
<head>
  <title>Smart City Analytics Platform</title>
 
</head>
<body>
  <h1>Smart City Analytics Platform</h1>
  <p>This project predicts traffic congestion using sensor and camera data. It uses two models: LSTM for time-series patterns and Random Forest for fast predictions. Includes a dashboard for stats and an API for real-time predictions.</p>

  <h2>What’s Used</h2>
  <ul>
    <li><b>Models</b>: LSTM (deep learning for time-series traffic data) and Random Forest (quick, accurate predictions).</li>
    <li><b>Dataset</b>: <a href="https://www.kaggle.com/datasets/ziya07/smart-mobility-traffic-dataset">Kaggle Smart Mobility Traffic Dataset</a> (vehicle count, speed, time).</li>
    <li><b>Tools</b>: Python, Streamlit (dashboard), Flask (API), Docker (containers), MLflow (experiment tracking), DVC (data and model versioning).</li>
  </ul>

  <h2>Clone and Run Locally</h2>
  <p>To run this on your computer:</p>
  <ol>
    <li><b>Clone the project</b>:
      <pre><code>git clone git@github.com:Karthik-099/smart-city-analytics.git
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
    <li><b>Get dataset and models</b>:
      <p>Option 1: Download <a href="https://www.kaggle.com/datasets/ziya07/smart-mobility-traffic-dataset">traffic_data.csv</a> to <code>data/raw/traffic_data.csv</code>.</p>
      <p>Option 2: Pull dataset and models (Random Forest, LSTM) with DVC:</p>
      <pre><code>dvc pull</code></pre>
    </li>
    <li><b>Run the data pipeline</b> (processes data, trains models):
      <pre><code>dvc repro</code></pre>
    </li>
    <li><b>Run the dashboard</b>:
      <pre><code>streamlit run deployment/app.py</code></pre>
      <p>Visit <a href="http://localhost:8501">http://localhost:8501</a> for stats, charts, predictions.</p>
    </li>
    <li><b>Run the API</b>:
      <pre><code>python deployment/api.py</code></pre>
      <p>Test it:
      <pre><code>curl -X POST -H "Content-Type: application/json" -d '{"Hour": 8, "DayOfWeek": 1, "IsWeekend": 0, "RushHour": 1, "VehicleCount_lag1": 150, "AverageSpeed_lag1": 30, "VehicleCount_rolling_mean": 145}' http://localhost:5000/predict</code></pre>
      <p>Expect: <code>{"congestion": 1}</code> (congestion).</p>
    </li>
  </ol>

  <h2>Pull and Run with Docker</h2>
  <p>Using Docker? Here’s how:</p>
  <ol>
    <li><b>Pull the dashboard image</b>:
      <pre><code>docker pull karthik75/smart-city-analytics-dashboard:latest</code></pre>
    </li>
    <li><b>Run the dashboard</b>:
      <pre><code>docker run -p 8501:8501 karthik75/smart-city-analytics-dashboard:latest</code></pre>
      <p>Visit <a href="http://localhost:8501">http://localhost:8501</a>.</p>
    </li>
    <li><b>Pull the API image</b>:
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
    <li><b>Dashboard</b>: <a href="http://localhost:8501">http://localhost:8501</a> (local or Docker).</li>
    <li><b>API</b>: <a href="http://localhost:5000/predict">http://localhost:5000/predict</a> (after running API).</li>
    <li><b>Experiments</b>: Run <code>mlflow ui</code> and visit <a href="http://localhost:5000">http://localhost:5000</a>.</li>
  </ul>

  <h2>Questions?</h2>
  <p>Visit <a href="https://github.com/Karthik-099/smart-city-analytics">github.com/Karthik-099/smart-city-analytics</a>. Open an issue if stuck!</p>
</body>
</html>
