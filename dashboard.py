from flask import Flask, request, render_template
import joblib
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load models
models = {}
for year in range(1, 6):
    try:
        models[year] = joblib.load(f'model_year_{year}.pkl')
        logging.info(f"Loaded model for year {year}")
    except FileNotFoundError:
        logging.warning(f"Model for year {year} not found")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get input
            interim_multiple = float(request.form['interim_multiple'])
            staleness_freq = float(request.form['staleness_freq'])
            markdown_freq = float(request.form['markdown_freq'])
            year = int(request.form['year'])
            
            # Validate inputs
            if interim_multiple < 0:
                return render_template('index.html', prediction="Invalid input: Interim Multiple must be non-negative (≥ 0).")
            if not (0 <= staleness_freq <= 1):
                return render_template('index.html', prediction="Invalid input: Staleness Frequency must be between 0 and 1.")
            if not (0 <= markdown_freq <= 1):
                return render_template('index.html', prediction="Invalid input: Markdown Frequency must be between 0 and 1.")
            if year not in range(1, 6):
                return render_template('index.html', prediction="Invalid input: Year must be between 1 and 5.")
            
            # Predict
            if year in models:
                model = models[year]
                X = pd.DataFrame([[interim_multiple, staleness_freq, markdown_freq]], 
                                columns=['interim_multiple', 'staleness_freq', 'markdown_freq'])
                prob = model.predict_proba(X)[0][1] * 100  # Probability of increase (%)
                prediction = f"Likelihood of valuation increase: {prob:.2f}%"
            else:
                prediction = f"No model available for year {year}"
        except ValueError:
            prediction = "Invalid input: Please ensure all fields are numeric and within the specified ranges."
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    logging.info("Starting Flask dashboard...")
    app.run(debug=True)