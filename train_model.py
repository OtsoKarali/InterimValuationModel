import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data from SQLite
logging.info("Loading data from SQLite...")
engine = create_engine('sqlite:///pe_data.db')
data = pd.read_sql('SELECT * FROM investments', engine)

# Clean data (Appendix A.2)
data = data.dropna(subset=['interim_multiple', 'staleness_freq', 'markdown_freq', 'exit_multiple'])
data = data[data['exit_multiple'] >= 0]

# Features and target
features = ['interim_multiple', 'staleness_freq', 'markdown_freq']
target = 'valuation_increase'

# Train and evaluate by year
results = {}
models = {}
for year in range(1, 6):
    logging.info(f"Training model for year {year}...")
    year_data = data[data['year'] == year]
    if len(year_data) < 10:
        logging.warning(f"Skipping year {year}: insufficient data")
        continue
    
    X = year_data[features]
    y = year_data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    models[year] = model
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[year] = {
        'accuracy': accuracy,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score']
    }
    
    logging.info(f"Year {year} - Accuracy: {accuracy:.4f}")

# Save models
for year, model in models.items():
    joblib.dump(model, f'model_year_{year}.pkl')
    logging.info(f"Saved model for year {year} to model_year_{year}.pkl")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
logging.info("Results saved to model_results.csv")

# Print results
print("Model Performance by Year:")
for year, metrics in results.items():
    print(f"\nYear {year}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (class 1): {metrics['precision']:.4f}")
    print(f"Recall (class 1): {metrics['recall']:.4f}")
    print(f"F1-Score (class 1): {metrics['f1_score']:.4f}")