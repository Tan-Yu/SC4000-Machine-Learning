# Media Campaign Cost Prediction: An Advanced Ensemble-Based Approach

## Project Overview
This project presents our solution for the "Regression with a Tabular Media Campaign Cost Dataset" Kaggle competition (Playground Series Season 3 Episode 11). Our approach leveraged a diverse ensemble of machine learning models, strategic feature engineering, and effective cross-validation techniques to achieve top 3 placement on both public and private leaderboards with an RMSLE of 0.29260, representing a significant improvement over baseline approaches.

## Key Features
- Comprehensive exploratory data analysis of synthetic tabular data
- Strategic feature engineering and selection
- Advanced model tuning across multiple algorithm families
- Sophisticated ensemble methodology with weighted blending
- Detailed visualization and error analysis
- 70.2% improvement over simple linear regression baseline

## Setup and Installation

### Using start.sh (Linux/Mac) or start.bat (Windows)
Setup scripts are provided for easy installation:

#### For Linux/Mac:
```bash
# Give execution permission
chmod +x start.sh

# Run the setup script
./start.sh
```

#### For Windows:
```
# Run the batch file
start.bat
```

The setup scripts will:
- Create a virtual environment
- Install all required dependencies
- Check for the necessary datasets
- Configure the environment

### Manual Setup
All required packages are listed in `requirements.txt`. Install using:
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Jupyter (optional)
pip install jupyter
```

## Dataset
The project uses two datasets:
- `data/train.csv`: Training dataset with 360,336 samples
- `data/test.csv`: Test dataset for generating predictions with 240,224 samples

The datasets contain 15 features related to store characteristics, product attributes, and sales metrics, with the target variable being the campaign cost (ranging from approximately 50 to 160).

## Model Pipeline
1. **Data Preprocessing**
   - Target transformation using log1p for RMSLE optimization
   - Feature engineering including merging similar features (salad_bar and prepared_food)
   - Various encoding techniques for categorical features (One-hot, Target, Ordinal)

2. **Model Development**
   - **Baseline models**:
     - Simple Linear Regression
     - Ridge with Polynomial Features (degree 2)
     - Basic Random Forest
   - **Advanced models** (19 total):
     - Linear models with polynomial features
     - Tree-based models (Random Forest, Extra Trees)
     - Gradient Boosting variants (LightGBM, XGBoost, CatBoost, HistGradientBoosting)
     - Specialized models with different encoding techniques and feature sets

3. **Ensemble Strategy**
   - Meta-model approach using Ridge Regression
   - Weighted ensemble of 19 diverse base models
   - Top weights assigned to LightGBM (0.233057), DART (0.214060), and RF (0.186047)
   - Grouped training approach for handling duplicates

## Performance
- Simple Linear Regression (baseline): 0.32547 RMSLE
- Ridge with Polynomial Features (degree 2): 0.30814 RMSLE
- Basic Random Forest: 0.29300 RMSLE
- Best single model (LightGBM): 0.085809 RMSE / ~0.29285 RMSLE
- Simple average ensemble: ~0.29275 RMSLE  
- Weighted ensemble (final solution): 0.29260 RMSLE
- Competition ranking: 3rd out of 954 submissions (top 0.3%)

## Visualizations
The script generates 29 detailed visualizations in the `figures` directory, including:
- Feature correlation heatmap
- Target distribution before and after transformation
- Feature cardinality analysis
- Duplicate distribution analysis
- Pairwise relationships between key features
- Cross-validation diagram
- Feature importance charts
- Feature transformation comparison
- Model performance metrics
- Ensemble architecture and weights
- Error distribution analysis
- Ablation studies for features and model components

## Key Insights
1. **Feature Importance**: 
   - avg_cars_at home(approx).1 (27.6%)
   - total_children (25.8%)
   - num_children_at_home (22.1%)
   
2. **Model Diversity**: The ensemble benefits from diverse model types (linear, tree-based, and gradient boosting)

3. **Computational Efficiency**: Our grouped training approach significantly reduced computation time without sacrificing performance, reducing training dataset size from 360,336 to about 3,079 samples

4. **Feature Encoding**: Different encoding techniques produced varying results for different models

5. **Error Distribution**: Models showed systematic underestimation for high target values and higher error rates for both very low and very high target values

## Execution

### Python Script
Run the main script to generate all visualizations and predictions:
```bash
python src/assignment.py
```

### Jupyter Notebook (Recommended)
We've converted the Python script to a Jupyter notebook for better interactivity and visualization:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the notebook file in the Jupyter interface

The notebook version offers:
- Interactive execution of code cells
- Inline visualization of figures
- Easier experimentation with model parameters
- Better documentation with markdown cells

## Output Files
- Prediction files: `submission_simple_avg.csv`, `submission_weighted.csv`, `submission_top10.csv`
- Visualization figures saved to the `figures` directory
- Jupyter notebook with all code and visualizations (when using notebook version)

## Future Improvements
- Hyperparameter optimization with Bayesian methods
- Feature interaction discovery
- Deeper neural network approaches
- Time-series analysis for temporal aspects
- Stacking with multiple meta-models
