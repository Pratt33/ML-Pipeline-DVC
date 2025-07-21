# ML Recommendation System

## Overview
This project is a machine learning pipeline for product recommendations using Amazon Fine Food Reviews. It uses DVC for data and model versioning, Git for code management, and Streamlit for deployment and visualization.

---

## Project Structure
- **src/**: Data ingestion, preprocessing, training, and evaluation scripts
- **api/**: FastAPI backend (optional, for local API use)
- **app.py**: Streamlit web app for recommendations
- **visualizations.py**: Script to generate data visualizations
- **data/**: Raw and processed data (managed by DVC)
- **models/**: Trained models (managed by DVC)
- **metrics/**: Evaluation metrics (managed by DVC)
- **notebook/**: Jupyter notebooks for exploration

---

## Setup
1. **Clone the repository**
   ```sh
git clone <your-repo-url>
cd <project-directory>
```
2. **Install dependencies**
   ```sh
pip install -r requirements.txt
```
3. **Install DVC (if not already installed)**
   ```sh
pip install dvc
```
4. **Pull data and models with DVC**
   ```sh
dvc pull
```

---

## Usage
### 1. **Run the Streamlit App**
```sh
streamlit run app.py
```
- Open your browser at [http://localhost:8501](http://localhost:8501)
- Enter a user ID to get recommendations

### 2. **Run Visualizations**
```sh
python visualizations.py
```
- Plots will be saved in the `visualizations/` directory

### 3. **Reproduce the Pipeline**
```sh
dvc repro
```
- This will run the full pipeline: data ingestion, preprocessing, training, and evaluation

---

## Best Practices
- Use **DVC** to manage large files (data, models, metrics)
- Use **Git** for code and pipeline versioning
- Use **Streamlit** for interactive deployment (no Docker needed)
- Keep `dvc.yaml` and `dvc.lock` tracked in Git
- Never commit large data/model files directly—let DVC handle them

---

## Deployment
- Deploy the app using Streamlit Cloud, or any server with Python and Streamlit installed
- No Docker or containerization required

---

## Author
- [Your Name] 