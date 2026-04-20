# TerraVision - Satellite Land Use Classification

TerraVision is a deep learning web application that classifies satellite imagery into 10 land use categories using a fine-tuned ResNet18 model trained on the EuroSAT dataset. Upload any satellite image through the web interface and get an instant terrain prediction with a confidence score and ranked probability breakdown.

## Tech Stack

| Layer | Tools |
|---|---|
| Model | PyTorch · ResNet18 (transfer learning + fine-tuning) |
| Dataset | EuroSAT RGB (27,000 satellite images, 10 classes) |
| Web App | Streamlit |
| Image Processing | TorchVision |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11 |

## Classes

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

## Live Demo

🔗 [terravision.streamlit.app](https://ujasb-terravision.streamlit.app)

## Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/UjasBanke/TerraVision.git
cd TerraVision

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the web app (model.pth is included — no training needed)
streamlit run app.py
```

App runs at `http://localhost:8501`

## Want to retrain the model?

```bash
python main.py
```

Downloads the EuroSAT dataset (~90MB) on first run. Trains in two phases — Phase 1 trains the classifier head only (10 epochs), Phase 2 fine-tunes the deeper ResNet layers (10 epochs). Best checkpoint saved as `model.pth`.
