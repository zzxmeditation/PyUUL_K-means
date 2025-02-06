# K-Means Clustering with PyUUL

This is a Streamlit-based web application for performing K-Means clustering on molecular structures using PyUUL. The application allows users to upload ligand and peptide files, process them, and cluster them based on their molecular features.
We also host the model on the [HuggingFace Spaces](https://huggingface.co/spaces/ouczzx1999/pyuul-kmeans), so you can start your inference using just your browser.
## Features
- Upload ligand (PDB format)
- Perform K-Means clustering using PyUUL and KMeans from `sklearn`
- Download clustering results as CSV and PDB files
- Support for user-provided PDB databases (peptide files (ZIP format))

## Installation
This app is designed to run on Hugging Face Spaces using Streamlit. To run locally, follow these steps:

```bash
# Clone the repository
git clone https://huggingface.co/spaces/ouczzx1999/pyuul-kmeans
cd pyuul-kmeans
# Install dependencies
pip install -r requirements.txt
# Run the application
streamlit run app.py
```

## Usage
1. Upload a ligand file (`.pdb`).
2. Optionally, upload a peptide ZIP file containing multiple PDB files.
3. Choose the number of clusters and iterations for K-Means clustering.
4. Run clustering and view the results.
5. Download the result folder containing clustered PDB files and CSV outputs.

## Requirements
- `Python 3.x`
- `Streamlit`
- `numpy`
- `pandas`
- `torch`
- `scikit-learn`
- `pyuul_kmeans`
- `shutil`, `os`, `zipfile`, `random`

## Author
Developed by **Zixuan Zhang**, Ocean University of China.

## License
This project is open-source and available under the MIT License.
