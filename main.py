import os
import pathlib
import random
import shutil
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import torch
from pyuul_kmeans import VolumeMaker, utils
from sklearn.cluster import KMeans

# Constants for file paths
UPLOAD_FOLDER_PATH = "lig"
TMP_FOLDER_PATH = "tmp"
PDB_FOLDER_PATH = "4pepzip"
USER_PDB_FOLDER_PATH = "user_pdb"
RESULT_FOLDER_PATH = "result"
RESULT_PDB_FOLDER_PATH = os.path.join(RESULT_FOLDER_PATH, "pdb")

# Device selection
device = "cpu"


def set_random_seed(seed=100):
    """
    Set random seed for reproducibility across multiple libraries.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def clear_folder(folder_path):
    """
    Clear all files in a specified folder.

    Args:
        folder_path (str): Path to the folder to be cleared.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass


def clean_not_folder(files, folder):
    """
    Remove files in a folder that are not in the specified list.

    Args:
        files (list): List of filenames to keep.
        folder (str): Path to the folder to clean.
    """
    for filename in os.listdir(folder):
        if filename not in files:
            os.remove(os.path.join(folder, filename))


def pyuul_clustering(folder, n_clusters):
    """
    Perform clustering using PyUUL and KMeans.

    Args:
        folder (str): Path to the folder containing PDB files.
        n_clusters (int): Number of clusters for KMeans.

    Returns:
        list: List of PDB filenames in the same cluster as the ligand.
    """
    coords, atname, pdbname, pdb_num = utils.parsePDB(folder)
    radius = utils.atomlistToRadius(atname)

    PointCloudSurfaceObject = VolumeMaker.PointCloudVolume(device=torch.device(device))
    coords, radius = coords.to(device), radius.to(device)

    SurfacePoitCloud = PointCloudSurfaceObject(coords, radius)
    feature = SurfacePoitCloud.view(pdb_num, -1).cpu()

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, init="k-means++", random_state=100)
    y = kmeans.fit_predict(feature)

    result_dict = dict(zip(pdbname, y))
    ligand_class = result_dict['lig.pdb']

    return [key for key, value in result_dict.items() if value == ligand_class]


outlist = []


def kmeans_clustering(ligand_path, tmp_folder, n_clusters, iterations):
    """
    Perform iterative KMeans clustering and save results.

    Args:
        ligand_path (str): Path to the ligand file.
        tmp_folder (str): Path to the temporary folder for processing.
        n_clusters (int): Number of clusters for KMeans.
        iterations (int): Number of iterations to perform.
    """
    for i in range(iterations):
        if i == 0:
            shutil.copytree(ligand_path, tmp_folder, dirs_exist_ok=True)
            output = pyuul_clustering(tmp_folder, n_clusters)
            outlist.append(output)
            path_csv = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{i + 1}.csv")
            pd.DataFrame({'Name': output}).to_csv(path_csv, index=False)
        else:
            clean_not_folder(outlist[i - 1], tmp_folder)
            output = pyuul_clustering(tmp_folder, n_clusters)
            outlist.append(output)
            path_csv = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{i + 1}.csv")
            pd.DataFrame({'Name': output}).to_csv(path_csv, index=False)

    final_csv_path = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{iterations}.csv")
    if os.path.exists(final_csv_path):
        final_df = pd.read_csv(final_csv_path)
        pdb_files = final_df['Name'].tolist()

        os.makedirs(RESULT_PDB_FOLDER_PATH, exist_ok=True)

        for pdb_file in pdb_files:
            src_path = os.path.join(tmp_folder, pdb_file)
            dst_path = os.path.join(RESULT_PDB_FOLDER_PATH, pdb_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)


def main():
    """
    Main function to run the Streamlit app.
    """
    peptide_folder_path = PDB_FOLDER_PATH
    clear_folder(RESULT_FOLDER_PATH)
    clear_folder(UPLOAD_FOLDER_PATH)
    clear_folder(RESULT_PDB_FOLDER_PATH)
    clear_folder(USER_PDB_FOLDER_PATH)
    clear_folder(TMP_FOLDER_PATH)

    st.title("K-Means Clustering with PyUUL")

    st.sidebar.header("Input Parameters")
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=100, value=2)
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=1, max_value=100, value=2)

    st.write("### Clustering Parameters")
    st.write(f"Number of Clusters: {num_clusters}")
    st.write(f"Number of Iterations: {num_iterations}")

    st.sidebar.header("Upload Ligand File")
    uploaded_ligand = st.sidebar.file_uploader("Upload a ligand file (PDB format)", type="pdb")

    if uploaded_ligand is not None:
        os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)
        lig_file_path = os.path.join(UPLOAD_FOLDER_PATH, "lig.pdb")
        with open(lig_file_path, "wb") as f:
            f.write(uploaded_ligand.getbuffer())
        st.success(f"Ligand file uploaded successfully: {uploaded_ligand.name}")

    if 'use_user_pdb' not in st.session_state:
        st.session_state.use_user_pdb = False

    if st.sidebar.button('Use Your PDB Database'):
        st.session_state.use_user_pdb = not st.session_state.use_user_pdb

    if st.session_state.use_user_pdb:
        st.sidebar.header("Upload Peptide Files")
        uploaded_peptides = st.sidebar.file_uploader("Upload peptide files (zip format)", type="zip")

        if uploaded_peptides is not None:
            os.makedirs(USER_PDB_FOLDER_PATH, exist_ok=True)
            peptide_file_path = os.path.join(USER_PDB_FOLDER_PATH, uploaded_peptides.name)
            with open(peptide_file_path, "wb") as f:
                f.write(uploaded_peptides.getbuffer())
            st.success("Peptide files uploaded successfully.")
            peptide_folder_path = USER_PDB_FOLDER_PATH
        else:
            peptide_folder_path = PDB_FOLDER_PATH
            st.warning("No files uploaded or uploaded_peptides is None.")

    if st.button("Run Clustering"):
        st.write("Running clustering...")
        set_random_seed(100)

        pepzips = pathlib.Path(peptide_folder_path)
        for pepzip in pepzips.iterdir():
            azip = zipfile.ZipFile(pepzip)
            azip.extractall(path='tmp')
        kmeans_clustering(UPLOAD_FOLDER_PATH, TMP_FOLDER_PATH, num_clusters, num_iterations)
        st.write("Clustering completed!")

        st.write("### Clustering Results")
        for i in range(num_iterations):
            result_file = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{i + 1}.csv")
            if os.path.exists(result_file):
                result_df = pd.read_csv(result_file)
                st.write(f"Iteration {i + 1} Results:")
                st.dataframe(result_df)

        st.write("### Download Result Folder")
        if os.path.exists(RESULT_FOLDER_PATH):
            shutil.make_archive("result", "zip", RESULT_FOLDER_PATH)
            with open("result.zip", "rb") as f:
                result_zip_data = f.read()
            st.download_button(
                label="Download Result Folder (ZIP)",
                data=result_zip_data,
                file_name="result.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
