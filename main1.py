# 将160000个多肽分为10个压缩包，放到1个文件夹，读取这个文件夹，并解压到tmp文件夹
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
RESULT_PDB_FOLDER_PATH = os.path.join(RESULT_FOLDER_PATH, "pdb")  # 结果中的 PDB 文件夹

# Device selection
device = "cpu"


# Set a random seed function
def set_random_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Clear all files in a folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass  # 如果文件不存在，忽略


# Cleanup files in a given folder that are not part of the 'files' list
def clean_not_folder(files, folder):
    for filename in os.listdir(folder):
        if filename not in files:
            os.remove(os.path.join(folder, filename))


# Perform the clustering
def pyuul_clustering(folder, n_clusters):
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

    # 获取和lig一个类别的四肽分子的名称
    return [key for key, value in result_dict.items() if value == ligand_class]


outlist = []


# Main clustering function
def kmeans_clustering(ligand_path, tmp_folder, n_clusters, iterations):
    # Iterate over the number of iterations
    for i in range(iterations):
        if i == 0:
            shutil.copytree(ligand_path, tmp_folder, dirs_exist_ok=True)
            # 聚类后和ligand一类的pep的名称
            output = pyuul_clustering(tmp_folder, n_clusters)
            outlist.append(output)
            # Fix the path by using os.path.join for proper path handling
            path_csv = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{i + 1}.csv")
            # Save results to CSV
            pd.DataFrame({'Name': output}).to_csv(path_csv, index=False)

        else:
            # Clean up non-relevant files after each iteration
            clean_not_folder(outlist[i - 1], tmp_folder)
            # 聚类后和ligand一类的pep的名称
            output = pyuul_clustering(tmp_folder, n_clusters)
            outlist.append(output)
            path_csv = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{i + 1}.csv")
            # Save results to CSV
            pd.DataFrame({'Name': output}).to_csv(path_csv, index=False)

    # After the last iteration, copy the PDB files listed in the final CSV to result/pdb
    final_csv_path = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{iterations}.csv")
    if os.path.exists(final_csv_path):
        final_df = pd.read_csv(final_csv_path)
        pdb_files = final_df['Name'].tolist()

        # Create the result/pdb folder if it doesn't exist
        os.makedirs(RESULT_PDB_FOLDER_PATH, exist_ok=True)

        # Copy the PDB files to result/pdb
        for pdb_file in pdb_files:
            src_path = os.path.join(tmp_folder, pdb_file)
            dst_path = os.path.join(RESULT_PDB_FOLDER_PATH, pdb_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)


# Streamlit app main function
def main():
    # Clear folders at the start of the app
    peptide_folder_path = PDB_FOLDER_PATH
    #global peptide_folder_path
    clear_folder(RESULT_FOLDER_PATH)
    clear_folder(UPLOAD_FOLDER_PATH)
    clear_folder(RESULT_PDB_FOLDER_PATH)
    clear_folder(USER_PDB_FOLDER_PATH)

    st.title("K-Means Clustering with PyUUL")

    # Sidebar for user inputs
    st.sidebar.header("Input Parameters")
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=2)
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=1, max_value=10, value=2)

    # Main content
    st.write("### Clustering Parameters")
    st.write(f"Number of Clusters: {num_clusters}")
    st.write(f"Number of Iterations: {num_iterations}")

    # File upload for ligand
    st.sidebar.header("Upload Ligand File")
    uploaded_ligand = st.sidebar.file_uploader("Upload a ligand file (PDB format)", type="pdb")

    if uploaded_ligand is not None:
        # Save the uploaded ligand file to the UPLOAD_FOLDER_PATH
        os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)
        lig_file_path = os.path.join(UPLOAD_FOLDER_PATH, "lig.pdb")
        with open(lig_file_path, "wb") as f:
            f.write(uploaded_ligand.getbuffer())
        st.success(f"Ligand file uploaded successfully: {uploaded_ligand.name}")

    # Initialize session state for button
    if 'use_user_pdb' not in st.session_state:
        st.session_state.use_user_pdb = False

    # Button to switch to user's PDB database
    if st.sidebar.button('Use Your PDB Database'):
        st.session_state.use_user_pdb = not st.session_state.use_user_pdb

    if st.session_state.use_user_pdb:
        # File upload for peptide files
        st.sidebar.header("Upload Peptide Files")
        uploaded_peptides = st.sidebar.file_uploader("Upload peptide files (zip format)", type="zip")

        if uploaded_peptides is not None:

            os.makedirs(USER_PDB_FOLDER_PATH, exist_ok=True)

            peptide_file_path = os.path.join(USER_PDB_FOLDER_PATH, uploaded_peptides.name)
            with open(peptide_file_path, "wb") as f:
                f.write(uploaded_peptides.getbuffer())
            st.success(f"peptide files uploaded successfully.")
            peptide_folder_path = USER_PDB_FOLDER_PATH

        else:
            peptide_folder_path = PDB_FOLDER_PATH
            st.warning("No files uploaded or uploaded_peptides is None.")

    # st.write(peptide_folder_path)

    if st.button("Run Clustering"):
        st.write("Running clustering...")
        set_random_seed(100)  # Set random seed for reproducibility
        # st.write(peptide_folder_path)

        pepzips = pathlib.Path(peptide_folder_path)
        for pepzip in pepzips.iterdir():
            azip = zipfile.ZipFile(pepzip)
            azip.extractall(path='tmp')
        kmeans_clustering(UPLOAD_FOLDER_PATH, TMP_FOLDER_PATH, num_clusters, num_iterations)
        st.write("Clustering completed!")

        # Display results
        st.write("### Clustering Results")
        for i in range(num_iterations):
            result_file = os.path.join(RESULT_FOLDER_PATH, f"kmeans-{i + 1}.csv")
            if os.path.exists(result_file):
                result_df = pd.read_csv(result_file)
                st.write(f"Iteration {i + 1} Results:")
                st.dataframe(result_df)

        # Add download button for the result folder
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
