from inference_gui2 import MODELS_DIR
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QCheckBox, QTableWidget,
    QApplication)
import huggingface_hub
import os
import glob
import shutil
import sys
from pathlib import Path

# Only enable this if you plan on training off a downloaded model.
DOWNLOAD_DISCRIMINATORS = False
MODELS_DIR = os.path.join("so-vits-svc",MODELS_DIR)

class DownloadStrategy:
    def __init__(self, repo_id : str, model_dir : str):
        """ Pull from HF to find available models """
        pass

    def get_available_model_names(self) -> list:
        """ Returns a list of model names """
        pass

    def check_present_model_name(self, name : str) -> bool:
        """ Returns True if model is already installed """
        return False

    def download_model(self, name : str):
        """ Downloads model corresponding to name """
        pass

class FolderStrategy(DownloadStrategy):
    def __init__(self, repo_id, model_dir):
        self.repo_id = repo_id
        self.model_dir = model_dir
        self.model_repo = huggingface_hub.Repository(
            local_dir=self.model_dir, clone_from=self.repo_id,
            skip_lfs_files=True)
        self.model_folders = os.listdir(model_dir)
        self.model_folders.remove('.git')
        self.model_folders.remove('.gitattributes')

    def get_available_model_names(self):
        return self.model_folders

    def check_present_model_name(self, name):
        return bool(name in os.listdir(MODELS_DIR))

    def download_model(self, model_name):
        print("Downloading "+model_name)
        basepath = os.path.join(self.model_dir, model_name)
        targetpath = os.path.join(MODELS_DIR, model_name)
        gen_pt = next(x for x in os.listdir(basepath) if x.startswith("G_"))
        disc_pt = next(x for x in os.listdir(basepath) if x.startswith("D_"))
        cfg = next(x for x in os.listdir(basepath) if x.endswith("json"))
        try:
            clust = next(x for x in os.listdir(basepath) if x.endswith("pt"))
        except StopIteration as e:
            clust = None

        huggingface_hub.hf_hub_download(repo_id = self.repo_id,
            filename = model_name + "/" + gen_pt, local_dir = targetpath)

        if DOWNLOAD_DISCRIMINATORS:
            huggingface_hub.hf_hub_download(repo_id = self.repo_id,
                filename = model_name + "/" + disc_pt, local_dir = targetpath)
        if clust is not None:
            huggingface_hub.hf_hub_download(repo_id = self.repo_id,
                filename = model_name + "/" + clust, local_dir = targetpath)
        shutil.copy(os.path.join(basepath, cfg), os.path.join(targetpath, cfg))

from zipfile import ZipFile

class ZipStrategy(DownloadStrategy):
    def __init__(self, repo_id, model_dir):
        self.repo_id = repo_id
        self.model_dir = model_dir
        self.model_repo = huggingface_hub.Repository(
            local_dir=self.model_dir, clone_from=self.repo_id,
            skip_lfs_files=True)
        self.model_zips = glob.glob(model_dir + "/**/*.zip", recursive=True)
        self.model_zips.remove('.git')
        for fi in self.model_zips:
            if not fi.endswith('.zip'):
                self.model_zips.remove(fi)

        self.model_names = [Path(x).stem for x in self.model_zips]
        self.rel_paths = {Path(x).stem :
            Path(x).relative_to(model_dir) for x in self.model_zips}

    def get_available_model_names(self):
        return self.model_names

    def check_present_model_name(self, name):
        return bool(name in os.listdir(MODELS_DIR))

    def download_model(self, model_name):
        huggingface_hub.hf_hub_download(repo_id = self.repo_id,
            filename = self.rel_paths[model_name], local_dir = MODELS_DIR)
        zip_path = os.path.join(MODELS_DIR,model_name+'.zip')
        with ZipFile(zip_path, 'r') as zipObj:
            zipObj.extractall(MODELS_DIR)
        os.remove(zip_path)

class DownloaderGui (QMainWindow):
    def __init__(self):
        print("Downloading repos...")
        self.strategies = [
            FolderStrategy("therealvul/so-vits-svc-4.0",
                "repositories/hf_vul_model"),
            ZipStrategy("Amo/so-vits-svc-4.0_GA",
                "repositories/hf_amo_models"),
            ZipStrategy("HazySkies/SV3",
                "repositories/hf_hazy_models")]
        print("Finished downloading repos")

        self.setWindowTitle("so-vits-svc 4.0 Downloader")
        self.central_widget = QFrame()
        self.layout = QVBoxLayout(self.central_widget)

        self.model_table = QtWidgets.QTableWidget()
        self.layout.addWidget(self.model_Table)
        self.model_table.setColumnCount(3)
        self.model_table.setHorizontalHeaderLabels(
            ['Model name', 'Check to install', 'Detected on system?'])

        self.available_models = {}
        self.present_map = {}
        self.checkbox_map = {}
        for i,v in enumerate(self.strategies):
            available_models = v.get_available_model_names()
            for m in available_models:
                self.available_models[m] = i
                self.present_map[m] = v.check_present_model_name(m)
                self.checkbox_map[m] = QCheckBox()

        # Populate table
        self.model_table.setRowCount(len(self.available_models.items()))
        for i,k,v in enumerate(self.available_models.items()):
            # Model name
            self.model_table.setItem(i,0,str(k))
            # Check to install
            self.model_table.setCellWidget(i,1,self.checkbox_map[k])
            # Detected on system?
            self.model_table.setItem(i,2,str(self.present_map[k]))

        self.download_button = QPushButton("Download selected")
        self.layout.addWidget(self.download_button)
        self.download_button.clicked.connect(self.download_selected)

    def download_selected(self):
        for i,k,v in enumerate(self.available_models.items()):
            if self.checkbox_map[k].isChecked():
                self.strategies[self.available_models[k]].download_model(
                    self.available_models[k])

if __name__ == '__main__':
    if Path(os.getcwd()).stem == 'so-vits-svc':
        os.chdir('..')
    app = QApplication(sys.argv)
    w = DownloaderGui()
    w.show()
    app.exec()
