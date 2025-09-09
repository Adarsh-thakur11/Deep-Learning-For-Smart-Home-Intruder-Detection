README.md
#  Smart Home Intruder Detection (Multimodal Deep Learning)

This repository implements a **multimodal deep learning pipeline** for **intruder detection in smart homes** using **audio, video, and ambient sensor data**.  
The project combines **YOLOv8 (video)**, **YAMNet + MLP (audio)**, and **XGBoost (sensors)**, then fuses them to detect potential intruder activities.

---

## Repository Structure

Smart_Home_Intruder_Detection/
│── notebooks/           # Jupyter notebooks for Audio, Video, Sensor, Fusion
│   ├── Audio.ipynb
│   ├── Video.ipynb
│   ├── Sensor.ipynb
│   └── 3d_vs_2D_MultiModal_intruder_detection.ipynb
│
│── outputs/             # Processed datasets, YOLO formats, results
│── data/                # Place raw datasets here
│── runs/                # YOLO training runs
│── src/                 # Utility scripts (paths, utils)
│── openv6main.yaml      # YOLO dataset config file (Open Images V6 subset)
│── README.md            # Project documentation
│── requirements.txt     # Python dependencies


---

## Datasets Used

### Audio
- **ESC-50 dataset** (environmental sounds)  
  [ESC-50 GitHub](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file)  
- **VOIC dataset (gunshot, glassbreak)**  
  [Zenodo VOIC dataset](https://zenodo.org/records/3514950)

  download "VOICe_clean.7z" file with size 3.5 GB.

### Video
- **DCSASS dataset** (home burglary/stealing/robbery scenarios)  
  [Kaggle DCSASS dataset](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset/data)

### Sensor
- **Ambient Sensor-based Human Activity Recognition**  
   [Kaggle Human Activity Recognition](https://www.kaggle.com/datasets/utkarshx27/ambient-sensor-based-human-activity-recognition)

### Object Detection (Weapons/Intruder items)
- **Open Images V6 subset (OIDv6)**  
  [Open Images V6](https://storage.googleapis.com/openimages/web/index.html)  

To download via **OIDv6 Toolkit (command line):**

git clone https://github.com/EscVM/OIDv6_Classifier.git
cd OIDv6_Classifier
pip install -r requirements.txt

# Example: download Person, Weapon, Handgun...
python3 main.py downloader --classes Person Weapon Handgun Shotgun Helmet Scarf Knife "Kitchen knife" "Baseball bat" Backpack Flashlight --type_csv train --limit 200


 How to Prepare Data
	1.	Place all raw datasets in: Smart_Home_Intruder_Detection/data/
   
   Example:
    data/
     ├── ESC-50/
     ├── VOIC/
     ├── DCSASS_raw/
     └── human-activity-recognition-sensor/


    2.	Run preprocessing notebooks (Audio.ipynb, Sensor.ipynb, Video.ipynb).
	    •	Outputs (converted, cleaned, augmented datasets) are saved to:
             Smart_Home_Intruder_Detection/outputs/
    3.	Final YOLO-ready video dataset is structured as:
        outputs/openv6_yolo_format/
                    ├── images/train
                    ├── images/val
                    ├── labels/train
                    └── labels/val
    4.  If you want to skip preprocessing, you can download my preprocessed outputs/ folder directly from Google Drive (link= "https://drive.google.com/drive/folders/1x9uXNgBAzUa66ey9ep8Cbsdxlfg8gmxR?usp=sharing ").
	
Training YOLOv8 (Video)
    yolo detect train data=openv6main.yaml model=yolov8n.pt epochs=20 imgsz=640 batch=16 name=openv6_intruder_model
        •	openv6main.yaml points to outputs/openv6_yolo_format/.
        •	Training runs are stored under runs/detect/.


Training Audio Model

    Open Audio.ipynb and run all cells:
	    •	Extracts ESC-50 + VOIC audio
	    •	Converts using ffmpeg
	    •	Augments (time stretch, pitch shift, noise)
	    •	Trains YAMNet + MLP classifier



Training Sensor Model

    Open Sensor.ipynb:
	    •	Reads ambient smart home rawdata
	    •	Windows into 30s slices
	    •	Labels intrusion (ON > 3 threshold)
	    •	Trains XGBoost classifier
	    •	Generates probabilities for fusion



Fusion Model
	    •	Notebook: 3d_vs_2D_MultiModal_intruder_detection.ipynb
    	•	Combines predictions from audio, video, and sensor models
    	•	Produces final intruder detection decision



Visualizations
    	•	Class distributions for audio, video, sensors
    	•	Confusion matrices for each modality
	    •	Accuracy under noise / label corruption tests
	    •	Training metrics (YOLO mAP curves, Audio accuracy, Sensor F1)



🛠️ Requirements

Install dependencies:
    pip install -r requirements.txt
       Main packages:
	•	ultralytics (YOLOv8)
	•	tensorflow, tensorflow_hub
	•	scikit-learn, xgboost
	•	librosa, pydub
	•	matplotlib, seaborn
	•	pandas, numpy


 Notes
	•	Use openv6main.yaml (not openv6.yaml) to avoid conflicts.
	•	Datasets are large; only configs + scripts are stored in GitHub. Place raw datasets manually in data/ or download preprocessed outputs/ from Google Drive.
	•	Pretrained YOLO weights (yolov8n.pt) are downloaded automatically on first run.

Citation

If you use this repo, please cite the datasets:
	•	Piczak, K.J. (2015). ESC-50: Dataset for Environmental Sound Classification.
	•	Hervás, M. (2019). DCSASS Dataset. Kaggle.
	•	Utkarshx27 (2020). Ambient Sensor HAR Dataset. Kaggle.
	•	Kuznetsova, A. et al. (2020). Open Images V6. IJCV.


