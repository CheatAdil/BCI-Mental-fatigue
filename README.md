# Mental Fatigue Detection from EEG
CSCI 490 — Brain–Computer Interface course project

## Project Summary
This project demonstrates detection of mental fatigue from EEG signals using the SEED-VIG dataset. We implemented preprocessing, feature extraction, a PyTorch-based ML model, and a real-time simulation pipeline that outputs “fatigue detected” alerts.  

## Dataset
- Dataset: **SEED-VIG** (vigilance EEG dataset).  
- Access requires signed license agreement from the BCMI lab at SJTU.  
- Preprocessing performed:
  - Artifact removal (ICA and filtering).  
  - Bandpass filtering (~1–50 Hz).  
  - Normalization (z-score scaling).  
  - FFT-based frequency feature extraction.  
  - Converted to NumPy arrays for model training.  

## Methods
### Model
- Implemented in **PyTorch**.  
- Architecture (fast single-sample inference):
  - Conv1D → BatchNorm → ReLU  
  - Conv1D → BatchNorm → ReLU → Dropout  
  - Flatten → Linear → Softmax output  
- Loss: CrossEntropyLoss.  
- Training/validation split with hyperparameter search.  

### Real-Time Pipeline
- **EEG Simulator**: Streams dataset samples sequentially to mimic real-time acquisition.  
- **Main Detection Script**: Consumes simulator output, runs inference through the trained model, and displays an alert when fatigue is detected.  

## Results
- Model successfully trained on SEED-VIG with validation and hyperparameter tuning.  
- Real-time simulation confirmed that fatigue states could be detected online.  
- (Insert final accuracy / F1 / relevant metric here if you have it.)  

## Contributions
- Data acquisition and preprocessing.  
- Model implementation and training.  
- EEG simulator development.  
- Real-time fatigue detection script.  
- Documentation, graphs, and presentation.  

## Notes
- Raw dataset is not included in this repository.  
- Processed data loaders and scripts are provided.  
