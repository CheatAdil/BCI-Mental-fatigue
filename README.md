* Brain Computer Interface (CSCI 490) Course Project -- Mental Fatigue Project 

BCI-MENTAL-FATIGUE/
├── data/                     # raw & processed data (not under version control)
│   ├── raw/
│   └── processed/
│
├── ml/                       # Python package for your code
│   ├── data/                 # data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py        # EEGDataset class
│   │   └── datamodule.py     # optional LightningDataModule or similar
│   │
│   ├── models/               # model definitions
│   │   ├── __init__.py
│   │   └── eeg_net.py        # EEGNet class
│   │
│   ├── engines/              # training / evaluation logic
│   │   ├── __init__.py
│   │   └── trainer.py        # Trainer class or functions
│   │
│   ├── utils/                # helper functions (metrics, plotting)
│   │   ├── __init__.py
│   │   └── utils.py
│   │
│   └── config/               # configuration files (YAML, JSON)
│       └── default.yaml
│
├── scripts/                  # entry‐point scripts
│   ├── train.py              # calls ml.engines.trainer
│   └── evaluate.py           # runs inference & computes metrics
│
├── tests/                    # unit & integration tests
│   └── test_dataset.py
│
├── requirements.txt
├── setup.py   or   pyproject.toml
└── README.md
