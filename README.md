# ECG-Trust AI
### Adaptive ECG Classification with Explanation Stability and Reliability Assessment

An experimental AI research prototype for ECG heartbeat classification, explainability and reliability assessment.

## Project Overview

ECG-Trust AI combines deep learning and Explainable Artificial Intelligence (XAI) to investigate ECG heartbeat classification and the consistency of model explanations under controlled signal perturbations.

The project uses a 1D Convolutional Neural Network (CNN) and Integrated Gradients to classify and interpret ECG heartbeat predictions.

## Key Features

- Four-class ECG heartbeat classification
- MIT-BIH Arrhythmia Database support
- ECG signal preprocessing and segmentation
- CNN-based heartbeat prediction
- Integrated Gradients visualization
- Signal-quality estimation
- Explanation Stability Score (ESS)
- Experimental ECG Reliability Score (ERS)
- Controlled noise experiments
- Interactive Streamlit dashboard
- Batch ECG heartbeat analysis
- PDF report generation

## Heartbeat Categories

| Class | Description |
|---|---|
| N | Normal |
| S | Supraventricular ectopic |
| V | Ventricular ectopic |
| F | Fusion |

## Technology Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Integrated Gradients

## Project Structure

```text
ECG-Trust-AI/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── datasets/
│   ├── mitbih/
│   ├── physionet2017/
│   └── ptbxl/
│
├── results/
│   ├── trained_models/
│   ├── evaluation_results/
│   └── experiment_outputs/
│
├── notebooks/
│   └── research_experiments.ipynb
│
└── README.md
```

*Folder structure is illustrative and can be adjusted to match the actual repository.*

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ECG-Trust-AI.git

cd ECG-Trust-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and prepare the required ECG dataset and model artifacts according to the project instructions.

## Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Experimental Results

The selected CNN achieved the following final test results:

| Metric | Result |
|---|---:|
| Accuracy | 81.97% |
| Macro F1-score | 0.3560 |
| Weighted F1-score | 0.8317 |

The model showed uneven performance across heartbeat categories, particularly for minority classes.

## Proposed Research Measures

### Explanation Stability Score (ESS)

ESS compares attribution overlap and correlation between explanations generated from clean and perturbed ECG signals.

### ECG Reliability Score (ERS)

The experimental ERS combines:

- Signal Quality Score (SQS)
- Prediction Confidence
- Explanation Stability Score (ESS)

The current ERS formulation is experimental and has not been clinically validated.

## Research Limitations

- Limited recognition of some minority heartbeat classes
- Variation in recording-wise classification performance
- Experimental signal-quality estimation
- Provisional ERS weights requiring further validation
- Explanation stability does not establish medical correctness

## Disclaimer

ECG-Trust AI is an academic research prototype. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions.

## Author

**Apoorva Sahu**

B.Tech — Computer Science and Engineering (Data Science)

## Acknowledgements

MIT-BIH Arrhythmia Database — PhysioNet

