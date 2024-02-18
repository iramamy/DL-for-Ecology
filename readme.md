# Developing a Convolutional Neural Network for Bird Call Identification: A Case Study at Intaka Island

## Overview
This repository contains prediction code and my audio data for my Ecology class at AIMS, South Africa, recorded during our expedition in [Intaka Island](https://intaka.co.za/). 
The project focuses on the development of a Convolutional Neural Network (CNN) for identifying bird calls, contributing to ecological research and wildlife monitoring efforts. 

The whole dataset used to train the model can be accessed upon request to the AIMS, South Africa, AI students. But while waiting for the approval of your request, you can still simulate what we did by using our model and follow the steps below. 

Thanks for all DeepLearning For Ecology lecturers and tutors: Dr.Emmanuel Dufourq, Dr. Lorene Jeantet, Matthew Van den Berg, and Milanto F. Rasolofohery.

## Repository Structure
- **Data/**: This directory stores my unique audio datasets and the annotated files of each recording.
- **Predictions/**: This directory holds all predicted labels for the whole test files.

## Usage
To use the code in this repository, follow these steps:
1. Clone the repository to your local machine.
2. Load the saved model `model3.keras`.
3. Run the main script `predict.py` to make prediction on the audio files.
4. Once done, check all CSV files containing all predicted labels on your working directory.
5. Compare them with each annotated files.

## Installation
Before running the code, make sure you have installed all the required dependencies. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Dependencies
The code in this repository requires the following dependencies:
```
- Python 3.x
- Librosa
- resampy
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

```

## Reference
- [Jeantet, L., Dufourq, E., 2023. Improving deep learning acoustic classifiers with contextual information for wildlife monitoring. Ecological Informatics, 77, 102256.](https://www.sciencedirect.com/science/article/pii/S1574954123002856)

## License
This project is licensed under the MIT License.
