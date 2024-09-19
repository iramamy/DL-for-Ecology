# 🌿 Developing a Convolutional Neural Network for Bird Call Identification: A Case Study at Intaka Island 🐦🎶

## 🌟 Overview
This repository contains prediction code and audio data for my **Ecology class** at **AIMS, South Africa**, recorded during our expedition at [**Intaka Island**](https://intaka.co.za/).  
The project focuses on the development of a **Convolutional Neural Network (CNN)** for identifying bird calls, contributing to **ecological research** and **wildlife monitoring** efforts. 🌍🌳

The complete dataset used to train the model is available upon request to the **AIMS, South Africa, AI students**. While waiting for approval, you can still simulate our work by using our model and following the steps below. 

**Special Thanks** to all DeepLearning for Ecology lecturers and tutors:  
- 🎓 **Dr. Emmanuel Dufourq**  
- 🎓 **Dr. Lorene Jeantet**  
- 🧑‍🏫 **Matthew Van den Berg**  
- 🧑‍🏫 **Milanto F. Rasolofohery**

---

## 📂 Repository Structure
- **Data/**: Contains the unique **audio datasets** and **annotated files** for each recording.
- **Predictions/**: Holds the **predicted labels** for the entire test files.

---

## 🛠️ Usage
Follow these steps to use the code:

1. **Clone the repository** to your local machine.
2. **Load the saved model** `model3.keras`.
3. **Run the script** `predict.py` to make predictions on the audio files.
4. **Check the CSV files** containing predicted labels in your working directory.
5. **Compare the predictions** with the annotated files.

---

## 🔧 Installation
Before running the code, ensure you have installed all the required dependencies. Install them using pip:

```bash
pip install -r requirements.txt
```

## 📦 Dependencies
```
This project requires the following:
- Python 3.x
- Librosa
- resampy
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn
```

## 📖 Reference
- [Jeantet, L., Dufourq, E., 2023. Improving deep learning acoustic classifiers with contextual information for wildlife monitoring. Ecological Informatics, 77, 102256.](https://www.sciencedirect.com/science/article/pii/S1574954123002856)

## License
This project is licensed under the MIT License.
