\# 🍅 Tomato Disease Detection Using Deep Learning



A deep learning-based web application that detects tomato leaf diseases from images using Convolutional Neural Networks (CNNs).



\## 📌 Project Overview



This project uses the PlantVillage dataset to classify tomato leaves into five classes:



1\. Tomato Healthy

2\. Tomato Early Blight

3\. Tomato Late Blight

4\. Tomato Leaf Mold

5\. Tomato Septoria Leaf Spot



The system uses two models:



\- \*\*Tomato/Non-Tomato Gate\*\* – checks whether the uploaded image is a tomato leaf.

\- \*\*Tomato Disease Classifier\*\* – predicts the disease when the image is identified as a tomato leaf.



The application is built using TensorFlow, Keras and Streamlit.



\## 🧠 System Architecture



```text

Leaf Image

&#x20;   ↓

Tomato / Non-Tomato Gate

&#x20;   ↓

&#x20;┌───────────────┐

&#x20;│               │

Non-Tomato      Tomato

&#x20;│               │

&#x20;↓               ↓

Unsupported   Disease Classifier

Image             ↓

&#x20;         ┌───────┼────────┐

&#x20;         ↓       ↓        ↓

&#x20;      Healthy  Disease  Disease

\## 📊 Model Performance



\### Tomato Disease Classifier



Test dataset: 723 images



\- Accuracy: \*\*97.51%\*\*

\- Precision: \*\*97.50%\*\*

\- Recall: \*\*97.51%\*\*

\- F1 Score: \*\*97.50%\*\*



\### Tomato/Non-Tomato Gate



Test dataset: 1,446 images



\- Accuracy: \*\*97.44%\*\*

\- Precision: \*\*99.42%\*\*

\- Recall: \*\*95.44%\*\*

\- F1 Score: \*\*97.39%\*\*



\## 🛠️ Technologies Used



\- Python

\- TensorFlow

\- Keras

\- CNN

\- NumPy

\- Pandas

\- Scikit-learn

\- Pillow

\- Streamlit

\- Matplotlib

\- Git

\- GitHub



\## 📂 Project Structure



```text

s2/

│

├── app/

│   └── app.py

│

├── data/

│   └── tomato/

│

├── models/

│   ├── tomato\_disease\_cnn.keras

│   └── tomato\_gate.keras

│

├── outputs/

│   ├── tomato/

│   └── tomato\_gate/

│

├── src/

│   ├── prepare\_tomato\_dataset.py

│   ├── train\_tomato.py

│   ├── evaluate\_tomato.py

│   ├── prepare\_tomato\_gate.py

│   ├── train\_tomato\_gate.py

│   └── evaluate\_tomato\_gate.py

│

├── requirements.txt

├── .gitignore

└── README.md

