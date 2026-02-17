Animal Species Detector 🐾
Overview

This project implements an Animal Species Classification system using Transfer Learning with VGG16.
The model is trained on 8 animal categories and achieves 86.87% test accuracy.

Classes:
  1. Butterfly
  2. Cat
  3. Cow
  4. Dog
  5. Elephant
  6. Horse
  7. Sheep
  8. Squirrel

 Model Architecture:
  Pretrained VGG16 (ImageNet weights)
  Fine-tuned last 4 convolutional layers
  Global Average Pooling
  Dropout (0.5)
  Dense Softmax Output Layer
  Optimizer: Adam
  Loss Function: Categorical Crossentropy

📊 Performance
  Test Accuracy: 86.87%
  Macro F1 Score: 0.87
  Confusion Matrix included
  Classification Report included

  
📁 Dataset Structure
  dataset/
    train/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/


Total:
  640 Training Images
  160 Testing Images
  8 Classes


🚀 How to Run
  1. Clone the repository
    git clone https://github.com/raghavrajoria/Animal-Species-Detector.git
    cd animal-species-detector

  2.Install dependencies
    pip install -r requirements.txt

  3.Add dataset in dataset/
  4.Run training
    python src/train.py


📌 Future Improvements
  Replace VGG16 with EfficientNet
  Add Grad-CAM visualizations
  Deploy using Streamlit
  Expand dataset with more species
  Convert to TensorFlow Lite for edge deployment


📷 Sample Output
  Accuracy Graph
  Confusion Matrix
  Classification Report


🛠 Tech Stack
  Python
  TensorFlow / Keras
  NumPy
  Matplotlib
  Seaborn
  Scikit-learn


👤 Author
  Raghav Rajoria
  Machine Learning & Backend Develop


  
