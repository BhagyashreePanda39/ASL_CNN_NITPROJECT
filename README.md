# ASL_CNN_NITPROJECT
CNN Model for American Sign Language Recognition
## Overview
This project involves building a Convolutional Neural Network (CNN) to recognize hand gestures corresponding to the American Sign Language (ASL) alphabet. The model processes images and classifies them into one of the 29 ASL categories, providing a foundation for assistive communication tools.

---

## Features
- **Deep Learning Architecture:** Utilizes CNNs with multiple layers for efficient feature extraction and classification.
- **Data Preprocessing:** Resizes images to 64x64 pixels and normalizes pixel values for faster and accurate training.
- **Model Evaluation:** Employs metrics like confusion matrices and classification reports to evaluate model performance.
- **Visualization:** Includes visualizations of training/validation accuracy, loss curves, and a confusion matrix for detailed performance insights.

---

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - TensorFlow
  - Keras
  - OpenCV
  - Matplotlib
  - Seaborn
  - NumPy
  - Pandas
- **Tools:** Jupyter Notebook, VS Code

---

## Dataset
The dataset consists of labeled images representing ASL gestures. Images are processed for consistency:
- **Image Size:** Resized to 64x64 pixels
- **Normalization:** Pixel values normalized to a range of [0, 1]

---

## How It Works
1. **Data Preprocessing:**
   - Load and preprocess image data.
   - Split the dataset into training and testing sets.

2. **Model Architecture:**
   - **Convolutional Layers:** Extract features from the input images.
   - **Pooling Layers:** Reduce spatial dimensions and computational complexity.
   - **Dropout Layers:** Prevent overfitting by randomly deactivating neurons during training.
   - **Fully Connected Layers:** Classify images into one of the ASL categories.

3. **Training:**
   - Compile the model with `sparse_categorical_crossentropy` loss and the Adam optimizer.
   - Train the model over multiple epochs with callbacks like early stopping and learning rate reduction.

4. **Evaluation:**
   - Test the model on unseen data to measure accuracy.
   - Generate confusion matrices and classification reports to assess performance.

---

## Results
- Achieved high accuracy in classifying ASL gestures.
- Successfully visualized model performance using validation metrics and confusion matrices.

---
## Learning Outcomes
- Enhanced understanding of CNN architecture and image processing techniques.
- Improved skills in data preprocessing and model evaluation.
- Learned to use performance metrics like confusion matrices and classification reports effectively.
- Gained experience in coordinating with mentors and solving real-world problems during an internship program.

---

## Future Scope
- Extend the model for real-time ASL gesture recognition using a webcam.
- Increase dataset diversity for better generalization.
- Optimize the model for deployment in mobile or web applications.

---

## Contributing
Contributions are welcome! Feel free to fork the repository, raise issues, or submit pull requests to improve this project.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact
**Author:** Bhagyashree Panda  
**Email:** [bhagyashreepandaup37@gmail.com](mailto:bhagyashreepandaup37@gmail.com)  
**GitHub:** [https://github.com/BhagyashreePanda39](https://github.com/BhagyashreePanda39)
