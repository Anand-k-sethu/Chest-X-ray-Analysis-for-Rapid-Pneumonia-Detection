# Pneumonia Detection from Chest X-rays

## Project Overview

This project uses **machine learning** and **deep learning** to detect pneumonia in chest X-ray images. The goal is to create a model that can distinguish between **normal** and **pneumonia**-infected lungs, potentially aiding healthcare professionals in diagnosing pneumonia more quickly and accurately. The model utilizes the **VGG16** architecture, a popular deep convolutional neural network (CNN), to classify the images. This approach can help in reducing the burden on healthcare systems by providing rapid, automated analysis.

### Key Features:
- **Accurate Classification**: Classifies X-ray images into **Normal** and **Pneumonia** categories.
- **Efficient and Scalable**: Leverages a pre-trained **VGG16** model for optimal performance with minimal resource use.
- **Automated Diagnosis**: Helps automate the diagnostic process, potentially increasing the efficiency and accuracy of diagnoses in clinical settings.
  
### Performance Metrics:
- **Accuracy**: 96.0%
- **Precision**: 94.23%
- **Recall**: 98.0%
- **F1-Score**: 96.08%

These results demonstrate the model’s strong ability to correctly identify pneumonia cases from chest X-rays, making it a valuable tool for healthcare professionals.

---

## Problem Statement & Relevance

Pneumonia is a common yet severe lung infection that demands prompt diagnosis and treatment. Traditional diagnostic methods may involve long waiting times and require expert knowledge. This project automates the process of detecting pneumonia from chest X-ray images, significantly improving the speed and accuracy of diagnoses. Early detection can drastically improve treatment outcomes, particularly in underserved areas with limited access to healthcare.

---

## Project Breakdown

### 1. **Data Preprocessing**
The data consists of **Pneumonia** and **Normal** chest X-ray images, and we use the following preprocessing steps:
- **Resizing**: All images are resized to 224x224 pixels to match the input size expected by the VGG16 model.
- **Normalization**: Pixel values are scaled to the range `[0, 1]` for efficient model training.
- **Undersampling**: The dataset is balanced by undersampling both classes (Normal and Pneumonia) to 1000 images each, ensuring fairness during training.

### 2. **Dataset Splitting**
The dataset is split into three parts:
- **Training Set (80%)**: Used to train the model.
- **Validation Set (10%)**: Used to tune hyperparameters and evaluate the model during training.
- **Test Set (10%)**: Used to evaluate the model's final performance.

### 3. **Model Architecture**
The model utilizes the **VGG16** architecture, which is a convolutional neural network known for its ability to learn rich feature representations in image data. The model is fine-tuned to classify chest X-ray images into two categories: **Normal** and **Pneumonia**.

### 4. **Model Evaluation**
After training the model, we evaluate its performance on the test set using key metrics:
- **Accuracy**: How often the model is correct.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.

### 5. **Training & Results**
The model was trained for 10 epochs, and the final results were:
- **Training Accuracy**: 99.33%
- **Validation Accuracy**: 98.0%
- **Final Test Accuracy**: 96.0%

A confusion matrix was generated to visualize performance:
```
Confusion Matrix:
[[94  6]
 [ 2 98]]
```

---

## Steps to Run the Code

### 1. **Install Required Libraries**
Make sure the following Python libraries are installed:
```bash
pip install tensorflow keras opencv-python matplotlib pandas
```

### 2. **Data Loading**
The dataset consists of chest X-ray images in the following structure:
- `/train/PNEUMONIA`, `/train/NORMAL`
- `/test/PNEUMONIA`, `/test/NORMAL`
- `/val/PNEUMONIA`, `/val/NORMAL`

### 3. **Undersampling the Dataset**
The dataset is undersampled to ensure a balanced number of **Normal** and **Pneumonia** images (1000 images per class).

```python
desired_num_images = 1000
pneumonia_undersampled = pneumonia[:desired_num_images]
normal_undersampled = normal[:desired_num_images]
```

### 4. **Data Preprocessing**
The images are resized to 224x224 pixels, normalized, and converted into the appropriate format for input into the VGG16 model:
```python
def preprocess_image_VGG16(image_list, new_size=(224, 224)):
    X = []  # images
    y = []  # labels

    for image in image_list:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, new_size)
        img_normalize = img_resized.astype(np.float32)/255.0
        X.append(img_normalize)

        if 'NORMAL' in image:
            y.append(0)  # Normal class
        elif 'PNEUMONIA' in image:
            y.append(1)  # Pneumonia class

    return X, y
```

### 5. **Model Training**
The VGG16 model is fine-tuned to classify chest X-ray images. It is trained for 10 epochs with a learning rate adjustment:
```bash
Epoch 10/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 504s 10s/step - accuracy: 0.9933 - loss: 0.0159 - val_accuracy: 0.9800 - val_loss: 0.1408
```

### 6. **Model Evaluation**
After training, the model is evaluated on the test dataset, with the following results:
- **Accuracy**: 96.0%
- **Precision**: 94.23%
- **Recall**: 98.0%
- **F1-Score**: 96.08%

---

## Future Improvements
- **Data Augmentation**: Apply techniques like rotation, flipping, and zooming to increase the variety of the training set and improve generalization.
- **Advanced Architectures**: Experiment with other deep learning models like ResNet, DenseNet, or EfficientNet for potential improvements in performance.
- **Transfer Learning**: Fine-tuning more advanced pre-trained models could further enhance results.

---

## Conclusion

This project successfully demonstrates the power of deep learning, particularly convolutional neural networks (CNNs), in automating medical image classification. By training a VGG16 model on chest X-ray images, we achieved high accuracy in detecting pneumonia. This approach can significantly assist healthcare professionals in making quicker, more accurate diagnoses, especially in resource-limited settings.

---

## Directory Structure
```
/chest-xray-pneumonia
    /train
        /PNEUMONIA
        /NORMAL
    /test
        /PNEUMONIA
        /NORMAL
    /val
        /PNEUMONIA
        /NORMAL
```

## Thanks for reading
