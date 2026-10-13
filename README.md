# M3lem — Digital Twin for Traditional Moroccan Embroidery Learning

The **M3lem project** is a **digital twin designed to teach traditional Moroccan embroidery**. Here is how we developed it.

To begin, we used **OpenMV and MediaPipe** to capture artisans’ hand movements. Each gesture was converted into a **sequence of frames**, with each frame representing the positions of the hand landmarks, particularly the wrist and fingers. We then **normalized** the data to standardize the wrist position and hand size, ensuring that the model would not be affected by differences in hand size or orientation.

To improve data quality and prevent overfitting, we applied **data augmentation** by generating slight variations of the sequences. We also ensured that the dataset had **balanced classes**. The classes were encoded as **one-hot vectors** for the model, while class labels were converted into integers using `LabelEncoder`. We then **split the dataset into training, validation, and test sets**, using stratified sampling to preserve the proportion of each class in each subset. To address class imbalance, we used **SMOTE** to synthetically oversample underrepresented classes and a **Random Undersampler** to reduce the size of overrepresented classes.

## Deep Learning Model Architecture

For the model, we implemented the following **sequential deep learning architecture**:

- **`Conv1D(64)`** with ReLU activation and L2 regularization, followed by **Batch Normalization** and **Max Pooling** to extract local patterns and reduce the temporal dimensionality of the sequences.
- **`Conv1D(128)`** using the same techniques to capture more complex patterns.
- **`LSTM(128, return_sequences=True)`** to capture long-term temporal dependencies within the sequences.
- **`LSTM(64)`** to summarize the entire sequence into a more compact feature vector.
- A **`Dense(64, activation='relu')`** layer followed by **Batch Normalization**, and finally a **`Dense(num_classes, activation='softmax')`** output layer to generate probabilities for each type of embroidery stitch.

## Model Evaluation and Robustness Testing

After training, we evaluated the model on the test set and verified that all classes could be predicted (**unique predictions = 10/10**). We also tested its robustness by introducing **noise** and shuffling the order of frames, and compared its performance against baseline models such as **Random Forest** and **Logistic Regression**.

## 3D Integration and Digital Twin

Finally, we integrated the system into **Unreal Engine**, using a **MetaHuman character created in Blender**. The digital character reproduces the gestures recognized by the AI model, providing an immersive learning experience that allows users to learn traditional embroidery digitally.

## Technologies Used

- **Gesture Capture:** OpenMV, MediaPipe
- **Data Processing:** Data Normalization, Data Augmentation, Label Encoding
- **Class Imbalance Handling:** SMOTE, Random Undersampling
- **Deep Learning:** CNN (Conv1D), LSTM, TensorFlow/Keras
- **Baseline Models:** Random Forest, Logistic Regression
- **3D Development:** Unreal Engine, MetaHuman, Blender

## Project Objective

Overall, this project combines **motion capture, deep learning for gesture recognition, and interactive 3D visualization**. It aims to train new artisans while preserving Morocco’s traditional embroidery craftsmanship.
