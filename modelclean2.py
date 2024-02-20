import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from matplotlib.figure import Figure
from scipy.spatial.distance import cosine

class ImageClassifier:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.model = None
        self.label_encoder = LabelEncoder()

    def create_model(self, output_shape, num_classes):
        self.model = Sequential([
            Dense(128, activation="relu", input_shape=(output_shape,)),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).resize((224, 224))
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        return np.expand_dims(image_array, axis=0)

    def extract_features(self, image_path):
        image_array = self.load_and_preprocess_image(image_path)
        features = self.base_model.predict(image_array)
        return features.flatten()

    def prepare_data(self, input_folder):
        features = []
        labels = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                image_path = os.path.join(input_folder, filename)
                feature = self.extract_features(image_path)
                label = filename.split("_")[1] + "_" + filename.split("_")[2]
                features.append(feature)
                labels.append(label)
        labels = self.label_encoder.fit_transform(labels)
        return np.array(features), np.array(labels)
    
    def train(self, features, labels, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
        self.create_model(features.shape[1], len(np.unique(labels)))
        history = self.model.fit(X_train, y_train, epochs=10, validation_split=0.2)
        return X_test, y_test, history

    def evaluate_and_plot_confusion_matrix(self, features, labels, save_path=None):
        y_pred = self.model.predict(features)
        y_pred_classes = np.argmax(y_pred, axis=1)
        confusion_mtx = confusion_matrix(labels, y_pred_classes)
        confusion_mtx_normalized = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]
        class_names = self.label_encoder.classes_

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(confusion_mtx_normalized, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        # plt.xlabel('Predicted labels')
        # plt.ylabel('True labels')
        # plt.title('Normalized Confusion Matrix')
        # if save_path:
        #     plt.savefig(save_path)
        # plt.show()

        confusion_df = pd.DataFrame(confusion_mtx_normalized, index=class_names, columns=class_names)
        print(confusion_df)
        if save_path:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_mtx_normalized, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Normalized Confusion Matrix')
        plt.savefig(save_path)
        return confusion_df

    def plot_results(self, history, save_path):
        # Plot training and validation loss
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'loss_plot.png'))
        plt.close()

        # Plot training and validation accuracy 
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
        plt.close()

    def save_model(self, model, model_path, label_encoder_path):
        model.save(model_path)
        np.save(label_encoder_path, self.label_encoder.classes_)

    def load_model(self, model_path, label_encoder_path):
        
        self.model = load_model(model_path)

        self.label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
    
    # def compute_similarity(self, model, image_path, reference_features):
    #     test_image = self.load_and_preprocess_image(image_path)
    #     test_features = model.predict(test_image)

    #     similarity_scores = []
    #     for ref_features in reference_features:
    #         ref_image = self.load_and_preprocess_image(ref_features)
    #         sim_score = ssim(test_image, ref_image)
    #         similarity_scores.append(sim_score)

    #     return similarity_scores
    def load_and_preprocess_image1(self, image_path):
        img = Image.open(image_path).resize((224, 224))
        img = img.convert('RGB')  
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = preprocess_input(img_array) 
        return img_array

    def predict_image_class(self, image_path):
        processed_image = self.load_and_preprocess_image1(image_path)
        features = self.base_model.predict(processed_image)
        flattened_features = features.reshape(1, -1) 
        prediction = self.model.predict(flattened_features)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_class = self.label_encoder.inverse_transform(predicted_class_index)
        return predicted_class
    
    from scipy.spatial.distance import cosine

    def compute_similarity(self, image_path, reference_features):
        test_image = self.load_and_preprocess_image1(image_path)
        test_features = self.base_model.predict(test_image).flatten()

        similarity_scores = []
        for ref_features in reference_features:
            sim_score = 1 - cosine(test_features, ref_features)  
            similarity_scores.append(sim_score)

        return similarity_scores
