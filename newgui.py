from modelclean2 import ImageClassifier
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from ttkbootstrap import Style
class ImageClassifierGUI:
    def __init__(self, master):
        self.master = master
        self.style = Style(theme='darkly')
        self.master.title("Image Classifier GUI")
        self.classifier = None
        # GUI Components
        self.setup_gui()

    def setup_gui(self):
        self.master.configure(bg='#333') 
        self.master.geometry('800x600')
        # Adjust grid configuration
        for row_index in range(8):  
            self.master.grid_rowconfigure(row_index, weight=1)
        for col_index in range(3): 
            self.master.grid_columnconfigure(col_index, weight=1)
        # Font and color definitions
        button_style = {'font': ('Roboto', 18, 'bold'), 'background': '#007bff', 'foreground': '#ffffff', 'borderwidth': 1, 'relief': 'flat'}
        
        # Input folder selection
        tk.Label(self.master, text="Input Folder:").grid(row=0, column=0, sticky="w")
        self.input_folder_var = tk.StringVar()
        tk.Entry(self.master, textvariable=self.input_folder_var).grid(row=0, column=1, sticky="we")
        tk.Button(self.master, text="Browse", command=self.browse_folder,font = ('Roboto', 12, 'bold'), background = '#007bff', foreground = '#ffffff', borderwidth = 1, relief = 'flat').grid(row=0, column=2)

        # Load and Preprocess Button
        tk.Button(self.master, text="Load & Preprocess Images", command=self.load_and_preprocess,**button_style).grid(row=1, columnspan=3, sticky="we")

        # Train Model Button
        tk.Button(self.master, text="Train Model", command=self.train_model,**button_style).grid(row=2, columnspan=3, sticky="we")

        # Evaluate Model Button
        tk.Button(self.master, text="Evaluate Model", command=self.evaluate_model,**button_style).grid(row=3, columnspan=3, sticky="we")

        # Save Model Button
        tk.Button(self.master, text="Save Model", command=self.save_model,**button_style).grid(row=4, columnspan=3, sticky="we")

        # Load Model Button
        tk.Button(self.master, text="Load Model", command=self.load_model,**button_style).grid(row=5, columnspan=3, sticky="we")

        # Test Model Button
        tk.Button(self.master, text="Test Model", command=self.test_model,**button_style).grid(row=6, columnspan=3, sticky="we")

        # Similarity Button
        tk.Button(self.master, text="Compute Similarity", command=self.compute_and_display_similarity,**button_style).grid(row=7, columnspan=3, sticky="we")

        # Status Label
        self.status_label = tk.Label(self.master, text="Status: Ready",font=('Roboto', 12, 'bold'), bg='#333', fg='#fff')
        self.status_label.grid(row=8, columnspan=3, sticky="w")

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        self.input_folder_var.set(folder_path)

    def load_and_preprocess(self):
        input_folder = self.input_folder_var.get() 
        if not input_folder:
            messagebox.showwarning("Warning", "Please select an input folder.")
            return
        self.classifier = ImageClassifier(input_folder)
        self.features, self.labels = self.classifier.prepare_data(input_folder)
        self.status_label.config(text="Status: Images loaded and preprocessed.")

    def train_model(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "Please load and preprocess images first.")
            return

        X_test, y_test, self.history = self.classifier.train(self.features, self.labels)

        self.classifier.plot_results(self.history, self.input_folder_var.get())
        self.X_test, self.y_test = X_test, y_test
        
        self.status_label.config(text="Status: Model trained and results plotted.")


    def evaluate_model(self):
        if self.classifier is None or not hasattr(self.classifier, 'model') or self.classifier.model is None:
            messagebox.showwarning("Warning", "Please load and preprocess images and then train the model first.")
            return
        evaluation_results = self.classifier.evaluate_and_plot_confusion_matrix(self.features, self.labels, self.input_folder_var.get())
        table_frame = tk.LabelFrame(self.master, text="Confusion Matrix")
        table_frame.grid(row=9, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        confusion_matrix_text = tk.Text(table_frame, height=10, width=80)
        confusion_matrix_text.grid(row=0, column=0, columnspan=len(evaluation_results), padx=10, pady=10, sticky="w")
        confusion_matrix_text.insert("1.0", evaluation_results.to_string())
        self.status_label.config(text=f"Status: Model evaluated.")
    def save_model(self):
        if self.classifier is None or not hasattr(self.classifier, 'model') or self.classifier.model is None:
            messagebox.showwarning("Warning", "No model to save. Please train the model first.")
            return
        model_save_path = os.path.join(self.input_folder_var.get(), "model.h5")
        label_encoder_path = os.path.join(self.input_folder_var.get(), "label_encoder.npy")

        # Save the model
        self.classifier.model.save(model_save_path)
        np.save(label_encoder_path, self.classifier.label_encoder.classes_)
        self.status_label.config(text="Status: Model and label encoder saved successfully.")

    
    def load_model(self):
        # Prompt the user to select the model file
        model_load_path = filedialog.askopenfilename(title="Select Model File", filetypes=(("HDF5 Model", "*.h5"), ("All Files", "*.*")))
        if not model_load_path:
            messagebox.showwarning("Warning", "No model file selected.")
            return

        label_encoder_path = filedialog.askopenfilename(title="Select Label Encoder File", filetypes=(("Numpy File", "*.npy"), ("All Files", "*.*")))
        if not label_encoder_path:
            messagebox.showwarning("Warning", "No label encoder file selected.")
            return

        if self.classifier is None:
            self.classifier = ImageClassifier(model_load_path) 

        self.classifier.load_model(model_load_path, label_encoder_path)
        self.status_label.config(text="Status: Model and label encoder loaded successfully.")


    def test_model(self):
        if self.classifier is None or not hasattr(self.classifier, 'model') or self.classifier.model is None:
            messagebox.showwarning("Warning", "Model not loaded. Please load or train a model first.")
            return
        
        image_path = filedialog.askopenfilename(title="Select an Image",
                                                filetypes=(("JPEG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*")))
        if not image_path: 
            return
        
        predicted_class = self.classifier.predict_image_class(image_path)
        
        messagebox.showinfo("Prediction", f"Predicted class: {predicted_class[0]}")
    def compute_and_display_similarity(self):
        image_path = filedialog.askopenfilename(title="Select a Test Image",
                                                filetypes=(("JPEG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*")))
        if not image_path: 
            return
        self.classifier = ImageClassifier(image_path)
        similarity_scores = self.classifier.compute_similarity(image_path, self.features)

        scores_message = "\n".join([f"Reference {i+1}: {score:.2f}" for i, score in enumerate(similarity_scores)])
        messagebox.showinfo("Similarity Scores", scores_message)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()
