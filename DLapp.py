import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import numpy as np
import keras
# Load your trained model here
model = keras.models.load_model('G:/project/deepmodel1.h5')

def predict_cancer(image_path):
    # Preprocess the image for the model
    img = cv2.imread(image_path)

    # Resize the image
    img = cv2.resize(img, (256, 256))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reshape the image to match the input shape of the model
    X = gray.reshape(1, 256, 256, 1)
    
    
    # Make predictions
    prediction = model.predict(X)

    # Convert the predictions to class labels
    class_label = np.argmax(prediction)

    # Print the prediction
    if class_label == 0:
        result="Negative"
    elif class_label == 1:
        result="Positive"
   
    return result

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Cancer Prediction App")
        
        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse)
        self.browse_button.pack(pady=10)
        
        self.image_label = tk.Label(self.master)
        self.image_label.pack()
        
        self.result_label = tk.Label(self.master, text="Result: ")
        self.result_label.pack(pady=10)
        
    def browse(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.show_image(file_path)
            self.show_result(file_path)
    
    def show_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=image)
        self.image_label.image = image
    
    def show_result(self, file_path):
        result = predict_cancer(file_path)
        self.result_label.config(text=f"Result: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Lung Cancer Image Classifier")
    root.geometry("800x600")
    root.configure(bg='#F0F0F0')
    app = App(root)
    root.mainloop()