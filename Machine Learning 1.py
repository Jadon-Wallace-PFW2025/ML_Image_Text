import os
import cv2
import pytesseract
from PIL import Image
import csv
import matplotlib.pyplot as plt

# Function to read image files in the current directory where the script is located
def read_image_files(directory):
    image_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_files.append(os.path.join(directory, filename))
    return image_files

# Function to recognize text from an image using Tesseract
def recognize_text(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    
    # Pre-process image (optional: apply smoothing, contrast enhancement, edge enhancement)
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Smoothing
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to do OCR on the image
    recognized_text = pytesseract.image_to_string(gray)
    
    return recognized_text

# Function to display the image and recognized text
def display_image_and_text(image_path, recognized_text):
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Recognized Text: {recognized_text}")
    plt.axis("off")
    plt.show()

# Function to save recognized text and image file names in a CSV file
def save_to_csv(image_files, recognized_texts, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image File Name', 'Recognized Text'])
        for image_file, recognized_text in zip(image_files, recognized_texts):
            writer.writerow([os.path.basename(image_file), recognized_text])

# Main function to process images and save recognized text to CSV
def main():
    # Get the directory where this script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Use the script directory as the image directory
    image_directory = script_directory  # Looks in the same directory as the script
    
    # Output CSV will also be saved in the same directory
    output_csv = os.path.join(script_directory, 'output.csv')
    
    # Step 1: Read image files
    image_files = read_image_files(image_directory)
    
    # Step 2 & 3: Process images and recognize text
    recognized_texts = []
    for image_file in image_files:
        recognized_text = recognize_text(image_file)
        recognized_texts.append(recognized_text)
        display_image_and_text(image_file, recognized_text)
    
    # Step 4: Save results to CSV
    save_to_csv(image_files, recognized_texts, output_csv)

if __name__ == "__main__":
    main()
