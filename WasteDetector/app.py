# Required libraries
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO

# --- Auxiliary functions ---

def clean_lbl():
    lblimg.config(image='')
    lblimgtxt.config(image='')

def show_images(img, imgtxt):
    # Displays the images corresponding to the waste type
    try:
        for source_img, label in [(img, lblimg), (imgtxt, lblimgtxt)]:
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            source_img = Image.fromarray(source_img)
            photo = ImageTk.PhotoImage(image=source_img)
            label.configure(image=photo)
            label.image = photo
    except Exception as e:
        print(f"Error displaying images: {e}")
        clean_lbl()

def scanning():
    # Captures and classifies in real time from the camera
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error capturing frame")
            cap.release()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = imutils.resize(frame_rgb, width=640)

        try:
            result = model.predict(source=frame, save=False, verbose=False)[0]
            probs = result.probs

            if probs is not None:
                class_index = int(np.argmax(probs.data.cpu().numpy()))
                confidence = float(np.max(probs.data.cpu().numpy()))
                label_text = f'{clsName[class_index]} {int(confidence * 100)}%'

                # Update the text in the GUI Label
                lblClase.config(text=label_text, fg='green', font=("Arial", 18, 'bold'))

                # Show the text over the video
                cv2.putText(frame_rgb, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (255, 0, 0), 3)

                show_images(class_images[class_index], text_images[class_index])
            else:
                clean_lbl()

        except Exception as e:
            print(f"Error in prediction: {e}")
            clean_lbl()

        # Display in the interface
        im = Image.fromarray(frame_resized)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img

        lblVideo.after(10, scanning)

# --- Main interface ---

def ventana_principal():
    global cap, lblVideo, model, clsName
    global lblimg, lblimgtxt, class_images, text_images, lblClase

    pantalla = Tk()
    pantalla.title("RECICLAJE INTELIGENTE")
    pantalla.geometry("1280x720")

    # Background image
    try:
        fondo_img = PhotoImage(file="Recibot/setUp/Canva.png")
        background = Label(pantalla, image=fondo_img)
        background.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        print(f"Error loading background: {e}")

    # Initialize labels
    lblVideo = Label(pantalla)
    lblVideo.place(x=317, y=127)

    lblimg = Label(pantalla)
    lblimg.place(x=75, y=260)

    lblimgtxt = Label(pantalla)
    lblimgtxt.place(x=995, y=310)

    lblClase = Label(pantalla, text='', font=("Comic Sans MS", 18, 'bold'))
    lblClase.place(x=318, y=127)  # Position of class text

    # Load model
    try:
        model = YOLO("Recibot/runs/classify/train6/weights/best.pt")
        clsName = ['metal', 'organico', 'papel_y_carton', 'plastico', 'vidrio']
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load images for each class
    rutas_imgs = [
        "metal", "organico", "papel_y_carton", "plastico", "vidrio"
    ]
    class_images = []
    text_images = []

    for clase in rutas_imgs:
        try:
            img = cv2.imread(f"Recibot/setUp/{clase}.png")
            txt = cv2.imread(f"Recibot/setUp/{clase}txt.png")

            if img is None or txt is None:
                raise ValueError(f"Images not found for: {clase}")

            class_images.append(img)
            text_images.append(txt)
        except Exception as e:
            print(f"Error loading images for {clase}: {e}")
            class_images.append(np.zeros((100, 100, 3), dtype=np.uint8))
            text_images.append(np.zeros((100, 100, 3), dtype=np.uint8))

    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    scanning()
    pantalla.mainloop()

# Run application
ventana_principal()
