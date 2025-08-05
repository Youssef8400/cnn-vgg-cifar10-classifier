import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.utils import to_categorical

# Charger les données
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float64') / 255.0
y_test = to_categorical(y_test, 10)

# Charger le modèle
model = load_model('youssef999.h5')

# Noms des classes
class_names = [
    "avion", "voiture", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

# Fenêtre principale
root = tk.Tk()
root.title("\U0001F9E0 Application CNN CIFAR-10")
root.geometry("1200x700")
root.configure(bg="#f8f9fa")

# Titre principal
title = tk.Label(root, text="\U0001F9E0 Prédiction CNN sur CIFAR-10", font=("Helvetica", 22, "bold"), bg="#f8f9fa", fg="#343a40")
title.pack(pady=20)

# Cadre pour les boutons
button_frame = tk.Frame(root, bg="#f8f9fa")
button_frame.pack(pady=10)

# Fonction : Affichage d'un nombre personnalisé de prédictions aléatoires
def show_custom_predictions():
    try:
        count = int(num_images_entry.get())
        if count < 1 or count > 100:
            raise ValueError("Entrer un nombre entre 1 et 100")

        rows = int(np.ceil(count / 5))
        fig, axes = plt.subplots(rows, 5, figsize=(12, rows * 2.5))
        axes = axes.flatten()

        for i in range(count):
            idx = np.random.randint(0, len(x_test))
            image = x_test[idx]
            true_label = np.argmax(y_test[idx])
            pred_label = np.argmax(model.predict(image.reshape(1, 32, 32, 3), verbose=0))
            axes[i].imshow(image)
            axes[i].set_title(f"Réel: {class_names[true_label]}\nPré: {class_names[pred_label]}", fontsize=9)
            axes[i].axis("off")

        for i in range(count, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

# Fonction : Prédiction par indice avec affichage amélioré
def predict_by_index():
    try:
        index = int(index_entry.get())
        if index < 0 or index >= len(x_test):
            raise ValueError("Indice invalide.")

        image = x_test[index]
        true_label = np.argmax(y_test[index])
        pred_label = np.argmax(model.predict(image.reshape(1, 32, 32, 3), verbose=0))

        window = tk.Toplevel(root)
        window.title(f"Prédiction - Indice {index}")
        window.configure(bg="#ffffff")

        img = Image.fromarray((image * 255).astype(np.uint8)).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)

        frame = tk.Frame(window, bd=3, relief="ridge", bg="#ffffff")
        frame.pack(padx=20, pady=20)

        label_img = tk.Label(frame, image=img_tk, bg="#ffffff")
        label_img.image = img_tk
        label_img.pack(pady=10)

        label_text = tk.Label(frame, text=f"Réel: {class_names[true_label]}\nPrédit: {class_names[pred_label]}", font=("Arial", 14), bg="#ffffff", fg="#333")
        label_text.pack(pady=5)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))

# Fonction : Importation d'image externe
def load_external_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path).resize((32, 32)).convert("RGB")
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 32, 32, 3)
            prediction = model.predict(img_array, verbose=0)
            pred_label = np.argmax(prediction)

            img_disp = img.resize((128, 128))
            img_tk = ImageTk.PhotoImage(img_disp)

            image_label.config(image=img_tk)
            image_label.image = img_tk

            result_label.config(text=f"Image importée\nPrédiction: {class_names[pred_label]}")
        except:
            messagebox.showerror("Erreur", "Impossible de charger l'image")

# Boutons et entrées avec style
btn_style = {"font": ("Arial", 12), "bg": "#007acc", "fg": "white", "padx": 10, "pady": 5, "bd": 0}

# Bouton 1 : Prédictions aléatoires personnalisées
tk.Label(button_frame, text="Nombre d'images :", font=("Arial", 12), bg="#f8f9fa").grid(row=0, column=0, padx=5)
num_images_entry = tk.Entry(button_frame, width=5, font=("Arial", 12))
num_images_entry.insert(0, "25")
num_images_entry.grid(row=0, column=1, padx=5)
btn1 = tk.Button(button_frame, text="Afficher les prédictions", command=show_custom_predictions, **btn_style)
btn1.grid(row=0, column=2, padx=10)

# Bouton 2 : Prédiction par indice
tk.Label(button_frame, text="Indice [0-9999] :", font=("Arial", 12), bg="#f8f9fa").grid(row=1, column=0, pady=10)
index_entry = tk.Entry(button_frame, width=10, font=("Arial", 12))
index_entry.grid(row=1, column=1, padx=5)
predict_btn = tk.Button(button_frame, text="Prédire", command=predict_by_index, **btn_style)
predict_btn.grid(row=1, column=2, padx=10)

# Cadre de drop/click pour charger une image
drop_frame = tk.Frame(root, bg="#e9ecef", width=300, height=180, relief="groove", bd=2)
drop_frame.pack(pady=20)

drop_label = tk.Label(drop_frame, text="📁 Glissez-déposez une image ici\nou cliquez pour importer", font=("Arial", 12), bg="#e9ecef", fg="#6c757d", justify="center")
drop_label.place(relx=0.5, rely=0.5, anchor="center")

def on_click_import(event):
    load_external_image()

# Associer le clic au cadre
drop_frame.bind("<Button-1>", on_click_import)
drop_label.bind("<Button-1>", on_click_import)

# Image affichée et résultat
image_label = tk.Label(root, bg="#f8f9fa")
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f8f9fa")
result_label.pack()

# Lancer l'application
root.mainloop()





