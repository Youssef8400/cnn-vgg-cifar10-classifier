# cnn-vgg-cifar10-classifier

Créer, entraîner et évaluer un modèle de réseau de neurones convolutifs (CNN) basé sur l'architecture VGG avec Keras pour classer les images du dataset CIFAR-10.


*Configuration & Exécution*

- Python : version recommandée : Python 3.10 ou supérieure
- pip install -r requirements.txt (Installation des dépendances)
- python model.py ( Pour entraîner le modèle , et le sauvegarder dans un fichier .h5 )
- python deploiement.py ( Pour lancer l’interface graphique de prédiction )
  









---

## 1. Description du Dataset

Le dataset CIFAR-10 contient 60 000 images couleur en 32x32 pixels réparties en 10 classes : avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau et camion.

- Format des images : (32, 32, 3)
  - 32x32 : taille réduite qui permet un entraînement rapide.
  - 3 : les images sont en couleur (RVB).
- Répartition :
  - 50 000 images pour l'entraînement
  - 10 000 images pour le test

---

## 2. Prétraitement


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

- Normalisation des images (division par 255) pour ramener les valeurs entre 0 et 1.

- Conversion des étiquettes en one-hot encoding.

- Affichage des dimensions et distribution des classes.

## 3. Architecture CNN (VGG simplifié) 

Le modèle est basé sur une version simplifiée de l’architecture VGG :

Trois blocs composés de :

  Deux couches Conv2D avec activation ReLU

  Une couche MaxPooling2D

  Une couche Dropout pour la régularisation

Partie finale :

  Flatten

  Dense(1024) + Dropout

  Dense(10, activation='softmax') pour la classification

<div align="center"> <img src="https://github.com/user-attachments/assets/fe178fc0-f5ec-4154-8e1a-632ec595905d" width="500"/> <br><i>Architecture du modèle</i> </div>


## 4. Compilation et Entraînement

- Optimiseur : Adam

- Fonction de perte : Categorical Crossentropy

- Entraînement sur 50 époques

- Taille de batch : 64

- Validation sur le jeu de test

<div align="center"> <img src="https://github.com/user-attachments/assets/6c460fec-95e4-4a5f-8037-eb3d1fdee006" width="700"/> <br><i>Courbes de précision et de perte</i> </div>



## 5. Évaluation du modèle

<div align="center"> <img src="https://github.com/user-attachments/assets/feeb3838-3a71-485d-b5e2-a115a18cae24" width="400"/> <br><i>Résultats sur le jeu de test</i> </div>

## 6. Sauvegarde du modèle

Le modèle entraîné est sauvegardé dans le fichier youssef999.h5 :


## 7. Interface graphique avec Tkinter
Une interface graphique permet de tester le modèle avec différentes options :

7.1 Prédiction d’un nombre d’images aléatoires
Exemple 1 :

<div align="center"> <img src="https://github.com/user-attachments/assets/d8aa93c5-4593-497d-9bda-5806feafef5a" width="800"/> </div>
Exemple 2 :

<div align="center"> <img src="https://github.com/user-attachments/assets/a30a7a30-6d06-4559-ba44-8418bf13014b" width="800"/> </div>
7.2 Prédiction à un indice donné
Exemple 1 :

<div align="center"> <img src="https://github.com/user-attachments/assets/846c4a13-5318-4d45-8968-eca6ea1a44c5" width="400"/> </div>
Exemple 2 :

<div align="center"> <img src="https://github.com/user-attachments/assets/dcaa5040-f241-48f0-b7ae-625dcfc34512" width="300"/> </div>
7.3 Importer une image externe
Exemple 1 :

Image à prédire :

![cheval](https://github.com/user-attachments/assets/d1c0f59b-5b01-4a49-9e05-57ec841ab923)


Prédiction :

<div align="center"> <img src="https://github.com/user-attachments/assets/de484ca5-7ac8-4fd4-96dc-9876e96246bc" width="400"/> </div>
Exemple 2 :

Image à prédire :

![camio](https://github.com/user-attachments/assets/e3c1e96c-737f-4a6a-8dac-6a3f519a5db4)


Prédiction :

<div align="center"> <img src="https://github.com/user-attachments/assets/d30243b1-16ce-4f1f-90f7-c9d6fba20de3" width="400"/> </div>



## 8. Matrice de confusion
<div align="center"> <img src="https://github.com/user-attachments/assets/9f42c9b7-14ce-43f0-a937-728596c2ace1" width="700"/> <br><i>Matrice de confusion du modèle</i> </div>


## 9. Limites du modèle

- Confusions fréquentes entre classes visuellement similaires (chat/chien, oiseau/avion).

- Images petites et bruitées rendant la classification difficile.

- VGG est une architecture simple comparée à des modèles plus récents comme ResNet ou EfficientNet.

- Pas d’augmentation de données (data augmentation) utilisée.



## 10. Pistes d’amélioration

- Ajouter des techniques de data augmentation.

- Utiliser une architecture plus avancée.

- Ajouter du fine-tuning avec des poids pré-entraînés.

- Régler dynamiquement le taux d’apprentissage.

- Appliquer une évaluation plus détaillée par classe.



