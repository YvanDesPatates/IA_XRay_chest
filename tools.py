import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


# Permet de parcourir les images, et pour chaque image, on applique une fonction de callback
# On peut optionnellement appeler une fonction de callback pour chaque dossier
def browse_imgs(img_callback, path_folder_callback=None, limit_size=None):
    common_path = "./chest_Xray"
    images_files = os.listdir(common_path)
    subfolders = ["train", "val", "test"]
    categories = ["NORMAL", "PNEUMONIA"]

    for subfolder in subfolders:
        for category in categories:
            # pour avoir tous les chemins des 6 dossiers
            folder_path = os.path.join(common_path, subfolder, category)
            # liste de toutes les images
            images_files = os.listdir(folder_path)
            if path_folder_callback is not None:
                path_folder_callback(folder_path, images_files)
            array_limit = limit_size if limit_size is not None else len(images_files)
            # récupération de toutes les (ou des 'limit_size' premières) images du dossier.
            for file_name in images_files[:array_limit]:
                if not file_name.endswith(".jpeg"):
                    continue
                image_path = os.path.join(folder_path, file_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_callback(img, category)


def display_imgs(imgs, titles=[], plot_size=(1, 1), figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    index = 0
    for image, title in zip(imgs, titles):
        index += 1
        ax = fig.add_subplot(plot_size[0], plot_size[1], index)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        if titles is not None:
            ax.set_title(title)

    plt.tight_layout()
    plt.show()


def display_distribution(datasetsY, datasets_names):
    fig, ax = plt.subplots(1, len(datasetsY), figsize=(5 * len(datasetsY), 5))
    # iterate on each datasetY to count the number of each category
    for index, datasetY in enumerate(datasetsY):
        dataset = datasetY.tolist()
        ax[index].bar(["NORMAL", "PNEUMONIA"], [dataset.count(0), dataset.count(1)])
        ax[index].set_title(datasets_names[index])

    plt.show()


def display_confusion_matrix_and_scores(y_test, y_pred):
    # Convert probabilities to class labels
    y_pred = np.round(y_pred).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
    plt.show()

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    print(f"recall : {recall} | precision : {precision} | f1score : {f1score}")


def plot_history(history, metrics, fig_size=(20, 6), nb_line=1, nb_column=None):
    if nb_column is None:
        nb_column = len(metrics)

    plt.figure(figsize=fig_size)
    plt.title("Evolution du modèle")

    index = 0
    for metric_name in metrics:
        index += 1
        plt.subplot(nb_line, nb_column, index)
        plt.plot(history.history[metric_name], marker='o', linestyle='-', label=metric_name)
        val_scores = "val_" + metric_name
        plt.plot(history.history[val_scores], label=val_scores)
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()

    plt.show()


def learning_curve():
    x_total = []
    y_total = []

    image_size = (100, 100)

    def load_datasets(img, category):
        new_img = cv2.resize(img, image_size)
        x_total.append(new_img)
        category = 0 if category == "NORMAL" else 1
        y_total.append(category)

    browse_imgs(load_datasets)

    x_total = np.array(x_total) / 255
    y_total = np.array(y_total)

    # on créer un model sans chercher à l'optimiser, juste pour avoir une idée de la learning curve
    num_classes = 1
    core_size = 8
    model = tf.keras.Sequential([
        layers.Input(shape=(100, 100, 1)),
        layers.Conv2D(16, core_size, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, core_size, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, core_size, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(300, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['recall'])

    max_dataset_size = len(y_total)
    tests_results = {}
    train_results = {}
    steps = [2500, 3500, 4000, max_dataset_size]
    for nb_img in steps:
        x_data = x_total[:nb_img]
        y_data = y_total[:nb_img]
        trainx, valx, trainy, valy = train_test_split(x_data, y_data, test_size=0.2)
        valx, testx, valy, testy = train_test_split(valx, valy, test_size=0.5)

        model.fit(trainx,
                  trainy,
                  validation_data=(valx, valy),
                  epochs=2,
                  verbose=0)

        y_pred = model.predict(testx, verbose=0)
        y_pred_label = np.round(y_pred).astype(int)
        tests_results[nb_img] = recall_score(testy, y_pred_label)

        y_pred = model.predict(trainx, verbose=0)
        y_pred_label = np.round(y_pred).astype(int)
        train_results[nb_img] = recall_score(trainy, y_pred_label)

    # plotting the learning curve
    plt.plot(tests_results.keys(), tests_results.values(), marker='.', label='Test Recall')
    plt.plot(train_results.keys(), train_results.values(), marker='.', label='Train Recall')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model recall')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
