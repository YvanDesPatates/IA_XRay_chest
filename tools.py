import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


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
