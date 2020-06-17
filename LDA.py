import os
import cv2
import numpy as np
from numpy import linalg as LA
import math as m
from numpy.linalg import inv
import matplotlib.pyplot as plt
from classification import readImage, gaussianKernel, convolution


# Reads the folder and each filename inside the folder
# returns the dictionary of Images with values as File name
def filename(path):
    dir_list = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    images = {}
    for folder in dir_list:
        img_filename = os.listdir(os.path.join(path, folder))
        images[folder] = {'Filename': img_filename[:100]}
    return images


def compute_mean(data):
    data_matrix = np.array(data)
    return np.mean(data_matrix, axis=0)


# with given dataset and overall mean, gives the overall covariance matirx
def find_covariance(dataset, mean):
    x_overall = np.array([])
    cov = np.matrix(np.zeros(num_of_features* num_of_features).reshape(num_of_features, num_of_features))
    for label, attributes in dataset.items():
        x = attributes['features'][:num_of_instances] - mean
        x_overall = np.append(x_overall, x)
    x = np.matrix(x_overall).reshape(num_of_labels * num_of_instances, num_of_features)
    for i in range(num_of_labels * num_of_instances):
        cov = cov + x[i].T * x[i]
    cov = (1/(num_of_labels * num_of_instances)) * cov
    return cov


# Classifies the 50 training images of Each digits
# calculates the LDF(g) for all the classes on each image
# Class with maximum g is the predicted class
def prediction_func(dataset, sample):
    predictions = []
    actual = []
    if sample is 'train':
        k = 0
        j = 50
    else:
        k = 50
        j = 100

    for digit in range(num_of_labels):
        for i in range(k, j):
            x = np.array([dataset[str(digit)]['features'][i]])
            g_arr = np.array([])
            labels = np.array([])
            for label, attributes in dataset.items():
                h = attributes['h']
                b = attributes['b']
                g = np.matmul(x, h) + b
                g_arr = np.append(g_arr, g)
                labels = np.append(labels, int(label))
            ind = np.argmax(g_arr)
            prediction = int(labels[ind])
            predictions.append(prediction)
            actual.append(digit)

    return actual, predictions


# For the given dataset, computes h and b value for each class
def compute_filters(dataset, covariance_matrix):
    h_arr = np.array([])
    b_arr = np.array([])
    for label, attributes in dataset.items():
        m = np.array([attributes['mean']])
        h = np.matmul(inv(covariance_matrix), np.transpose(m))
        dataset[label]['h'] = h
        b = (np.matmul(m, h)) * -0.5
        dataset[label]['b'] = b
    return dataset


# Compares between acctual and predicted class and find how many correct prediction
def find_accuracy(actual, predictions):
    correct = 0
    # comparing each values in list if they are similar
    for x, y in zip(actual, predictions):
        if x == y:
            correct += 1
    accuracy = (correct/float(len(actual))) * 100
    return accuracy


# Function to plot the confusion matrix
def plotConfusionMatrix(test_set, y_pred, cm,  normalize=True, title=None, cmap = None, plot = True):

    # Compute confusion matrix
    # Find out the unique classes
    classes = list(np.unique(list(test_set)))
    if cmap is None:
        cmap = plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Predicted label',
           xlabel='Actual label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if plot:
        plt.show()


# Generate a confusion matrix from predicted and actual classification
def find_confusion_matrix(p, a):
    num_of_classes = len(set(a))
    confusion_matrix = np.zeros((num_of_classes, num_of_classes))
    for i in range(len(a)):
        confusion_matrix[p[i]][a[i]] += 1
    # print('confusion matrix:',confusion_matrix)
    total = np.sum(confusion_matrix, axis=0)
    error = []
    for i in range(num_of_classes):
        error.append(((total[i] - confusion_matrix[i][i]) / total[i]) * 100)
    return confusion_matrix, error


# Finds the 20 R score for each training example
def compute_H_xy(s_xx, s_yy, s_xy,rgb_image, k, threshold):
    det = (s_xx * s_yy) - s_xy ** 2
    trace = s_xx + s_yy
    r = det - k * (trace ** 2)
    flattened_r = r.flatten()
    sorted_r = np.sort(flattened_r)
    feature = np.concatenate((sorted_r[:10], sorted_r[-10:]))
    return feature


# Implementation of Harris Corner detection and returns R score of each training images
def compute_feature(image_path, filename):
    # Read RGB image
    rgb_image = readImage(image_path+filename)
    # print("Reading:", image_path+filename)
    # convert RGB to gray scale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # generates a gaussian filter
    gaussian_filter = gaussianKernel(filter_size=3, sigma=5)
    # Performs convolution between gaussian filter and gray image
    gaussian_image = convolution(gaussian_filter, gray_image)

    # defining horizontal and vertical filter
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Derivative of gaussian image along x and y axes
    i_x = convolution(g_x, gaussian_image)
    i_y = convolution(g_y, gaussian_image)

    # Auto-correlation of gradient image
    i_xx = i_x * i_x
    i_yy = i_y * i_y
    # cross-correlation between x and y derivatives of image
    i_xy = i_x * i_y

    # smoothing window function with all the elements as one
    window = np.ones((3, 3), dtype=np.uint8)

    # convolution of i_xx, i_yy and i_xy with window function
    s_xx = convolution(window, i_xx)
    s_yy = convolution(window, i_yy)
    s_xy = convolution(window, i_xy)

    k = 0.06
    threshold = 12000

    # computes feature vector for each image using hessian matrix
    feature = compute_H_xy(s_xx, s_yy, s_xy, rgb_image, k, threshold)
    return feature


if __name__ == "__main__":
    path = '../data/DigitDataset/'
    # Read the name of files
    images = filename(path)
    mean_array = []

    # Generating feature vector for each image of every digit
    for digits, features in images.items():
        dataset = []
        full_path = path + str(digits) + '/'
        # computes feature of each image and append in dataset
        dataset.append([compute_feature(full_path, features['Filename'][i])
                        for i in range(len(features['Filename']))])
        a = np.array(dataset).reshape(-1, 20)
        images[digits]['features'] = a
        # mean of each class
        mean = compute_mean(a)
        images[digits]['mean'] = mean
        mean_array.append(mean)

    avg_mean = compute_mean(mean_array)
    num_of_instances = 50
    num_of_features = 20
    num_of_labels = len(images.keys())

    covariance_matrix = find_covariance(images, avg_mean)
    images = compute_filters(images, covariance_matrix)
    # Two loops when taking training and test sample for prediction
    test_sample = ['test', 'train']
    for i in range(2):
        actual, predictions = prediction_func(images, test_sample[i])
        accuracy = find_accuracy(actual, predictions)
        print('Accuracy:', accuracy)
        confusion_matrix, error = find_confusion_matrix(predictions, actual)
        print("Error:", error)
        plotConfusionMatrix(actual, predictions, confusion_matrix,
                            normalize=True,
                            title=None,
                            cmap=None, plot=True)
















