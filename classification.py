import cv2
import numpy as np
from numpy import linalg as LA
import math as m
import matplotlib.pyplot as plt


def mean(x):
    return sum(x)/(len(x))


def standard_deviation(x):
    avg = mean(x)
    sd = m.sqrt(sum([pow(i - avg, 2) for i in x]) / float(len(x) - 1))
    return sd


# Function to plot the feature vector distribution
def plot_feature(dict_class):
    i = 1
    for target, features in dict_class.items():
        lamda1_corners = [features[x][0] for x in range(len(dict_class[target]))]
        lamda2_corners = [features[x][1] for x in range(len(dict_class[target]))]
        plt.subplot(1, 3, i)
        plt.scatter(lamda1_corners, lamda2_corners)
        plt.xlabel('lamda 1')
        plt.ylabel('lamda 2')
        if target is 'corner':
            plt.title("Corner Distribution")
        elif target is 'edge':
            plt.title("Edge Distribution")
        else:
            plt.title("Flat Distribution")
        i = i+1
    plt.show()

# reads image and return it as numpy array with pixel values
def readImage(image):
    img = cv2.imread(image)
    return img


# saves Image in the working directory
def saveImage(name, image):
    cv2.imwrite(name, image)


# Finds the convolution between Image and given filter and returns new image array
def convolution(filter, gray_img):
    h, w = gray_img.shape[:2]
    offset = filter.shape[0] // 2
    new_Image = np.zeros((h, w))
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            image_filter = gray_img[y - offset:y + offset + 1, x - offset:x + offset + 1] * filter
            conv = image_filter.sum()
            new_Image[y, x] = abs(conv)
    return new_Image


# Make a dictionary with key as classes name and value as list of instances of that class
def group_by_class(dataset):
    dict_class = {}
    for i in range(len(dataset)):
        row = dataset[i]
        # last element of Instance is class of actual dataset
        target_class = row[-1]
        if target_class not in dict_class:
            dict_class[target_class] = []
        dict_class[target_class].append(row[:-1])
    return dict_class


# For each class find the mean and SD of each features
# Here key is class name and vlaue has four list as [mean,sd], for each instances
def mean_Sd_class( dict_class):
    dict_mean_SD = {}
    for target, features in dict_class.items():
        dict_mean_SD [target] = [(mean(attributes), standard_deviation(attributes)) for attributes in zip(*features)]
    return dict_mean_SD


# Finding likelihood of test data for each classes
# mathematical implementation of probablity density function of normal distribution
def joint_class_probablities(dict_mean_sd, dict_class, test_data):
    joint_prob = {}
    for target, features in dict_mean_sd.items():
        total_features = len(features)
        likelihood = 1
        for i in range(total_features):
            x = test_data[i]
            mean, sd = features[i]
            normal_prob = gaussian_pdf(x, mean, sd)
            likelihood *= normal_prob
        joint_prob[target] = likelihood
    return joint_prob


# class wtih maximum joint probability is considered as predicted class
# returns the class prediction for each testdata
def classification_function(dict_mean_sd, dict_class,  test_data):
    predictions = []
    for i in range(len(test_data)):
        probablities = joint_class_probablities(dict_mean_sd, dict_class, test_data[i])
        prediction = max(probablities, key=probablities.get)
        predictions.append(prediction)
    return predictions


# Finds the accuracy of the predicted result comparing with real test data
def find_accuracy(test_data, predictions):
    correct = 0
    # last element of test set is real classification,
    # so making list of true classification
    actual = [instances[-1] for instances in test_data]
    # comparing each values in list if they are similar
    for x, y in zip(actual, predictions):
        if x == y:
            correct += 1
    accuracy = (correct/float(len(actual))) * 100
    return accuracy


# Find the probablity from normal distribution
def gaussian_pdf(x, mean, sd):
    e = m.exp(-(m.pow(x-mean, 2) / (2*m.pow(sd, 2))))
    denominator = m.sqrt(2 * m.pi) * sd
    pdf = e / denominator
    return pdf


# computes the gaussian filter with given size and standard deviation value
def gaussianKernel(filter_size, sigma):
    g_filter = np.zeros((filter_size, filter_size), np.float32)
    x = filter_size // 2
    y = filter_size // 2

    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    g_filter = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    return g_filter


# Function to plot the confusion matrix
def plotConfusionMatrix(test_set, y_pred, cm,  normalize=True, title=None, cmap = None, plot = True):

    # Compute confusion matrix
    # Find out the unique classes
    y_true = []
    for x in range(len(test_set)):
        test = test_set[x][-1]
        y_true.append(test)
    classes = list(np.unique(list(y_true)))

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


# computes the confusion matrix and error rate
def find_confusion_matrix(predictions, actual):
    actual_class = []
    for x in range(len(actual)):
        test = actual[x][-1]
        actual_class.append(test)
    num_of_classes = len(set(actual_class))
    # replacing the class name with integer value
    p = list(map(lambda x: 0 if x == "corner" else x, predictions))
    p = list(map(lambda x: 1 if x == "edge" else x, p))
    p = list(map(lambda x: 2 if x == "flat" else x, p))

    a = list(map(lambda x: 0 if x == "corner" else x, actual_class))
    a = list(map(lambda x: 1 if x == "edge" else x, a))
    a = list(map(lambda x: 2 if x == "flat" else x, a))

    confusion_matrix = np.zeros((num_of_classes, num_of_classes))
    for i in range(len(actual_class)):
        confusion_matrix[p[i]][a[i]] += 1
    # print('confusion matrix:',confusion_matrix)
    total = np.sum(confusion_matrix, axis=0)
    error = []
    for i in range(num_of_classes):
        error.append(((total[i] - confusion_matrix[i][i])/total[i]) * 100)
    return confusion_matrix, error


# computes H-matrix of Harris detector with given parameters
# first finds determinant and traces of the matrix
# finally compare the response r value with threshold
# returns an image by marking corners
def compute_H_xy(s_xx, s_yy, s_xy,rgb_image, k, threshold):
    det = (s_xx * s_yy) - s_xy ** 2
    trace = s_xx + s_yy
    r = det - k * (trace ** 2)
    h, w = s_xx.shape[:2]
    dataset = []

    for y in range(h):
        for x in range(w):
            eigenvalues, _ = LA.eig(np.array([[s_xx[y, x], s_xy[y, x]],
                                              [s_xy[y, x], s_yy[y, x]]]))
            l1 = eigenvalues[0]
            l2 = eigenvalues[1]
            if r[y, x] > threshold:
                dataset.append([l1, l2, 'corner'])
            elif r[y, x] < -threshold:
                dataset.append([l1, l2, 'edge'])
            else:
                dataset.append([l1, l2, 'flat'])

    return dataset


if __name__ == "__main__":
    # Read RGB image
    rgb_image = readImage('../data/input_hcd1.jpg')
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

    # computes H matrix and obtain image with corners
    dataset = compute_H_xy(s_xx, s_yy, s_xy, rgb_image, k, threshold)
    dict_class = group_by_class(dataset)
    dict_mean_sd = mean_Sd_class(dict_class)

    predictions = classification_function(dict_mean_sd, dict_class, dataset)
    accuracy = find_accuracy(dataset, predictions)

    plot_feature(dict_class)
    confusion_matrix, error = find_confusion_matrix(predictions, dataset)
    print("Confusion Matrix", confusion_matrix)
    print('Accuracy:', accuracy)
    print('Error Rate[Corner,edge,flat]:', error)
    plotConfusionMatrix(dataset, predictions, confusion_matrix,
                        normalize=True,
                        title=None,
                        cmap=None, plot=True)










