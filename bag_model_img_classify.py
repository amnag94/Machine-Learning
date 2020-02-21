import os

import numpy as np
import pandas as pd

import cv2 as cv

from sklearn.cluster import KMeans
from sklearn import svm

def showImage(img, img_name):
    '''
        Show the image img in the window and give it
        a name using img_name
    :param img: image to display
    :param img_name: name of image for window name
    :return:
    '''

    cv.imshow(img_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def getKeyPoints(img_gray, img):
    '''
        Use SIFT from the cv2 library to get key points
        for the image passed i.e. img
    :param img_gray: grayscale of image img
    :param img: image to find key points for
    :return: detected key points and sift created
    '''

    sift = cv.xfeatures2d_SIFT.create()
    key_points = sift.detect(img_gray, None)

    img = cv.drawKeypoints(img_gray, key_points, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imwrite('sift_keypoints.jpg', img)

    #showImage(img, 'Image with key points')

    return key_points, sift

def computeDescriptors(img_gray, sift, key_points):
    '''
        Compute the descriptors for the key points
        and return them.
    :param img_gray: grayscale of image img
    :param sift: the SIFT variable created using cv2
    :param key_points: key points already detected
    :return: key points and descriptor
    '''

    return sift.compute(img_gray, key_points)

def formBagOfWords(descriptors, words_count):
    '''
        Using kmeans cluster the vectors in the
        descriptor to form groups that will form
        the words for bag of words.
    :param descriptors: around of vectors for key points
        in the image
    :param words_count: number of clusters for kmeans
    :return: kmeans model representing cluster centers,
        bag of words dictionary
    '''

    k_mean = KMeans(n_clusters=words_count, init="random", random_state=0)

    # Fit descriptor to kmeans
    # Assigns each vector in descriptor to a cluster
    k_mean.fit(descriptors)

    # Get labels for the vectors to form bag of words
    labels_vectors = np.array(k_mean.labels_)

    # Bag of words is a dictionary with size of clusters
    bag_of_words = {}

    # Count values for each cluster and add in the bag
    for cluster_index in range(words_count):
        # numpy arrays can use == to form array with
        # those values from original array
        bag_of_words[cluster_index + 1] = len(labels_vectors[labels_vectors == cluster_index])

    return k_mean, bag_of_words

def baggingOneImage(img_src, words_count):
    '''
        Perform bagging for this image with k means and
        bag of words.
    :param img_src: path for this image
    :return: bag of words for this image
    '''

    # Get the image
    img = cv.imread(img_src)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find key points for img using SIFT
    key_points, sift = getKeyPoints(img_gray, img)

    # Compute descriptors for key points
    key_points, descriptors = computeDescriptors(img_gray, sift, key_points)

    # Get clusters for vectors in descriptor with kmeans
    # and form the bag of words
    k_mean, bag_of_words = formBagOfWords(descriptors, words_count=words_count)

    return bag_of_words

def train_for_set(set_name, path, word_count):
    '''
        Form bag of words for images in given
        set
    :param set_name: name of train set
    :param path: path to set directory
    :param word_count: count for clusters
        for k means
    :return: set of bag of word attributes for
        this set
    '''

    # os.listdir gets list of images in directory at path
    train_set = os.listdir(path)

    # Set of bag of words for this set with labels
    classifier_train_set = []

    for image in train_set:
        bag_for_image = baggingOneImage(path + "/" + image, word_count)
        bag_values = list(bag_for_image.values())

        # Add the label
        bag_values.append(set_name)

        # Add to classifier train set
        classifier_train_set.append(bag_values)

        #print(image + " : %s" %bag_for_image)

    #print(classifier_train_set)
    return classifier_train_set

def main():

    # Parameters
    word_count = 20

    '''
        Training
    '''

    # We form a training set for classifier by adding
    # bag of words for each image in train set and
    # it is stored in this set
    classifier_train_set = []

    print("Accordian")
    # Accordion training set
    accordian_set = train_for_set('accordion', 'train/accordion', word_count)

    classifier_train_set.append(accordian_set)

    print("Dollar Bill")
    # Dollar Bill training set
    dollar_set = train_for_set('dollar_bill', 'train/dollar_bill', word_count)

    classifier_train_set.append(dollar_set)

    print("Motorbike")
    # Motorbike training set
    motorbike_set = train_for_set('motorbike', 'train/motorbike', word_count)

    classifier_train_set.append(motorbike_set)

    print("Soccer Ball")
    # Soccer Ball training set
    soccer_ball_set = train_for_set('Soccer_Ball', 'train/Soccer_Ball', word_count)

    classifier_train_set.append(soccer_ball_set)

    # We shuffle the data to enable training better


    # Train classifier using generated classifier train set


    print(classifier_train_set)

    #baggingOneImage("Lenna.png", word_count=word_count)

if __name__ == '__main__':
    main()