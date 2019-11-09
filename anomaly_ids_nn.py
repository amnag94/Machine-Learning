import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

def half_normal(train_set):
    '''
        Use half of normal set. Sets X and Y for training with 200000 normal rows and all attack rows (about 170000)
    '''

    # normal_half = train_set[train_set['label'] == 0.0]
    # length = len(train_set[train_set['label'] == 0.0]) / 2

    #     normal_half = train_set[train_set['label'] == 0.0][0:200000]
    normal_half = train_set[train_set['label'] == 0].sample(200000)

    normal_half = normal_half.append(train_set[train_set['label'] == 1])

    X_training = normal_half.drop('label', axis=1)

    # Get one hot style lists

    Y_training = np.array(normal_half['label']).reshape(-1, 1)
    # Y_training = pd.get_dummies(Y_training)

    return X_training, Y_training

def mini_batching(X_set, Y_set, ind, batch):
    '''
        Returns slices of the sets of size batch
    '''
    if (ind + batch - 1) >= len(X_set):
        resX = X_set[ind:]
        resY = Y_set[ind:]
    else:
        resX = X_set[ind:ind + batch]
        resY = Y_set[ind:ind + batch]
    return resX, resY

def testing(test, weights1, weights2, bias,  num_features):
    '''
        Testing model
    '''


    # Test sets
    X_test = test.drop('label', axis=1)
    Y_test = np.array(test['label']).reshape(-1, 1)

    # Testing
    test_num = X_test.shape[0]
    XT = tf.placeholder(shape=(test_num, num_features), dtype=tf.float64, name='XT')
    YT = tf.placeholder(shape=(test_num, 1), dtype=tf.int64, name='YT')

    WT1 = tf.Variable(weights1)
    WT2 = tf.Variable(weights2)
    B1 = tf.Variable(bias)

    AT1 = tf.nn.relu(tf.matmul(XT, WT1))
    Y_est = tf.nn.sigmoid(tf.matmul(AT1, WT2) + B1)

    # Initialize tensor flow
    init = tf.global_variables_initializer()

    # Session
    with tf.Session() as sess:
        sess.run(init)
        y_est_list = sess.run(Y_est, feed_dict={XT: X_test, YT: Y_test})

    threshold = 0.7

    # Accuracy
    estimate = np.where(y_est_list < threshold, 0, 1)
    accuracy = np.mean(estimate == Y_test.astype(int))

    # False positive and negative ratios
    conf = confusion_matrix(Y_test.astype(int), estimate, labels=[0, 1])

    false_pos = conf[0][1] / (conf[0][0] + conf[0][1])
    false_neg = conf[1][0] / (conf[1][0] + conf[1][1])

    print('Accuracy : %s ' % (accuracy * 100))

    print('Confusion Matrix :')
    print(conf)

    print('False pos ratio : %s ' % false_pos)
    print('False neg ratio : %s ' % false_neg)

def fit(train, num_rows, num_features, num_hid_nodes, num_classes, epoch):
    '''
        Training tensor flow neural network
    '''

    # Generate matrices using tensorflow elements
    X = tf.placeholder(shape=(num_rows, num_features), dtype=tf.float64, name='X')
    Y = tf.placeholder(shape=(num_rows, num_classes), dtype=tf.float64, name='Y')

    W1 = tf.Variable(np.random.rand(num_features, num_hid_nodes), name='W1')
    W2 = tf.Variable(np.random.rand(num_hid_nodes, num_classes), name='W2')
    b = tf.Variable(np.zeros(1), name='b')

    # Define equations
    H1 = tf.nn.relu(tf.matmul(X, W1))
    H2 = tf.matmul(H1, W2) + b
    Y_pred = tf.nn.sigmoid(H2)

    # Set loss functions
    delta = tf.keras.losses.binary_crossentropy(Y, Y_pred)
    loss = tf.reduce_mean(delta)

    # Set train instructions for tensorflow
    optimizer = tf.train.AdamOptimizer(0.0001)
    training = optimizer.minimize(loss)

    # Initiate session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    losses = []

    # Iterate epoch times for each data training
    for epoch_index in range(epoch):

        # Shuffle training samples
        train = train.sample(frac=1)
        X_train, Y_train = half_normal(train)

        count = 0
        index = 0

        while index < len(X_train):
            X_t, Y_t = mini_batching(X_train, Y_train, index, num_rows)

            if X_t.shape[0] < num_rows:
                break
            index += num_rows
            #print(index)

            # Train model with shuffled training set
            sess.run(training, feed_dict={X: X_t, Y: Y_t})
            new_loss = sess.run(loss, feed_dict={X: X_t.as_matrix(), Y: Y_t})

            # If loss is constant for too long
            if len(losses) == 0 or new_loss != losses[-1]:
                count = 0
            elif new_loss == losses[-1]:
                count += 1

            losses.append(new_loss)

            if count > len(X_train) / 10:
                break

            weights1 = sess.run(W1)
            weights2 = sess.run(W2)
            bias = sess.run(b)

    sess.close()

    return losses, weights1, weights2, bias

def main():
    # Data
    dataset = pd.read_csv('consolidated.csv')

    # Replace with only 0 and 1 as labels
    attacks = list(dataset['label'].unique())
    attacks.remove('normal')
    dataset['label'] = dataset['label'].replace('normal', 0)
    dataset['label'] = dataset['label'].replace(attacks, 1)

    # 70:30 train : test

    train, test = train_test_split(dataset, test_size=0.30, random_state=42, stratify=dataset['label'])

    X_train, Y_train = half_normal(train)

    # Number of iterations

    # For comparing epoches
    #epoches = [1, 5, 50]

    # Best epoch
    epoches = [50]

    num_features = 25

    # Batches
    num_rows = 2000

    num_hid_nodes = 12

    # Number of output nodes
    num_classes = 1

    for epoch in epoches:

        print('Epoch : %s' % epoch)

        start = time.time()
        losses, weights1, weights2, bias = fit(train, num_rows, num_features, num_hid_nodes, num_classes, epoch)
        end = time.time()

        train_time = end - start

        start = time.time()
        testing(test, weights1, weights2, bias, num_features)
        end = time.time()

        test_time = end - start

        print('Time : \n Train : %s' % train_time)
        print('Test : %s' % test_time)

if __name__ == '__main__':
    main()