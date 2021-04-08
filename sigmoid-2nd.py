import tensorflow as tf
#import pandas as pd
import tflearn
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
#import matplotlib.pyplot as plt

#trainingData =np.load('upsampledtrain-50-50-20.npy',allow_pickle=True)
#validationData =np.load('upsampledtest-50-50-20.npy',allow_pickle=True)
#much_data= np.load('imageDataNew-50-50-20.npy',allow_pickle=True)
much_data= np.load('halfcdata-50-50-20.npy',allow_pickle=True)
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
trainingData = much_data[:-100]
validationData = much_data[-100:]

x = tf.placeholder('float')
y = tf.placeholder('float')
size = 50
keep_rate = 0.8
NoSlices = 20

def convolution3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpooling3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def cnn(x):
    x = tf.reshape(x, shape=[-1, size, size, NoSlices, 1])
    convolution1 = tf.nn.relu(
        convolution3d(x, tf.Variable(tf.random_normal([3, 3, 3, 1, 32]))) + tf.Variable(tf.random_normal([32])))
    convolution1 = maxpooling3d(convolution1)
    convolution2 = tf.nn.relu(
        convolution3d(convolution1, tf.Variable(tf.random_normal([3, 3, 3, 32, 64]))) + tf.Variable(
            tf.random_normal([64])))
    convolution2 = maxpooling3d(convolution2)
    convolution3 = tf.nn.relu(
        convolution3d(convolution2, tf.Variable(tf.random_normal([3, 3, 3, 64, 128]))) + tf.Variable(
            tf.random_normal([128])))
    convolution3 = maxpooling3d(convolution3)
    convolution4 = tf.nn.relu(
        convolution3d(convolution3, tf.Variable(tf.random_normal([3, 3, 3, 128, 256]))) + tf.Variable(
            tf.random_normal([256])))
    convolution4 = maxpooling3d(convolution4)
    convolution5 = tf.nn.relu(
        convolution3d(convolution4, tf.Variable(tf.random_normal([3, 3, 3, 256, 512]))) + tf.Variable(
            tf.random_normal([512])))
    convolution5 = maxpooling3d(convolution4)
    fullyconnected = tf.reshape(convolution5, [-1, 1024])
    fullyconnected = tf.nn.relu(
        tf.matmul(fullyconnected, tf.Variable(tf.random_normal([1024, 1024]))) + tf.Variable(tf.random_normal([1024])))
    fullyconnected = tf.nn.dropout(fullyconnected, keep_rate)
    output = tf.matmul(fullyconnected, tf.Variable(tf.random_normal([1024, 2]))) + tf.Variable(tf.random_normal([2]))
    return output


def network(x):
    prediction = cnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    epochs = 1000
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for data in trainingData:
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = session.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                except Exception as e:
                    pass

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
           # if tf.argmax(prediction, 1) == 0:
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
            # print('Correct:',correct.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
            print('Accuracy:', accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}))
        print('Final Accuracy:', accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}))
        '''patients = []
        actual = []
        predicted = []

        finalprediction = tf.argmax(prediction, 1)
        actualprediction = tf.argmax(y, 1)
        for i in range(len(validationData)):
            patients.append(validationData[i][2])
        for i in finalprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}):
            if(i==1):
                predicted.append("Cancer")
            else:
                predicted.append("No Cancer")
        for i in actualprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}):
            if(i==1):
                actual.append("Cancer")
            else:
                actual.append("No Cancer")
        for i in range(len(patients)):
            print("Patient: ",patients[i])
            print("Actual: ", actual[i])
            print("Predcited: ", predicted[i])

        from sklearn.metrics import confusion_matrix
        y_actual = pd.Series(
            (actualprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})),
            name='Actual')
        y_predicted = pd.Series(
            (finalprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})),
            name='Predicted')
        df_confusion = pd.crosstab(y_actual, y_predicted)
        print(df_confusion)

        ## Function to plot confusion matrix
        def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
            
            plt.matshow(df_confusion, cmap=cmap)  # imshow  
            # plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(df_confusion.columns))
            plt.xticks(tick_marks, df_confusion.columns, rotation=45)
            plt.yticks(tick_marks, df_confusion.index)
            # plt.tight_layout()
            plt.ylabel(df_confusion.index.name)
            plt.xlabel(df_confusion.columns.name)
            plt.show()
        plot_confusion_matrix(df_confusion)'''
        # print(y_true,y_pred)
        # print(confusion_matrix(y_true, y_pred))
        # print(actualprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
        # print(finalprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
network(x)
