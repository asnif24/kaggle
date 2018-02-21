import tensorflow as tf
import numpy as np
import pandas as pd

file_path = "./data/"
batch_size = 100
# train_df = pd.read_csv(data_path+'train.csv')
# test_df = pd.read_csv(data_path+'test.csv')

class DataLoader(object):
    def __init__(self):
        super(DataLoader, self).__init__()
    
    def readData(self, file_path):
        train_df = pd.read_csv(file_path+"train.csv")
        test_df = pd.read_csv(file_path+"test.csv")
        self.train_data = train_df[train_df.columns.values[1:]].values
        self.train_label = self.toOneHot(train_df["label"].values)
        self.test_data = test_df.values
    
    def toOneHot(self, labels):
        onehot_label = []
        for label in labels:
            tmp = np.zeros(10)
            tmp[label] = 1
            onehot_label.append(tmp)
        return np.asarray(onehot_label)
    
    def getTrainData(self):
        return self.train_data
    
    def getTrainLabel(self):
        return self.train_label
    
    def getTestData(self):
        return self.test_data


class  VariableDefiner(object):
    def __init__(self):
        super(VariableDefiner, self).__init__()
    def weight_variable(self, shape):
        initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.random_normal(shape, mean=0.0, stddev=0.1)
        return tf.Variable(initial)

class Layer(VariableDefiner):
    def __init__(self, in_size, out_size):
        super(Layer, self).__init__()
        self.W = self.weight_variable([in_size, out_size])
        self.b = self.weight_variable([1, out_size])
        # self.W = tf.Variable(tf.random_normal([in_size, out_size], mean=0.0, stddev=0.1))
        # self.b = tf.Variable(tf.random_normal([1, out_size], mean=0.0, stddev=0.1))
    def output(self, inputs, activation_function=None):
        if activation_function == None:
            return tf.matmul(inputs, self.W) + self.b
        else :
            return activation_function(tf.matmul(inputs, self.W) + self.b)
    def getVariables(self):
        return [self.W, self.b]

class CNN(object):
    def __init__(self):
        super(CNN, self).__init__()
        dataloader = DataLoader()
        dataloader.readData(file_path)
        self.batch_size = batch_size
        self.train_data = dataloader.getTrainData()
        self.train_label = dataloader.getTrainLabel()
        self.test_data = dataloader.getTestData()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_data_label, logits=layer3.output()))
        self.train_step = tf.train.AdamOptimizer(0.003).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(layer3.output(), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.x_data = tf.placeholder(tf.float32, shape = [None, pic_dim])
        self.x_label = tf.placeholder(tf.float32, shape = [None, 10])

    def showStatus(self):
        print "step: "+str(self.step)+", loss:"+str(sess.run(loss, feed_dict={self.x_data: self.train_data, self.x_label: y_train}))+", accuracy:"+

    def printParameters(self):
        print "batch_size: ", batch_size, ", z_dim: ", z_dim, ", hidden_dim: ", hidden_dim, ", gamma: ",gamma

    def randomBatch(self, _batch_size):
        return (np.random.sample(_batch_size)*len(self.x_label)).astype(int)

    def train(self, training_steps):
        self.printParameters()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        for self.step in range(training_steps):
            randomBatch = self.randomBatch(self.batch_size)
            self.sess.run(self.train_step, feed_dict={self.x_data: self.train_data[randomBatch], self.x_label: self.train_label[randomBatch]})




