import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import time

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


class VariableDefiner(object):
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


class Convolution(object):
    def __init__(self):
        super(Convolution, self).__init__()
    def conv2d(self, x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding = 'SAME')
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class CNN(Convolution, VariableDefiner):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_var_W = {
            "conv1" : self.weight_variable([3,3, 1,24]),
            "conv2" : self.weight_variable([3,3,24,24]),
            "conv3" : self.weight_variable([3,3,24,48]),
            "conv4" : self.weight_variable([3,3,48,48]),
            "conv5" : self.weight_variable([3,3,48,72]),
            "conv6" : self.weight_variable([3,3,72,72])
        }
        self.conv_var_b = {
            "conv1" : self.bias_variable([24]),   
            "conv2" : self.bias_variable([24]),
            "conv3" : self.bias_variable([48]),
            "conv4" : self.bias_variable([48]),   
            "conv5" : self.bias_variable([72]),
            "conv6" : self.bias_variable([72])
        }
        self.layer_var = {
            "layer1" : Layer(4*4*72, 128),
            "layer2" : Layer(128, 10)
        }
        self.keep_prob = tf.placeholder(tf.float32)
    def output(self, x):
        x_origin = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(tf.add(self.conv2d(x_origin, self.conv_var_W["conv1"], 1), self.conv_var_b["conv1"]))     
        h_conv2 = tf.nn.relu(tf.add(self.conv2d(h_conv1, self.conv_var_W["conv2"], 1), self.conv_var_b["conv2"]))    
        h_max_pool1 = self.max_pool_2x2(h_conv2)
        # print h_max_pool1.shape

        h_conv3 = tf.nn.relu(tf.add(self.conv2d(h_max_pool1, self.conv_var_W["conv3"], 1), self.conv_var_b["conv3"]))    
        h_conv4 = tf.nn.relu(tf.add(self.conv2d(h_conv3, self.conv_var_W["conv4"], 1), self.conv_var_b["conv4"]))    
        h_max_pool2 = self.max_pool_2x2(h_conv4)
        # print h_max_pool2.shape

        h_conv5 = tf.nn.relu(tf.add(self.conv2d(h_max_pool2, self.conv_var_W["conv5"], 1), self.conv_var_b["conv5"]))    
        h_conv6 = tf.nn.relu(tf.add(self.conv2d(h_conv5, self.conv_var_W["conv6"], 1), self.conv_var_b["conv6"]))    
        h_max_pool3 = self.max_pool_2x2(h_conv6)
        # print h_max_pool3.shape

        h_conv6_reshape = tf.reshape(h_max_pool3, [-1,4*4*72])
        h_layer1 = self.layer_var["layer1"].output(h_conv6_reshape, tf.nn.sigmoid)
        h_drop = tf.nn.dropout(h_layer1, self.keep_prob)
        h_layer2 = self.layer_var["layer2"].output(h_drop, tf.nn.softmax)
        return h_layer2

class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        dataloader = DataLoader()
        dataloader.readData(file_path)
        self.batch_size = batch_size
        self.train_data = dataloader.getTrainData()
        self.train_label = dataloader.getTrainLabel()
        self.test_data = dataloader.getTestData()

        self.x_data = tf.placeholder(tf.float32, shape = [None, 784])
        self.x_label = tf.placeholder(tf.float32, shape = [None, 10])

        self.cnn = CNN()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.x_label, logits=self.cnn.output(self.x_data)))
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.x_label, 1), tf.argmax(self.cnn.output(self.x_data), 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        

    def showStatus(self):
        randomBatch = self.randomBatch(10000)
        loss, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict={self.x_data: self.train_data[randomBatch], self.x_label: self.train_label[randomBatch], self.cnn.keep_prob: 1.0})
        print "step: "+str(self.step)+", loss:"+str(loss)+", accuracy:"+str(accuracy)

    def printParameters(self):
        print "batch_size: ", batch_size
        # print "batch_size: ", batch_size, ", z_dim: ", z_dim, ", hidden_dim: ", hidden_dim, ", gamma: ",gamma

    def randomBatch(self, _batch_size):
        return (np.random.sample(_batch_size)*len(self.train_label)).astype(int)


    def outputCsv(self, data):
        filename = "./output/"+time.strftime("%m%d%H%M", time.localtime())+".csv"
        with open(filename, "wb") as f:
            writer = csv.writer(f)
            writer.writerow(["ImageId","Label"])
            for index, value in enumerate(data):
                writer.writerow([index+1, value])

    def train(self, training_steps):
        self.printParameters()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        for self.step in range(training_steps):
            randomBatch = self.randomBatch(self.batch_size)
            self.sess.run(self.train_step, feed_dict={self.x_data: self.train_data[randomBatch], self.x_label: self.train_label[randomBatch], self.cnn.keep_prob: 0.6})
            if self.step%500 == 0:
                self.showStatus()

    def test(self):
        predict = self.sess.run(tf.argmax(self.cnn.output(self.x_data), 1), feed_dict={self.x_data: self.test_data, self.cnn.keep_prob: 1.0})
        self.outputCsv(predict)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(50001)
    trainer.test()

