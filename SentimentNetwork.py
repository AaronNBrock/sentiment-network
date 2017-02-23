import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        np.random.seed(1)

        # Pre process data
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        self.label_vocab = list(label_vocab)

        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = {}
        for i, label in enumerate(self.review_vocab):
            self.label2index[label] = i

        # Init network
        self.layer_0_nodes = len(self.review_vocab)
        self.layer_1_nodes = hidden_nodes
        self.layer_2_nodes = 1

        self.weights_0_1 = 2 * np.random.random((self.layer_0_nodes, self.layer_1_nodes)) - 1
        self.weights_1_2 = 2 * np.random.random((self.layer_1_nodes, self.layer_2_nodes)) - 1

        self.layer_0 = np.zeros((1, self.layer_0_nodes))
        self.layer_1 = np.zeros((1, self.layer_1_nodes))

        self.learning_rate = learning_rate

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        self.activation_output_derivative = lambda output: output * (1 - output)

    def update_input_layer(self, review):
        # Reset layer
        self.layer_0 *= 0
        for word in review.split(' '):
            self.layer_0[0][self.word2index[word]] = 1

    @staticmethod
    def get_target_for_label(label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0

    def train(self, training_reviews, training_labels):

        training_review_indices = list()
        for review in training_reviews:
            indices = set()
            for word in review.split(" "):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])
            training_review_indices.append(list(indices))

        assert(len(training_review_indices) == len(training_labels))

        correct_so_far = 0
        start = time.time()

        for i in range(len(training_review_indices)):
            review_indices = training_review_indices[i]
            label = training_labels[i]

            # Forward pass #
            self.layer_1 *= 0
            for index in review_indices:
                self.layer_1 += self.weights_0_1[index]

            layer_2 = self.activation_function(self.layer_1.dot(self.weights_1_2))

            # Backward pass #
            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_grad = layer_2_error * self.activation_output_derivative(layer_2)

            layer_1_error = layer_2_error.dot(self.weights_1_2.T)
            layer_1_grad = layer_1_error

            # Update weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_grad) * self.learning_rate
            for index in review_indices:
                self.weights_0_1[index] -= layer_1_grad[0] * self.learning_rate

            # Log
            if np.abs(layer_2_error) < 0.5:
                correct_so_far += 1
            reviews_per_second = i / float(time.time() - start)

            sys.stdout.write(
                "\rProgress:" + str(100 * i / float(len(training_review_indices)))[:4] + "% Speed(reviews/sec):" + str(
                    reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(
                    i + 1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
        print("\n")

    def predict(self, review):
        # Input Layer

        # Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])

        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]

        # Output layer
        layer_2 = self.activation_function(self.layer_1.dot(self.weights_1_2))

        if (layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"

if __name__ == "__main__":

    # Load data
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()

    # Init network
    network = SentimentNetwork(reviews, labels, learning_rate=0.1)

    network.train(reviews, labels)

    review = "This movie is pretty good but I think it would be better if they had more potatoes!  " \
             "However, I do rate a perfect 5/7"

    print(review)
    print(network.predict(review))