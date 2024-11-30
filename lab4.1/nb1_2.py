import pandas as pd
import numpy as np
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
from argparse import ArgumentParser
import pickle
import time

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess the text (lowercasing, stopword removal, and stemming)
count = 0
def preprocess(text, stop_words):
    # Lowercase the text
    global count
    text = text.lower()
    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    ans = " ".join(filtered_words)
    count+=1
    return ans

# Naive Bayes classifier for Bernoulli and Multinomial models
class NaiveBayesClassifier:
    def __init__(self, model_type='multinomial'):
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.class_probs = defaultdict(float)
        self.vocab = set()
        self.model_type = model_type
        self.class_labels = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        self.class_counter = defaultdict(int)
        self.class_word_counter = defaultdict(int)

    def train(self, texts, labels):
        num_docs = len(labels)
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))

        # Count words and classes
        for i, text in enumerate(texts):
            label = labels[i]
            class_counts[label] += 1
            words = text.split()
            self.vocab.update(words)
            if self.model_type == 'bernoulli':
                words = set(words)  # Bernoulli model considers presence/absence of words
            for word in words:
                word_counts[label][word] += 1

        self.class_counter = class_counts

        # Compute class probabilities (log scale to avoid underflow)
        for label in class_counts:
            self.class_probs[label] = np.log(class_counts[label] / num_docs)
            if self.model_type == 'multinomial':
                self.class_word_counter[label] = sum(word_counts[label].values())
            
        
        # Compute word probabilities with Laplace smoothing
        vocab_size = len(self.vocab)
        for word in self.vocab:
            for label in class_counts:
                if self.model_type == 'bernoulli':
                    word_prob = (word_counts[label][word] + 1) / (class_counts[label] + 2)
                else:  # Multinomial model
                    word_prob = (word_counts[label][word] + 1) / (self.class_word_counter[label] + vocab_size)
                self.word_probs[label][word] = np.log(word_prob)

    def save_parameters(self, output_file):
        # Save parameters (class probabilities and word probabilities) in the specified format
        with open(output_file, 'w') as f:
            for label in self.class_labels:
                f.write(f"Class: {label}\n")
                # Sort the words for the current label
                sorted_words = sorted(self.word_probs[label].items())
                for word, log_prob in sorted_words:
                    f.write(f"{word}: {log_prob}\n")
                f.write("\n")
        print(f"Parameters saved to {output_file}")

    def predict(self, text):
        words = set(text.split()) if self.model_type == 'bernoulli' else text.split()
        class_scores = np.zeros(len(self.class_labels))

        count=0
        for word in words:
            if word not in self.vocab:
                count+=1
        if count!=0 and self.model_type == 'bernoulli':
            count=1
        for idx, label in enumerate(self.class_labels):
            score = self.class_probs[label]
            if self.model_type == 'multinomial':
                for word in words:
                    if word in self.vocab:
                        score += self.word_probs[label][word]
                    else:
                        score += np.log(1 / (len(self.vocab) + self.class_word_counter[label]))
            else:
                for word in self.vocab:
                    if word in words:
                        score += self.word_probs[label][word]
                    elif self.model_type == 'bernoulli':  # For Bernoulli, consider absence of word
                        score += np.log(1 - np.exp(self.word_probs[label][word]))
                score += count*np.log(1 / (2 + self.class_counter[label]))
            class_scores[idx] = score

        return class_scores

    def save_probabilities(self, test_texts, output_file):
        with open(output_file, 'w') as f:
            for i, text in enumerate(test_texts):
                probabilities = self.predict(text)
                f.write(f"Instance {i}:\n")
                for j, prob in enumerate(probabilities):
                    f.write(f"  Class: {self.class_labels[j]}, Probability: {prob}\n")
                f.write("\n")
        print(f"Predicted probabilities saved to {output_file}")

# Argument parser to handle command line inputs
parser = ArgumentParser()
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--stop", type=str, required=True)
parser.add_argument("--model", type=str, required=False, default="multinomial", help="Choose either 'bernoulli' or 'multinomial'")
parser.add_argument("--params_out", type=str, required=False, default="model_params.txt", help="File to save model parameters")
parser.add_argument("--probas_out", type=str, required=False, default="predicted_probabilities.txt", help="File to save predicted probabilities")
args = parser.parse_args()

t1 = time.time()

# Load stopwords from file
with open(args.stop, 'r') as f:
    stop_words = f.read().splitlines()

# Load training data from TSV
train_df = pd.read_csv(args.train, sep="\t", header=None, quoting=3)
train_df['processed_text'] = train_df.iloc[:, 2].apply(lambda x: preprocess(x, stop_words))

# Train the Naive Bayes model
texts = train_df['processed_text'].tolist()
labels = train_df.iloc[:, 1].tolist()  # Labels are in the 2nd column
nb = NaiveBayesClassifier(model_type=args.model)
nb.train(texts, labels)

# Save the model parameters to a file for debugging
nb.save_parameters(args.params_out)

# Load and preprocess the test data from TSV
test_df = pd.read_csv(args.test, sep='\t', header=None, quoting=3)
test_df['processed_text'] = test_df.iloc[:, 2].apply(lambda x: preprocess(x, stop_words))
print(count)

# Save predicted probabilities
nb.save_probabilities(test_df['processed_text'].tolist(), args.probas_out)

# Predict labels for the test data
predictions = [nb.class_labels[np.argmax(nb.predict(text))] for text in test_df['processed_text'].tolist()]

# Write the predictions to the output file
with open(args.out, 'w') as f:
    for i, prediction in enumerate(predictions):
        if i < len(predictions) - 1:
            f.write(prediction + '\n')
        else:
            f.write(prediction)
            

t2 = time.time()
print("total time: ",t2-t1)


