#!/usr/bin/env python
# coding: utf-8

# In[13]:


#QUESTION 1
# Problem (a) 
P_hostel = 0.60
P_day_scholar = 0.40
P_A_given_hostel = 0.30
P_A_given_day_scholar = 0.20

P_A = (P_A_given_hostel * P_hostel) + (P_A_given_day_scholar * P_day_scholar)
P_hostel_given_A = (P_A_given_hostel * P_hostel) / P_A

print(f"Probability that the student is a hostel resident given that they scored an A grade: {P_hostel_given_A:.4f}")

# Problem(b)
P_D = 0.01  
P_T_given_D = 0.99 
P_T_given_not_D = 0.02 
P_not_D = 1 - P_D
P_T = (P_T_given_D * P_D) + (P_T_given_not_D * P_not_D)
P_D_given_T = (P_T_given_D * P_D) / P_T
print(f"Probability of having the disease given a positive test result: {P_D_given_T:.4f}")


# In[14]:


#QUESTION 2
import pandas as pd

def create_sample_csv(filepath):
    data = {
        'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
        'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
        'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
        'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
        'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
create_sample_csv('buyer_data.csv')



# In[15]:


import pandas as pd
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_prob = {} 
        self.feature_prob = {}
        self.features = []
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prob = y.value_counts(normalize=True).to_dict()
        self.features = X.columns.tolist()
        self.feature_prob = {}
        for feature in self.features:
            self.feature_prob[feature] = {}
            for cls in self.classes:
                feature_values = X[y == cls][feature]
                value_counts = feature_values.value_counts(normalize=True).to_dict()
                self.feature_prob[feature][cls] = value_counts
        
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            post_probs = {}
            for cls in self.classes:
                prior = self.class_prob[cls]
                likelihood = 1
                for feature in self.features:
                    value = row[feature]
                    if value in self.feature_prob[feature][cls]:
                        likelihood *= self.feature_prob[feature][cls][value]
                    else:
                        likelihood *= 1e-10  
                post_probs[cls] = prior * likelihood
            predicted_class = max(post_probs, key=post_probs.get)
            predictions.append(predicted_class)
        return predictions

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('buys_computer', axis=1)
    y = df['buys_computer']
    return X, y

def main():
    create_sample_csv('buyer_data.csv')
    X, y = load_data('buyer_data.csv')
    nb = NaiveBayesClassifier()
    nb.fit(X, y)
    test_data = pd.DataFrame({
        'age': ['<=30', '31…40', '>40'],
        'income': ['high', 'medium', 'low'],
        'student': ['no', 'yes', 'no'],
        'credit_rating': ['fair', 'excellent', 'fair']
    })
    predictions = nb.predict(test_data)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()


# In[16]:


#QUESTION 3
import pandas as pd

def create_sample_csv(filepath):
    data = {
        'text': [
            'A great game',
            'The election was over',
            'Very clean match',
            'A clean but forgettable game',
            'It was a close election'
        ],
        'tag': [
            'Sports',
            'Not sports',
            'Sports',
            'Sports',
            'Not sports'
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

create_sample_csv('text_data.csv')


# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NaiveBayesTextClassifier:
    def __init__(self):
        self.class_prob = {}
        self.feature_prob = {}
        self.vocabulary = set()
        self.classes = []

    def preprocess_text(self, text):
        return text.lower().split()

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prob = y.value_counts(normalize=True).to_dict()
        self.feature_prob = {cls: {} for cls in self.classes}
        vocab = set()
        for cls in self.classes:
            texts = X[y == cls]
            all_words = ' '.join(texts).split()
            vocab.update(all_words)
            word_counts = pd.Series(all_words).value_counts()
            total_words = len(all_words)
            self.feature_prob[cls] = {word: (count + 1) / (total_words + len(vocab)) for word, count in word_counts.items()}

        self.vocabulary = vocab

    def predict(self, X):
        predictions = []
        for text in X:
            words = set(self.preprocess_text(text))
            post_probs = {}
            for cls in self.classes:
                prior = self.class_prob[cls]
                likelihood = 1
                for word in words:
                    likelihood *= self.feature_prob[cls].get(word, 1 / (len(self.vocabulary) + 1))
                post_probs[cls] = prior * likelihood
            predicted_class = max(post_probs, key=post_probs.get)
            predictions.append(predicted_class)
        return predictions

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df['text']
    y = df['tag']
    return X, y

def evaluate_model(y_true, y_pred):
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    true_positive = np.sum((np.array(y_true) == 'Sports') & (np.array(y_pred) == 'Sports'))
    false_positive = np.sum((np.array(y_true) != 'Sports') & (np.array(y_pred) == 'Sports'))
    false_negative = np.sum((np.array(y_true) == 'Sports') & (np.array(y_pred) != 'Sports'))
    true_negative = np.sum((np.array(y_true) != 'Sports') & (np.array(y_pred) != 'Sports'))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    return accuracy, precision, recall

def main():
    create_sample_csv('text_data.csv')
    X, y = load_data('text_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    nb = NaiveBayesTextClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy, precision, recall = evaluate_model(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    new_sentence = "A very close game"
    prediction = nb.predict([new_sentence])[0]
    print(f"The sentence '{new_sentence}' is classified as '{prediction}'")

if __name__ == "__main__":
    main()


# In[ ]:




