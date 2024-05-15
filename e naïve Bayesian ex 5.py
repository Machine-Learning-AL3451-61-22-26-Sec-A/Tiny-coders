import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def main():
    st.title("Naive Bayes Classifier")

    # File uploader to load the CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Load the dataset
        msg = pd.read_csv(uploaded_file, names=['message', 'label'])
        st.write('The dimensions of the dataset:', msg.shape)
        
        # Mapping labels to numerical values
        msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
        X = msg.message
        y = msg.labelnum

        # Splitting the dataset into train and test data
        split_ratio = st.slider('Select split ratio for training set', 0.1, 0.9, 0.67)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=(1 - split_ratio))
        st.write('Train set:', xtrain.shape, 'Test set:', xtest.shape)

        # Using CountVectorizer to convert text into feature vectors
        count_vect = CountVectorizer()
        xtrain_dtm = count_vect.fit_transform(xtrain)
        xtest_dtm = count_vect.transform(xtest)

        # Creating a DataFrame for tabular representation of feature vectors
        df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names())
        st.write('Feature vector (training data):')
        st.write(df)

        # Training Naive Bayes (NB) classifier on training data
        clf = MultinomialNB().fit(xtrain_dtm, ytrain)
        predicted = clf.predict(xtest_dtm)

        # Printing accuracy metrics
        st.write('Accuracy metrics')
        st.write('Accuracy of the classifier is', metrics.accuracy_score(ytest, predicted))
        st.write('Confusion matrix')
        st.write(metrics.confusion_matrix(ytest, predicted))
        st.write('Recall and Precision')
        st.write('Recall:', metrics.recall_score(ytest, predicted))
        st.write('Precision:', metrics.precision_score(ytest, predicted))

        # New data prediction example
        docs_new = st.text_area("Enter new text data (one sentence per line):").split("\n")
        if st.button('Predict'):
            X_new_counts = count_vect.transform(docs_new)
            predicted_new = clf.predict(X_new_counts)
            for doc, category in zip(docs_new, predicted_new):
                st.write(f'{doc} => {"pos" if category == 1 else "neg"}')

if __name__ == "__main__":
    main()
