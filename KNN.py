import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

def main():
    st.title("K Nearest Neighbors Classifier")

    # Load the iris dataset
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_labels = iris.target

    # Display the iris dataset
    st.write("Iris Dataset:")
    st.write("Features:")
    st.write(iris_data)
    st.write("Labels:")
    st.write(iris_labels)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.30)

    # Create and train the K Nearest Neighbors classifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)

    # Make predictions
    y_pred = classifier.predict(x_test)

    # Display confusion matrix and accuracy metrics
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Accuracy Metrics:")
    st.write(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
