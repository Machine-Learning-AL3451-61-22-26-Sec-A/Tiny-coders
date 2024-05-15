import streamlit as st
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei * ymat.T))
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

def main():
    st.title("Local Weighted Regression")

    # Load data points
    data = pd.read_csv('10data.csv')
    bill = np.array(data.total_bill)
    tip = np.array(data.tip)

    # Preparing and adding 1 in bill
    mbill = np.mat(bill)
    mtip = np.mat(tip)
    m = np.shape(mbill)[1]
    one = np.mat(np.ones(m))
    X = np.hstack((one.T, mbill.T))

    # Set k here
    k_value = st.slider("Select k value:", min_value=1, max_value=10, value=2)

    # Local Weight Regression
    ypred = localWeightRegression(X, mtip, k_value)
    SortIndex = X[:, 1].argsort(0)
    xsort = X[SortIndex][:, 0]

    # Plotting
    st.write("Plot:")
    st.pyplot(plt.scatter(bill, tip, color='green'))
    st.pyplot(plt.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5))
    plt.xlabel('Total bill')
    plt.ylabel('Tip')

if __name__ == "__main__":
    main()
