import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pyflux as pf

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from AlgorithmicTrading.DataHandler import DataHandler


def linearRegressionModel(data_frame):
    X = data_frame['Seconds'].values.reshape(-1, 1)
    y = data_frame['Close'].values

    # use train/test split with 80% of the data for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # fit the model to the training data
    logreg = LinearRegression()
    logreg.fit(X_train, y_train)
    logreg.fit(X_train, y_train)

    # make predictions on the testing data
    y_pred = logreg.predict(X_train)

    analyzeModel(y_pred, X_test, X_train, y_test, y_train)


def analyzeModel(y_pred, X_test, X_train, y_test, y_train):
    # calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print the evaluation metrics
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('R-squared:', r2)
    # create a Matplotlib figure and axis
    fig, ax = plt.subplots()
    # plot the testing data and the model's predictions
    ax.plot(X_train, y_train, 'g', label='Training Data')
    ax.plot(X_test, y_test, 'b', label='Testing Data')
    ax.plot(X_test, y_pred, 'r', label='Predictions')
    # add a legend
    ax.legend()
    ax.set_ylim(bottom=1.375, top=1.475)
    # show the plot
    plt.show()


def polynomialRegression(data_frame):
    X = data_frame['Seconds'].values.reshape(-1, 1)
    y = data_frame['Close'].values

    # use train/test split with 80% of the data for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    # Create a polynomial features transformer
    poly_transform = PolynomialFeatures(degree=2)

    # Transform the training data using the transformer
    X_train_poly = poly_transform.fit_transform(X_train)

    # Create a linear regression model
    poly_reg = LinearRegression()

    # Fit the model to the transformed training data
    poly_reg.fit(X_train_poly, y_train)

    # Transform the testing data using the transformer
    X_test_poly = poly_transform.transform(X_test)

    # Make predictions on the transformed testing data
    y_pred = poly_reg.predict(X_test_poly)

    analyzeModel(y_pred, X_test, X_train, y_test, y_train)

def fractalRegression(data_frame):
    # Create the model
    model = pf.FractalReg(data=data_frame, formula='Close~1', ar=4, ma=4)

    # Fit the model
    model.fit()

    # Make predictions using the model
    predictions = model.predict(h=len(X_test))



if __name__ == '__main__':
    #use datahandler to read in csv file
    dataHandler = DataHandler('../TradingAI/EURCAD.csv')
    data = dataHandler.read_data()

    data = dataHandler.format_tickstory_data(data)
    print(data.head())

    linearRegressionModel(data)
    polynomialRegression(data)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
