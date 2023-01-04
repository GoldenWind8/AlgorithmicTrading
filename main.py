import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from AlgorithmicTrading.DataHandler import DataHandler
import  tensorflow  as  tf


def linearRegressionModel(data_frame):
    X = data_frame['Seconds'].values.reshape(-1, 1)
    y = data_frame['Close'].values

    # use train/test split with 80% of the data for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # fit the model to the training data
    logreg = LinearRegression()
    logreg.fit(X_train, y_train)

    # make predictions on the testing data
    y_pred = logreg.predict(X_test)

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

def deepLearningModel(data_frame):
    # Split the data into training and testing sets
    train_data = data_frame.iloc[:int(len(data) * 0.8)]
    test_data = data_frame.iloc[int(len(data) * 0.8):]

    # Define the model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=(1,), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_data['Seconds'], train_data['Price'], epochs=50,
                        validation_data=(test_data['Seconds'], test_data['Price']))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data['Seconds'], test_data['Price'])
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Make predictions
    predictions = model.predict(test_data['Seconds'])




if __name__ == '__main__':
    #use datahandler to read in csv file
    dataHandler = DataHandler('Data/EURCAD.csv')
    data = dataHandler.read_data()

    data = dataHandler.format_tickstory_data(data)
    print(data.head())

    linearRegressionModel(data)
    polynomialRegression(data)

