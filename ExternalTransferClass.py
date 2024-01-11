import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Lambda
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import csv



#sample lstm code for published paper:  https://github.com/radioML/dataset
# Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors

testData = 'testData'

rows = 343  #total # of rows for dataset

def get_training_data():
    df = pd.read_csv("source_data.csv", header=0)
    df = df.drop_duplicates()
    df2 = pd.read_csv("trainingLabels.csv", usecols = ['IDname'])  #Identifier


    df3 = pd.merge(df, df2, on= "IDname")



    df3.to_csv('TrainingData/fullTraining'+testData+'.csv')


#This library fills in 0s for missing time steps.  Many datasets don't begin/end at the same timestep, and equal time
# steps is important for LSTM performance.
def preprocess_fillZero():
    # define date parser
    date_parser = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M")
    df3 = pd.read_csv('TrainingData/fullTraining'+testData+'.csv', header=0, parse_dates=['postdate'], date_parser=date_parser)
    print("Orig: ", df3.shape)

    del df3[df3.columns[0]]
    df3 = df3.drop_duplicates()
    print(df3.shape)
    print(df3.head())
    print(df3['postdate'].head())

    all_dates=pd.date_range(start='11-02-2022', end='05-15-2023')

    #Testing date field works
    print(all_dates[0] == df3['postdate'].iloc[0])

    #Set the original index, grouped by companyName and postdate
    df3.set_index(['IDName','postdate'], inplace = True)


    # Run the below to remap to all dates
    df_final = pd.concat([df3.loc[company].reindex(all_dates) for company in df3.index.get_level_values('IDName').unique()],
                         keys=df3.index.get_level_values('IDName').unique(), names=['IDName', 'pdate'])
    print(df_final.head(20))

    df_final.fillna(0, inplace=True)
    df_final.to_csv('training'+testData+'.csv')

def normalize():

    scaler = MinMaxScaler()
    encoder = LabelEncoder()

    #normalize
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    df_final = pd.read_csv('training'+testData+'.csv', header=0, parse_dates=['postdate'], date_parser=date_parser)
    # df_final['postdate'] = pd.to_datetime(['postdate'], format = "%m/%d/%Y %H:%M")
    df_final['year'] = df_final['pdate'].dt.year
    df_final['month'] = df_final['pdate'].dt.month
    df_final['monthstart'] = df_final['pdate'].dt.is_month_start*1
    df_final['monthend'] = df_final['pdate'].dt.is_month_end*1

    # shuffling the entire dataframe
    df_final = df_final.sample(frac = 1)



    Y = df_final[['IDName','class']].copy()
    Y = Y[Y['class'] != '0']
    Y = Y.drop_duplicates()

    #Making Y class binary
    Y = Y['class'].apply(lambda x: 1 if x == 'labelA' else 0)  #Example assumes first class is labeled 'labelA'
    print(Y.head(5))
    print("Y: ", Y.size)



    print(list(df_final.columns))


    X = df_final[['IDName', 'TransAMT', 'MinTransAMT', 'MaxTransAMT',
                  'transCt', 'StdTransAMT', 'y','m', 'm_start', 'm_end']].copy()
    print("X: ",X.size)
    print(X[['IDName','y', 'm', 'm_start', 'm_end']].head(5))

    X['IDName'] = encoder.fit_transform(X['IDName'])

    X = X.to_numpy()
    Y = Y.to_numpy()

    # Assuming your numpy array is named 'data'.  To normalize, the float, categorical and integers data have to be
    #separated.
    categorical_column_index = [0,6,7,8,9]
    float_columns_indices = [1, 2, 3, 5]
    integer_column_index = 4

    float_columns = X[:, float_columns_indices]
    integer_column = X[:, integer_column_index]

    scaler = MinMaxScaler()
    scaled_float_columns = scaler.fit_transform(float_columns)
    scaled_integer_column = scaler.fit_transform(integer_column.reshape(-1,1))

    # Concatenate the columns back together
    # scaled_data = np.concatenate(
    #     (X[:, categorical_column_index].reshape(-1, 1), scaled_float_columns, scaled_integer_column), axis=1)
    scaled_data = np.concatenate(
        (X[:, categorical_column_index], scaled_float_columns, scaled_integer_column), axis=1)
    # Reshape X to (200, 1, 38) for LSTM input
    X = X.reshape((rows, 195, 10))  # ((numSamples, numTimeSteps, numFeatures, use 7 when including date))
    # change numSamples
    np.save('X_'+testData+'.npy', X)
    np.save('Y_'+testData+'.npy', Y)



    return X, Y

# def inverse_sigmoid(x):
#     return -tf.math.log((1/x)-1)


def basicLSTM():  #sample code
    np.random.seed(240)
    train_data = 'training_data'



    train_acc = []
    valid_acc = []
    test_acc = []
    AUC = []
    PREC = []
    RECALL = []
    F1 = []





    X = np.load('X_'+testData+'.npy', allow_pickle=True)
    Y = np.load('Y_'+testData+'.npy', allow_pickle=True)




    #Convert numpy to tensor

    # If doing a full split
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=50)

    # Try implementing stratefied Kfolds
    skf = StratifiedKFold()
    skf.get_n_splits(X, Y, 5)
    # a = [10, 12, 14, 16, 18, 20]
    # b = [22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    # c = [.1, .2, .3, .4, .5]
    # d = [6, 8, 10, 12, 14]
    # epoch = [30, 35, 40, 45, 50]
    # batch = [5,6,7,8,9]

#implementing grid search, could alternatively use keras Tuner or another library
    a = [14]
    b = [22]
    c = [.3]
    d = [13]
    batch = [8]
    epoch = [20]

    fNeur = []
    sNeur = []
    dropR = []
    dNeur = []
    ep = []
    bSize = []
    run = []

    runCount = 0
    class_weight = {0: 1, 1: 7 }  #attempts Cross Entropy weight manipulation to deal with imbalanced data

    for i in a:
        for j in b:
            for k in c:
                for l in d:
                    for m in epoch:
                        for n in batch:
                            runCount += 1
                            StratifiedKFold(n_splits =5, random_state = 250, shuffle = True)
                            for train_index, test_index in skf.split(X,Y):

                                X_train, X_test = X[train_index], X[test_index]
                                Y_train, Y_test = Y[train_index], Y[test_index]
                                # X_test = np.append(X_test, X_addTest, axis=0)
                                # Y_test = np.append(Y_test, Y_addTest, axis = 0)


                                # Setting up shuffling X and Y test in unison; obtain row indices by permutating first dimension
                                # indices = np.random.permutation(X_test.shape[0])
                                # idx = np.random.permutation(len(X_test))
                                # # Shuffle X_test and Y_test in unison based on index found above
                                # X_test = X_test[idx]
                                # Y_test = Y_test[idx]


                                print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
                                print(X_train[0:5, :])
                                # print(Y_train[0:5])

                                print(X_train.dtype)
                                print(Y_train.dtype)
                                print(Y_train)
                                print(Y_test)

                                X_train = np.asarray(X_train).astype('float32')
                                X_train = tf.convert_to_tensor(X_train)
                                Y_train = tf.convert_to_tensor(Y_train)
                                X_test = tf.convert_to_tensor(X_test)
                                Y_test = tf.convert_to_tensor(Y_test)


                                model = Sequential()

                                model.add(LSTM(i, return_sequences=True, input_shape=(195, 10),
                                               kernel_initializer='uniform', recurrent_initializer='uniform', bias_initializer='zeros'))  #input shape:  numSteps, numFeatures
                                # uncomment the line above to initialize all weights uniformly
                                model.add(BatchNormalization())
                                model.add(Dropout(k))
                                # model.add(Dense(18))
                                # model.add(LSTM(100, return_sequences=True))

                                #
                                # model.add(Dense(20, activation='tanh'))

                                #use the below with another softmax layer
                                # model.add(Dense(1, activation=lambda x: K.log(x/(1-x)), use_bias=True))
                                model.add(LSTM(j))
                                # model.add(LSTM(34))
                                model.add(BatchNormalization())
                                # model.add(Dense(18, activation='relu'))
                                model.add(Dense(l, activation='tanh', kernel_regularizer='l2'))


                                # model.add(Lambda(lambda x: 1 - x))
                                model.add(Dense(1, activation='sigmoid'))
                                # model.add(Dense(2, activation = 'softmax'))
                                # model.add(Lambda(lambda x: 1 - x))


                                #adding in adams learning rate
                                opt = keras.optimizers.Adam(learning_rate=.001)

                                model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'] )
                                # model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])
                                h = model.fit(X_train, Y_train, epochs=m, batch_size=n, validation_split=.2, verbose=2, class_weight=class_weight)
                                train_acc.append(h.history['accuracy'])
                                valid_acc.append(h.history['val_accuracy'])
                                fNeur.append(i)
                                sNeur.append(j)
                                dropR.append(k)
                                dNeur.append(l)
                                ep.append(m)
                                bSize.append(n)
                                run.append(runCount)

                                #The below code evaluates on test data and gives predictions for a certain value
                                # history.history
                                #
                                # print("Evaluate model on test data")
                                results = model.evaluate(X_test, Y_test, batch_size=n)
                                print("test loss, test acc:", results)
                                test_acc.append(results)

                                #Adding in AUC
                                y_pred_proba = model.predict(X_test)
                                fpr, tpr, _ = metrics.roc_curve(Y_test, y_pred_proba)
                                auc = metrics.roc_auc_score(Y_test, y_pred_proba)
                                print("AUC Score:", auc)
                                print(y_pred_proba)
                                y_pred_proba = np.transpose(y_pred_proba)[0]  # transformation to get (n,)
                                print(y_pred_proba.shape)  # now the shape is (n,)
                                # Applying transformation to get binary values predictions
                                #the below values probably aren't valuable, but hypertuning and observing this
                                #helped discover data inconsistencies.
                                y_pred_proba = list(map(lambda x: 0 if x < 0.222 else 1, y_pred_proba))
                                print(y_pred_proba)
                                AUC.append(auc)
                                precision = metrics.precision_score(Y_test, y_pred_proba)
                                PREC.append(precision)
                                reCal = metrics.recall_score(Y_test, y_pred_proba)
                                RECALL.append(reCal)
                                f1Score = metrics.f1_score(Y_test, y_pred_proba)
                                F1.append(f1Score)


                            print("Training Accuracy:", train_acc)
                            print("Validation Accuracy:", valid_acc)
                            print("Testing Accuracy:", test_acc)
                            scores = pd.DataFrame(
                                {'Training': train_acc, 'Validation': valid_acc, 'Testing': test_acc, 'Run': run,
                                 'fNeur': fNeur,
                                 'sNeur': sNeur, 'dropR': dropR, 'dNeur': dNeur, 'ePoch': ep, 'batchSize': bSize,
                                 'AUC': AUC, 'Precision': PREC, 'Recall': reCal, 'F1 Score':F1})
                            print(scores.head(5))
                            print("opening file and saving")
                            with open('Scores.csv', mode='a', newline= '') as file:
                                scores.to_csv(file, header=False)

    # putting all scores into a pandas dataframe




    # scores.to_csv('Scores.csv')
    #
    # # Generate a prediction using model.predict()
    # # and calculate it's shape:
    # print("Generate a prediction")
    # prediction = pd.DataFrame(model.predict(X_test))
    # prediction['real'] =
    # print("prediction shape:", prediction.shape)




get_training_data()  #run this to get full training data
preprocess_fillZero()
X, Y = normalize()
print(X.shape)
print(Y.shape)
print(Y)

basicLSTM()

