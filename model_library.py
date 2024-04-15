import numpy as np
import time
import os, shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import pydotplus, graphviz
import tensorflow as tf
import keras_tuner as kt
import keras
pd.set_option('chained_assignment',None)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from IPython.display import Image
from keras import layers

# This class is used to:
    # 1. Import the data used for modeling and prepare for use by splitting into train/test sets
    # 2. Create methods for each model used
        # a. Models include: Linear Regression, Random Forest, Decision Tree, and Neural Network
        # b. Options in each model to tune return plots exported as images in separate folder
        # c. Options to return the model itself for further use
        # d. Option to return scoring metrics for each model
        # e. Features can be selected for each model from the data
        # f. Option to change hyperparameters for each model if 'hyperparameters' is passed as an argument
        # g. Option to tune if params dictionary is passed as an argument
    # 3. If scoring metrics are returned, they are stored in a class pandas dataframe

#  Main References:
    # 1. https://www.tensorflow.org/tutorials/keras/regression
    # 2. Reference documentation for packages used in this class
    # 3. Data from: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset

class ModelLibrary:

    def __init__(self, file_path, response = 'RUL'):
        # write inputs
        self.response = response
        # scoring dataframe
        self.scores = pd.DataFrame(columns = ['Model', 'Features', 'Parameters', 'R2_train', 
                                              'MAE_train', 'R2_test', 'MAE_test',
                                              'query_time', 'train_time'
                                              ])
        # import data
        self.data = pd.read_csv(file_path)
        # drop first 20 cycles if battery id = B0005, B0006, B0007 
        self.data = self.data[(self.data['battery_id']=='B0018') |
                              (self.data['cycle'] > 20)]
        self.df_train, self.df_test = train_test_split(self.data, test_size = 0.25, random_state=30)
        self.X_train = self.df_train.drop(columns = [response])
        self.y_train = self.df_train[response]
        self.X_test = self.df_test.drop(columns = [response])
        self.y_test = self.df_test[response]
        # default hyperparameters for each model
        self.rf_hyperparameters = {'n_estimators': 50, 'max_depth': 5, 'random_state': 93}
        self.dt_hyperparameters = {'min_samples_split': 15, 'random_state': 30}
        self.dnn_tuned_hyperparameters = {'max_layers': 2}

    # helper function to plot predicted vs actual
    def plot_pred_vs_actual(self, y_test, y_pred):
        f1 = plt.figure(1)
        plt.scatter(y_test, y_pred)
        plt.plot([0, 160], [0, 160], color='red')
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')
        plt.title('Predicted vs Actual RUL')
        plt.show()
    
    # function to empty scores dataframe
    def clear_scores(self):
        self.scores = pd.DataFrame(columns = self.scores.columns)
    
    # function to write scores dataframe 
    def write_scores(self, name):
        self.scores.to_csv(name, index=False)
    
    # Linear Regression
    def lin_reg(self, features, plot=False, return_model=False, write_scores=False):
        # select data from test/train datasets
        X_train = self.X_train[features]
        X_test = self.X_test[features]
        
        # create model
        lin_reg = LinearRegression()
        r2_train = cross_val_score(lin_reg, X_train, self.y_train, cv=7, scoring='r2').mean()
        mae_train = cross_val_score(lin_reg, X_train, self.y_train, cv=7, scoring='neg_mean_absolute_error').mean()*-1
        
        # fit to all training data
        # time to train model
        start = time.time()
        lin_reg.fit(X_train, self.y_train)
        end = time.time()
        train_time = end - start
        y_pred = lin_reg.predict(X_test)
        r2_test = lin_reg.score(X_test, self.y_test)
        mae_test = mean_absolute_error(self.y_test, y_pred)
        print('Linear Regression Results: ')
        print(f'Training R2: {r2_train}')
        print(f'Test R2: {r2_test}')

        # time to query model
        start = time.time()
        for i in range(1000):
            row = X_test.sample()
            lin_reg.predict(row)
        end = time.time()
        query_time = (end - start) / 1000

        if plot:
            self.plot_pred_vs_actual(self.y_test, y_pred)
        
        if write_scores:
            self.scores.loc[len(self.scores.index)] = ['Linear Regression', features, None, r2_train, 
                                                       mae_train, r2_test, mae_test, query_time, train_time]
        
        if return_model:
            return lin_reg
    
    # Random Forest
    def rf(self, features, plot=False, return_model=False, write_scores=False, params=None):

        # select data from test/train datasets
        X_train = self.X_train[features]
        X_test = self.X_test[features]

        # train and fit simple or tuned model
        if not params:
            # create model
            rf = RandomForestRegressor(**self.rf_hyperparameters)
            r2_train = cross_val_score(rf, X_train, self.y_train, cv=7, scoring='r2').mean()
            mae_train = cross_val_score(rf, X_train, self.y_train, cv=7, scoring='neg_mean_absolute_error').mean()*-1
            # time to train
            start = time.time()
            rf.fit(X_train, self.y_train)
            end = time.time()
            train_time = end - start
            y_pred = rf.predict(X_test)
            r2_test = rf.score(X_test, self.y_test)
            mae_test = mean_absolute_error(self.y_test, y_pred)
            print('Random Forest Results: ')
            print(f'Training R2: {r2_train}')
            print(f'Train MAE: {mae_train}')
            print(f'Test R2: {r2_test}')
            print(f'Test MAE: {mae_test}')
            hyperparameters_used = self.rf_hyperparameters

        else:
            rf_tuned = GridSearchCV(estimator=RandomForestRegressor(random_state=30), param_grid=params, cv=7).fit(X_train, self.y_train)
            print(f'Best parameters: {rf_tuned.best_params_}')
            rf = rf_tuned.best_estimator_
            r2_train = cross_val_score(rf, X_train, self.y_train, cv=7, scoring='r2').mean()
            mae_train = cross_val_score(rf, X_train, self.y_train, cv=7, scoring='neg_mean_absolute_error').mean()*-1
            # time to train
            start = time.time()
            rf.fit(X_train, self.y_train)
            end = time.time()
            train_time = end - start
            y_pred = rf.predict(X_test)
            r2_test = rf.score(X_test, self.y_test)
            mae_test = mean_absolute_error(self.y_test, y_pred)
            print('Random Forest Results (Tuned): ')
            print(f'Training R2: {r2_train}')
            print(f'Train MAE: {mae_train}')
            print(f'Test R2: {r2_test}')
            print(f'Test MAE: {mae_test}')
            hyperparameters_used = rf_tuned.best_params_

        # time to query model
        start = time.time()
        for i in range(1000):
            row = X_test.sample()
            rf.predict(row)
        end = time.time()
        query_time = (end - start) / 1000

        if plot:
            self.plot_pred_vs_actual(self.y_test, y_pred)
        
        if write_scores:
            self.scores.loc[len(self.scores.index)] = ['Random Forest', features, hyperparameters_used, r2_train,
                                                        mae_train, r2_test, mae_test, query_time, train_time]
        
        if return_model:
            return rf

    # Decision Tree
    def dt(self, features, plot=False, return_model=False, write_scores=False, params=None):

        # select data from test/train datasets
        X_train = self.X_train[features]
        X_test = self.X_test[features]

        # train and fit simple or tuned model
        if not params:
            # create model
            dt = DecisionTreeRegressor(**self.dt_hyperparameters)
            r2_train = cross_val_score(dt, X_train, self.y_train, cv=7, scoring='r2').mean()
            mae_train = cross_val_score(dt, X_train, self.y_train, cv=7, scoring='neg_mean_absolute_error').mean()*-1
            # time to train
            start = time.time()
            dt.fit(X_train, self.y_train)
            end = time.time()
            train_time = end - start
            y_pred = dt.predict(X_test)
            r2_test = dt.score(X_test, self.y_test)
            mae_test = mean_absolute_error(self.y_test, y_pred)
            print('Decision Tree Results: ')
            print(f'Training R2: {r2_train}')
            print(f'Train MAE: {mae_train}')
            print(f'Test R2: {r2_test}')
            print(f'Test MAE: {mae_test}')
            hyperparameters_used = self.dt_hyperparameters

        else:
            dt_tuned = GridSearchCV(estimator=DecisionTreeRegressor(random_state=30), n_jobs=-1, param_grid=params, cv=7).fit(X_train, self.y_train)
            print(f'Best parameters: {dt_tuned.best_params_}')
            dt = dt_tuned.best_estimator_
            r2_train = cross_val_score(dt, X_train, self.y_train, cv=7, scoring='r2').mean()
            mae_train = cross_val_score(dt, X_train, self.y_train, cv=7, scoring='neg_mean_absolute_error').mean()*-1
            # time to train
            start = time.time()
            dt.fit(X_train, self.y_train)
            end = time.time()
            train_time = end - start
            y_pred = dt.predict(X_test)
            r2_test = dt.score(X_test, self.y_test)
            mae_test = mean_absolute_error(self.y_test, y_pred)
            print('Decision Tree Results (Tuned): ')
            print(f'Training R2: {r2_train}')
            print(f'Train MAE: {mae_train}')
            print(f'Test R2: {r2_test}')
            print(f'Test MAE: {mae_test}')
            hyperparameters_used = dt_tuned.best_params_

        # time to query model
        start = time.time()
        for i in range(1000):
            row = X_test.sample()
            dt.predict(row)
        end = time.time()
        query_time = (end - start) / 1000

        if plot:
            self.plot_pred_vs_actual(self.y_test, y_pred)
            # visualize tree
            export_graphviz(dt, out_file='tree.dot', feature_names=features, rounded=True, filled=True)
            graph = pydotplus.graph_from_dot_file('tree.dot')
            display(Image(graph.create_png()))
        
        if write_scores:
            self.scores.loc[len(self.scores.index)] = ['Decision Tree', features, hyperparameters_used, r2_train,
                                                        mae_train, r2_test, mae_test, query_time, train_time]      
        if return_model:
            return dt
            
    # Neural Network
    def nn(self, features, plot=False, return_model=False, write_scores=False, simple=True, max_layers=3, tuner_model=None):
        
        # helper function for plot loss
        def plot_loss(history):
            f2 = plt.figure(2)
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            # plt.ylim([0, 50])
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True)

        # select data from test/train datasets
        X_train = self.X_train[features]
        X_test = self.X_test[features]

        # create normalization layer
        if len(features) == 1:
            normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
        else:
            normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(X_train))

        # build simple model with 2 hidden layers
        if simple:
            model_hidden_layers = {'layer_1': 64, 'layer_2': 64}
            model = keras.Sequential([normalizer, 
                                        layers.Dense(units=64,  activation='relu', input_shape=[len(features)], name='layer_1'),
                                        layers.Dense(units=64,  activation='relu', input_shape=[len(features)], name='layer_2'),
                                        layers.Dense(1, name='output_layer')
                                    ])
            model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
            # train time
            start = time.time()
            history = model.fit(X_train, self.y_train, epochs=100, verbose=0, validation_split=0.2)
            end = time.time()
            train_time = end - start
            y_pred = model.predict(X_test)
            
            # get scores for test and training datasets
            r2_train = r2_score(self.y_train, model.predict(X_train))
            mae_train = mean_absolute_error(self.y_train, model.predict(X_train))
            r2_test = r2_score(self.y_test, y_pred)
            mae_test = mean_absolute_error(self.y_test, y_pred)
            print('Neural Network Results: ')
            print(f'Training R2: {r2_train}')
            print(f'Train MAE: {mae_train}')
            print(f'Test R2: {r2_test}')
            print(f'Test MAE: {mae_test}')

        if not tuner_model:
            # default tuner for neural network
            print('Using default tuner for Neural Network')
            def tuner(hp):
                model = keras.Sequential()
                # add normalization layer
                model.add(normalizer)
                # add dense layers
                for i in range(hp.Int('num_layers', 1, max_layers)):
                    units = hp.Int(f'units{i}', min_value=10, max_value=200, step=10)
                    model.add(layers.Dense(units, activation='relu'))
                # add output layer
                model.add(layers.Dense(1))
                # compile model
                model.compile(loss='mean_absolute_error',
                            metrics=[keras.metrics.MeanAbsoluteError()],
                            optimizer=tf.keras.optimizers.Adam(0.001))
                return model
        else:
            tuner = tuner_model

        if not simple:
            cwd = os.getcwd()
            dir = 'untitled_project'
            path = os.path.join(cwd, dir)
            if os.path.exists(path):
                shutil.rmtree(path)
            # define tuner
            tuner = kt.Hyperband(
                hypermodel=tuner,
                objective=kt.Objective('val_mean_absolute_error', direction='min'),
                max_epochs=10,
                factor=3,
                overwrite=True
            )
            tuner.search(X_train, self.y_train, epochs=10, validation_split=0.2, verbose=False)
            # get optimal parameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print(best_hps.values)
            # build model with optimal parameters
            model = tuner.hypermodel.build(best_hps)
            # train time
            start = time.time()
            history = model.fit(X_train, self.y_train, epochs=150, validation_split=0.2, verbose=False)
            end = time.time()
            train_time = end - start
            # get epoch with lowest MAE
            val_loss_per_epoch = history.history['val_mean_absolute_error']
            best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
            print(f'Best epoch: {best_epoch}')
            # get scores
            y_pred = model.predict(X_test)
            r2_train = r2_score(self.y_train, model.predict(X_train))
            mae_train = mean_absolute_error(self.y_train, model.predict(X_train))
            r2_test = r2_score(self.y_test, y_pred)
            mae_test = mean_absolute_error(self.y_test, y_pred)
            print('Neural Network Results (Tuned): ')
            print(f'Train R2: {r2_train}')
            print(f'Train MAE: {mae_train}')
            print(f'Test R2: {r2_test}')
            print(f'Test MAE: {mae_test}')

        # time to query model
        start = time.time()
        for i in range(1000):
            row = X_test.sample()
            model.predict(row)
        end = time.time()
        query_time = (end - start) / 1000

        if plot:
            plot_loss(history)
            self.plot_pred_vs_actual(self.y_test, y_pred)
        
        if write_scores:
            if simple:
                self.scores.loc[len(self.scores.index)] = ['Neural Network', features, model_hidden_layers, r2_train, 
                                                           mae_train, r2_test, mae_test, query_time, train_time]
            else:
                self.scores.loc[len(self.scores.index)] = ['Neural Network', features, best_hps.values, r2_train, 
                                                           mae_train, r2_test, mae_test, query_time, train_time]
        
        if return_model:
            return model
        

