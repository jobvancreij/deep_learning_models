def build_model(space,settings,dataset_training,save_model = False,normalized=True):
    """
    This function builds models according to the parameters given
    :param space: the deep learning specific variables
    :param settings: general settings
    :param dataset_training: the clean dataset before making sequences
    :param save_model: if true, return the model. Needed for training of the final model
    :param normalized: should the data be normalized
    :return:
    """
    import os
    os.environ['PYTHONHASHSEED']=str(38)
    import random
    random.seed(38)
    import numpy as np
    np.random.seed(38)
    import tensorflow as tf
    tf.random.set_seed(38)
    from LJT_helper_functions.dataset_prep import prepare_dataset_prediction
    from hyperopt import Trials, STATUS_OK, tpe, STATUS_FAIL
    from keras.layers import Dense, Dropout, Activation, Bidirectional
    from keras.models import Sequential
    from keras.optimizers import RMSprop, Adam
    from keras.utils import to_categorical
    import keras.backend as K
    from sklearn.metrics import confusion_matrix
    from keras.layers import LSTM, GRU, LeakyReLU, CuDNNLSTM
    from keras.callbacks import EarlyStopping
    from datetime import datetime
    import warnings
    import gc
    from tensorflow.keras.backend import clear_session
    warnings.filterwarnings('ignore')

    def retrieve_scores(model, X_test, y_test):
        """
        Calculate the scores of interest to return after training of the model. Such that, the optimization algorithms
        can make their decisions.
        :param model: the model trained
        :return:
        """
        x = model.predict(X_test)
        y_pred = np.argmax(x, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)  # recall
        specificity = tn / (fp + tn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f_1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
        return sensitivity, specificity, precision, accuracy, f_1

    def get_optimizer_function(space):
        """
        The input for optimizing is a string that says which optimization function should be chosen. This fucntion
        returns the correct optimization function
        :param space: all variables for deep learning choices
        """
        if space["optimizer"] == "leakyrelu":
            return LeakyReLU(lr=space["learning_rate"])  # no momentum
        elif space["optimizer"] == "rmsprop":
            return RMSprop(lr=space["learning_rate"])  # no momentum
        elif space["optimizer"] == "adam":
            return Adam(lr=space["learning_rate"])  # no momentum
        else:
            raise TypeError("NOT CORRECT ACTIVATION FUNCTION SELECTED")


    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=15,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    try:
        clear_session()
        print('---------------------------------NEW EXPERIMENT -----------------------------------------------')
        check = datetime.now()
        print(f"current time: {datetime.now().hour}:{datetime.now().minute}")
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()  # TF graph isn't same as Keras graph
        try:  # if exists remove
            del model
        except:
            pass
        # create the data needed for training and validating of the model
        X_train, X_test, y_train, y_test = prepare_dataset_prediction(dataset_training, {
            "coinpair": settings["coin"],
            "window_size": int(space["window_size"]),
            "training_size": settings["training_size"],
            "bins": settings["bins"],
            "time_steps_ahead_prediction": int(space["time_ahead_prediction"]),
            "include_transaction_cost": settings["include_transaction_cost"]
        }, normalized=normalized)
        #check the dataset distribution
        true_values_train =np.count_nonzero(y_train==1)
        negative_values_train = len(y_train) - true_values_train
        if true_values_train >= negative_values_train:
            weights_classes = {
                0: round(true_values_train/negative_values_train,2),
                1: 1
            }
        else:
            weights_classes = {
                0: 1,
                1: round(negative_values_train/true_values_train,2)
            }
        print('--data--')
        print("length training data = ",len(y_train))
        print(f"percentage True train = {true_values_train/len(y_train)}")
        print("length test data = ",len(y_test))
        print(f"percentage True test = {np.count_nonzero(y_test==1)/len(y_test)}")


        output_size = 2
        y_train_binary = to_categorical(y_train)
        y_test_binary = to_categorical(y_test)
        model = Sequential()
        for i in range(1, int(space['number_layers']) + 1):
            if space["bidrectional"]:
                if settings['algorithm'].upper() == 'LSTM':
                    model.add(Bidirectional(LSTM(int(space['neurons']),
                                                 # only return sequences for last layer
                                                 return_sequences=False if int(space['number_layers']) == i else True,
                                                 #dropout = space['dropout'],
                                                # recurrent_dropout = space['dropout'],
                                                 input_shape=(X_train.shape[1], X_train.shape[2]))))
                elif settings['algorithm'].upper() == "GRU":
                    model.add(Bidirectional(GRU(int(space['neurons']),
                                                # only return sequences for last layer
                                                return_sequences=False if int(space['number_layers']) == i else True,
                                               # dropout = space['dropout'],
                                               # recurrent_dropout = space['dropout'],
                                                input_shape=(X_train.shape[1], X_train.shape[2]))))
            else:
                if settings['algorithm'].upper() == 'LSTM':
                    model.add(LSTM(int(space['neurons']),
                                   return_sequences=False if int(space['number_layers']) == i else True,
                                   #dropout = space['dropout'],
                                   #recurrent_dropout = space['dropout'],
                                   input_shape=(X_train.shape[1], X_train.shape[2])))
                elif settings['algorithm'].upper() == "GRU":
                    model.add(GRU(int(space['neurons']),
                                  return_sequences=False if int(space['number_layers']) == i else True,
                                 # dropout = space['dropout'],
                                 # recurrent_dropout = space['dropout'],
                                  input_shape=(X_train.shape[1], X_train.shape[2])))

            model.add(Dropout(space['dropout']))
        model.add(Dense(output_size))
        model.add(Activation(space['activation_function']))

        #this also includes selp made loss fucnctions
        if space['loss_func'] == 'weighted_bce':
            print('use custom loss function')
            loss_function = weighted_bce
        else:
            loss_function = space['loss_func']
        # compoile
        model.compile(loss=loss_function, optimizer=get_optimizer_function(space), metrics=['accuracy'])
        result = model.fit(X_train, y_train_binary,
                           batch_size=int(space['batch_size']),
                           epochs=int(space['epochs']),
                           verbose=0,
                           validation_data=(X_test, y_test_binary),
                           shuffle=False,
                           callbacks=[early_stopping_monitor],
                           class_weight= weights_classes
                           )
        # print(model.summary())
        sensitivity, specificity, precision, accuracy, f_1 = retrieve_scores(model, X_test, y_test)
        # print(sensitivity, specificity,precision,accuracy,f_1 )

        # get the highest validation accuracy of the training epochs
        highest_val_ac = np.amax(result.history['val_accuracy'])
        print('highest_val_ac' ,highest_val_ac)
        # print(result.history['accuracy'])
        # print('Best validation accuracy of epoch:', accuracy)
        print({"sensitivity": sensitivity,
               "specificity": specificity,
               "precision": precision,
               "accuracy": accuracy,
               "f_1": f_1
               })
        print("Running time ",datetime.now() - check)

        if save_model: #when asked for the model return this
            return model
        del X_test
        del y_test
        del y_test_binary
        del X_train
        del y_train
        del y_train_binary
        return {'loss': 1 - accuracy,  # it is an minimalization problem
                'status': STATUS_OK,
                'sensitivity_val': sensitivity,
                'specificity_val': specificity,
                'precision_val': precision,
                'accuracy_val': accuracy,
                'highest_train_ac': np.amax(result.history['accuracy']),
                'f_1_val': f_1}
    except Exception as e:
        print(e)
        return {'status': STATUS_FAIL}
