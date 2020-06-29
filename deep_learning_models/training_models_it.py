def build_model(space,settings,dataset_training,save_model = False,normalized=True):
    from LJT_helper_functions.dataset_prep import prepare_dataset_prediction
    import numpy as np
    from hyperopt import Trials, STATUS_OK, tpe, STATUS_FAIL
    from keras.layers import Dense, Dropout, Activation, Bidirectional
    from keras.models import Sequential
    from keras.optimizers import RMSprop, Adam
    from keras.utils import to_categorical
    import keras.backend as K
    from sklearn.metrics import confusion_matrix
    from keras.layers import LSTM, GRU, LeakyReLU, CuDNNLSTM
    from LJT_helper_functions.deep_learning import get_data,make_bins
    from keras.callbacks import EarlyStopping
    import tensorflow as tf
    from datetime import datetime
    import warnings
    import gc
    import math
    np.random.seed(38)
    warnings.filterwarnings('ignore')
    final_columns = [col for col in df_copy.columns if '_next' not in col and 'bins' not in col and 'differe' not in col]



    # def get_f1(y_true, y_pred):  # taken from old keras source code
    #     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    #     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + K.epsilon())
    #     recall = true_positives / (possible_positives + K.epsilon())
    #     f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    #     return f1_val

    # def weighted_bce(y_true, y_pred):
    #     weights = (y_true * (1-percentage_true_training)) + 1.
    #     bce = K.binary_crossentropy(y_true, y_pred)
    #     weighted_bce = K.mean(bce * weights)
    #     return weighted_bce

    def retrieve_scores(model, X_test, y_test):
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
        print('---------------------------------NEW EXPERIMENT -----------------------------------------------')
        check = datetime.now()
        print(f"current time: {datetime.now().hour}:{datetime.now().minute}")
        gc.collect()
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()  # TF graph isn't same as Keras graph
        try:  # if exists remove
            del model
        except:
            pass
        dataset_training = make_bins(dataset_training,
                                     f"{settings['coin']}__ticker_info__close_price",
                                     int(space["time_ahead_prediction"]))
        bins = dataset_training['bins'].values
        bins_train = bins[:int(settings['training_size']*len(bins))]
        bins_test = bins[int(settings['training_size']*len(bins)):]

        print('Trues train', sum(bins_train)/len(bins_train))
        print('Trues test ',sum(bins_test)/len(bins_test))

        training_generator = get_data(dataset_training,
                                      type_data='train',
                                      batch_size=space['batch_size'],
                                      window_size=int(space["window_size"]),
                                      time_steps_ahead_prediction=int(space["time_ahead_prediction"]),
                                      training_size= settings["training_size"],
                                     final_columns=final_columns)

        validation_generator = get_data(dataset_training,
                                      type_data='test',
                                      batch_size=space['batch_size'],
                                      window_size=int(space["window_size"]),
                                      time_steps_ahead_prediction=int(space["time_ahead_prediction"]),
                                      training_size= settings["training_size"],
                                       final_columns=final_columns)

        output_size = 2
      #  y_train_binary = to_categorical(y_train)
       # y_test_binary = to_categorical(y_test)
        model = Sequential()
        input_shape=(space["window_size"], len(final_columns))
        for i in range(1, int(space['number_layers']) + 1):
            if space["bidrectional"]:
                if settings['algorithm'].upper() == 'LSTM':
                    model.add(Bidirectional(LSTM(int(space['neurons']),
                                                 # only return sequences for last layer
                                                 return_sequences=False if int(space['number_layers']) == i else True,
                                                 #dropout = space['dropout'],
                                                # recurrent_dropout = space['dropout'],
                                                 input_shape=input_shape)))
                elif settings['algorithm'].upper() == "GRU":
                    model.add(Bidirectional(GRU(int(space['neurons']),
                                                # only return sequences for last layer
                                                return_sequences=False if int(space['number_layers']) == i else True,
                                               # dropout = space['dropout'],
                                               # recurrent_dropout = space['dropout'],
                                                input_shape=input_shape)))
            else:
                if settings['algorithm'].upper() == 'LSTM':
                    model.add(LSTM(int(space['neurons']),
                                   return_sequences=False if int(space['number_layers']) == i else True,
                                   #dropout = space['dropout'],
                                   #recurrent_dropout = space['dropout'],
                                   input_shape=input_shape))
                elif settings['algorithm'].upper() == "GRU":
                    model.add(GRU(int(space['neurons']),
                                  return_sequences=False if int(space['number_layers']) == i else True,
                                 # dropout = space['dropout'],
                                 # recurrent_dropout = space['dropout'],
                                  input_shape=input_shape))

            model.add(Dropout(space['dropout']))
        model.add(Dense(output_size))
        model.add(Activation(space['activation_function']))

        #this also includes selp made loss fucnctions
        if space['loss'] == 'weighted_bce':
            print('use custom loss function')
            loss_function = weighted_bce
        else:
            loss_function = space['loss']
        # compoile
        model.compile(loss=loss_function, optimizer=get_optimizer_function(space), metrics=['accuracy'])
        result = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     epochs=int(space['epochs']),
                                     verbose=0,
                                     shuffle=False,
                                     callbacks=[early_stopping_monitor],
                                     steps_per_epoch=math.ceil(len(dataset_training)*settings["training_size"]/space['batch_size']),
                                     validation_steps = math.ceil(len(dataset_training)*(1-settings["training_size"])/space['batch_size'])
                         #  class_weight= weights_classes
                           )
        # print(model.summary())
        sensitivity, specificity, precision, accuracy, f_1 = retrieve_scores(model, X_test, y_test)
        # print(sensitivity, specificity,precision,accuracy,f_1 )

        # get the highest validation accuracy of the training epochs
        highest_val_ac = np.amax(result.history['val_accuracy'])
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

        return {'loss': 1 - highest_val_ac,  # it is an minimalization problem
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
