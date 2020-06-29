# Deep learning models Thesis

The goal of this repo is to build and test the deep learning models for the thesis
"Algorithmic trading of Ethereum/Bitcoin coinpair with deep learning and big data". <br> 

Additionally to training the deep learning models. This repository has the code for feature selection of the thesis. <br>

Before training a new model few codes have to be run that are described below <br> 

In order to install the packages access is needed to the following private github repo's: 
*   https://github.com/jobvancreij/LJT_helper_functions
*    https://github.com/jobvancreij/LJT_database

Install the library with (only possible with access): 
```console
pip install --upgrade git+https://github.com/jobvancreij/deep_learning_models
```

Step1 : specify the settings for the experiment (in the thesis those are generated)

```python
settings_analysis = {
    "algoirthm":"LSTM",
    "experiment_date":"5_28",
    "neurons":75,
    "dropout":0.4536499963739309,
    "loss_func": "binary_crossentropy",
    "activation_function":"relu",
    "number_layers":1,
    "batch_size":940,
    "epochs":500,
    "bidrectional":False,
    "window_size":70,
    "time_ahead_prediction":1,
    "optimizer":'adam',
    "learning_rate": 0.0025558826976665675
}

```
Step 2: Extract the general settings from firestore
```python
from LJT_database.firestore_codes import  retrieve_updates

general_settings=retrieve_updates(dataset=f"ETHBTC_LSTM_experiments",
                                        document="experiment_general_settings"
                                        )
```

Step 3: Query data, clean the data, and perform PCA.

```python
from deep_learning_models.feature_selection import create_reduced_features
from LJT_database.merge_dataset import retrieve_data_predictors
from LJT_database.feature_prep import feature_preperation

dataset_prepared = feature_preperation(retrieve_data_predictors(general_settings)) #retrieve and clea
df = create_reduced_features(dataset_prepared,general_settings)

```
Step 4: Run the training algorithm. This algorithm can also run without the previous steps and other data
than data used in this thesis. 
```python
from deep_learning_models.training_models import build_model

model = build_model(settings_analysis,general_settings,df,save_model=True,normalized=True)


```

