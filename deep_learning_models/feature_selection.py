import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from LJT_database.file_storage import extract_and_unpickle,pickle_and_store
def remove_low_corr(dataframe,coin,min_cor=0.4,):
    """
    Removes the columns which correlation is to low
    :return:
    """
    closing_price = dataframe[f"{coin}__ticker_info__close_price"].values
    last_start_time = dataframe['last_start_time'].values
    dataframe[f"{coin}_next"] = dataframe[f"{coin}__ticker_info__close_price"].shift(-10).fillna(0)
    # df['difference'] = df[f"{target_column}_next"] - df[target_column]
    # df['bins'] = df['difference'].apply(lambda x: 1 if x > 0 else 0) #value 1 if raise 0 if downwards

    correlation = dataframe.corr(method="pearson")
    dataframe = dataframe[[column for column in dataframe.columns if column not in (
        f"{coin}_next", 'difference')]]
    high_cor = correlation.loc[correlation[f"{coin}_next"].apply(lambda x: abs(x) > min_cor)][f"{coin}_next"].index
    del correlation
    df_after_cor = dataframe[[col for col in high_cor if col != f"{coin}_next"]]
    df_after_cor['last_start_time'] = last_start_time
    df_after_cor[f"{coin}__ticker_info__close_price"] = closing_price
    return df_after_cor

def add_diff_previous(df,settings):
    """
    Check the difference of current price with previous prices
    """
    minutes_lookback = settings['minutes_lookback']
    for i in range(1,minutes_lookback):
        df[f'next'] = df[f"{settings['coin']}__ticker_info__close_price"].shift(i).fillna(0)
        df['difference'] = df[f"next"] - df[f"{settings['coin']}__ticker_info__close_price"]
        df[f"price_change_lag_{i}"] = df['difference'].apply(lambda x: 1 if x > 0 else 0) #value 1 if raise 0 if downwards
        df.drop(columns=['next','difference'],inplace=True)
    return df
                                            
def scale_data(df,coin,type_scaler='min_max_scaler'):
    print('scaling')
    df = df.sort_values('last_start_time')
    closing_price = df[f"{coin}__ticker_info__close_price"].values
    last_start_time = df['last_start_time'].values
    if type_scaler == 'min_max_scaler':
        scaler = MinMaxScaler()
    elif type_scaler == 'standard_scaler':
        scaler = StandardScaler()
    else:
        raise TypeError("Please select correct scaler")
    columns_scaler = [col for col in df.columns]
    df_scaler = df.sort_values('last_start_time')[columns_scaler]
    scaler = scaler.fit(df_scaler)
    x_scaled = scaler.transform(df_scaler)
    
    df = pd.DataFrame(x_scaled, columns=columns_scaler).drop(['last_start_time'], axis=1)
    del df_scaler
    del x_scaled
    df['last_start_time'] = last_start_time #add clean last_start_time
    df['close_price_train'] = df[f"{coin}__ticker_info__close_price"] #change name close price so unscaled can be added
    df[f"{coin}__ticker_info__close_price"] = closing_price #add scaled closing price
    return df,scaler,columns_scaler

def initialize_pca(df,coin,percentage_variance=.99):
    pca = PCA(percentage_variance)
    # fit on data
    #init pca without close price and last_start_time
    columns_in_pca = [col for col in df.columns if col not in ['last_start_time',f"{coin}__ticker_info__close_price"]]
    pca = pca.fit(df[columns_in_pca])
    # access values and vectors
    print(f" The number of components in pca = ",pca.n_components_)

    
    B = pca.transform(df[columns_in_pca])
    columns = [f'column_{i}' for i in range(pca.n_components_)]
    df_components = pd.DataFrame(columns=columns, data=B)

    df_components['last_start_time'] = df['last_start_time'].values
    df_components[f"{coin}__ticker_info__close_price"] = df[f"{coin}__ticker_info__close_price"].values
    del df
    return df_components,pca,columns_in_pca



def create_reduced_features(df,settings):
    df = add_diff_previous(df,settings)
    scaler = extract_and_unpickle(settings['coin'],settings['experiment_date'],'scaler')
    pca = extract_and_unpickle(settings['coin'],settings['experiment_date'],'pca')
    columns_scaler = extract_and_unpickle(settings['coin'],settings['experiment_date'],'columns_scaler')
    columns_in_pca = extract_and_unpickle(settings['coin'],settings['experiment_date'],'columns_in_pca')

    #save features that are needed later on
    closing_price = df[f"{settings['coin']}__ticker_info__close_price"].values
    last_start_time = df['last_start_time'].values
    x_scaled = scaler.transform(df.sort_values('last_start_time')[columns_scaler]) #make sure scaler has right columns in right order
    df = pd.DataFrame(x_scaled, columns=columns_scaler).drop(['last_start_time'], axis=1)
    del x_scaled
    df['close_price_train'] = df[f"{settings['coin']}__ticker_info__close_price"] 
    B = pca.transform(df[columns_in_pca]) #make sure pca has right columns in right order
    del df
    columns = [f'column_{i}' for i in range(pca.n_components_)]
    df_components = pd.DataFrame(columns=columns, data=B)
    df_components['last_start_time'] = last_start_time
    df_components[f"{settings['coin']}__ticker_info__close_price"] = closing_price
    return df_components


def make_components_pca(dataset_prepared,settings, percentage_variance=.99,type_scaler='min_max_scaler',min_cor=0.4):
    """
    Make components with pca
    :param dataset_prepared: the
    :param settings:
    :param percentage_variance:
    :param min_cor:
    :param type_scaler:
    :return:
    """
    
    #df_after_cor = remove_low_corr(dataset_prepared,
    #                           settings['coin'],
    #                           min_cor=min_cor)
    #final_columns = df_after_cor.columns
    #pickle_and_store(settings["coin"],settings["experiment_date"],"final_columns",final_columns)
    dataset_prepared = add_diff_previous(dataset_prepared,settings)
    df,scaler,columns_scaler= scale_data(dataset_prepared,
                    coin = settings["coin"],
                    type_scaler=type_scaler,
                   )
    del dataset_prepared
    pickle_and_store(settings["coin"],settings["experiment_date"],"columns_scaler",columns_scaler)
    pickle_and_store(settings["coin"],settings["experiment_date"],"scaler",scaler)
    df_components,pca,columns_in_pca = initialize_pca(df,
                                   coin = settings["coin"],
                                   percentage_variance=percentage_variance
                                  )
    del df
    pickle_and_store(settings["coin"],
                     settings["experiment_date"],"columns_in_pca",columns_in_pca)              
    pickle_and_store(settings["coin"],
                     settings["experiment_date"],"pca",pca)
    return df_components,columns_scaler
