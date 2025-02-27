from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
# import skmap_bindings as skb
from joblib import Parallel, delayed
import threading
import numpy as np

def _single_prediction(predict, X, out, i, lock):
    prediction = predict(X, check_input=False)
    with lock:
        out[i, :] = prediction

def cast_tree_rf(model):
    model.__class__ = TreesRandomForestRegressor
    return model

class TreesRandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed according
        to a list of functions that receives the predicted regression targets of each 
        single tree in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            `dtype=np.float32. If a sparse matrix is provided, it will be
            converted into a sparse `csr_matrix.

        Returns
        -------
        s : an ndarray of shape (n_estimators, n_samples)
            The predicted values for each single tree.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # store the output of every estimator
        assert(self.n_outputs_ == 1)
        pred_t = np.empty((len(self.estimators_), X.shape[0]), dtype=np.float32)
        # Assign chunk of trees to jobs
        n_jobs = min(self.n_estimators, self.n_jobs)
        # Parallel loop prediction
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_single_prediction)(self.estimators_[i].predict, X, pred_t, i, lock)
            for i in range(len(self.estimators_))
        )
        return pred_t
    
def separate_data(prop, filt, space, output_folder, df): 
    # df = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/002_data_whole.csv',low_memory=False) 
    os.makedirs(output_folder, exist_ok=True)
    
    ### data set preparation
    # clean the data according to each properties
    df = df.loc[df[prop].notna()]
    df = df.loc[df[f'{prop}_qa']>filt]
    # df[prop].hist(bins=40)
    
    # set target variable
    if space=='log1p':
        df.loc[:,f'{prop}_log1p'] = np.log1p(df[prop])
        tgt = f'{prop}_log1p'
    else:
        tgt = prop
            
    # split calibration, train and test for benchmark
    bd_val = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/003.0_validate.pnts.rob_bd.csv',low_memory=False)
    oc_val = pd.read_csv(f'/home/opengeohub/xuemeng/work_xuemeng/soc/data/003.1_validate.pnts.rob_soc.csv',low_memory=False)
    idl = bd_val['id'].values.tolist() + oc_val['id'].values.tolist()
    idl = [str(i) for i in idl]
    test = df.loc[df['id'].isin(idl)] # individual test datasets
    cal_train = df.loc[~df['id'].isin(idl)] # calibration and train
    # get 10% of training data as calibration for parameter fine tuning and feature selection
    cal_train.reset_index(drop=True, inplace=True)
    cal = cal_train.groupby('tile_id', group_keys=False).apply(lambda x: x.sample(n=max(1, int(np.ceil(0.16 * len(x))))))
    # the rest as training dataset
    train = cal_train.drop(cal.index)
    # if test_size>0:
    #     test = test.iloc[0:round(len(test)*test_size)]
    #     train = train.iloc[0:round(len(train)*test_size)]
    #     cal = cal.iloc[0:round(len(cal)*test_size)]
    print(f'calibration size {len(cal)}, training size {len(train)}, test size {len(test)}')
    print(f'sum {len(cal)+len(train)+len(test)}, df {len(df)}')
    print(f'ratio cal:trai - {len(cal)/len(train):.2f}')
    cal.to_csv(f'{output_folder}/benchmark_cal.pnts_{prop}.csv',index=False)
    train.to_csv(f'{output_folder}/benchmark_train.pnts_{prop}.csv',index=False)
    test.to_csv(f'{output_folder}/benchmark_test.pnts_{prop}.csv',index=False)
    return cal, train, test

