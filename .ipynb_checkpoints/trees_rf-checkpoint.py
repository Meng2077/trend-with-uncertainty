from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from collections import defaultdict
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
    


from collections import defaultdict
def cast_node_rf(model, X_train, y_train):
    model.__class__ = NodeRandomForestRegressor
    model._store_leaf_training_targets(X_train, y_train)
    return model


def _leaf_observation_lookup(tree, X_test_row, train_leaf_dict, out, i, lock):
    leaf = tree.apply(X_test_row.reshape(1, -1))[0]
    values = train_leaf_dict.get(leaf, [])
    with lock:
        out[i] = values

class NodeRandomForestRegressor(RandomForestRegressor):
    def _store_leaf_training_targets(self, X_train, y_train):
        self._leaf_value_store_ = []
        for tree in self.estimators_:
            leaf_ids = tree.apply(X_train)
            leaf_dict = defaultdict(list)
            for idx, leaf_id in enumerate(leaf_ids):
                leaf_dict[leaf_id].append(y_train[idx])
            self._leaf_value_store_.append(leaf_dict)

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        n_samples = X.shape[0]
        pred_n = [[] for _ in range(n_samples)]
        lock = threading.Lock()

        for tree_idx, tree in enumerate(self.estimators_):
            leaf_dict = self._leaf_value_store_[tree_idx]
            per_tree_out = [[] for _ in range(n_samples)]

            Parallel(n_jobs=1)(  # per tree, still can run in parallel if needed
                delayed(_leaf_observation_lookup)(tree, X[i], leaf_dict, per_tree_out, i, lock)
                for i in range(n_samples)
            )

            # Append this tree's contribution
            for i in range(n_samples):
                pred_n[i].extend(per_tree_out[i])

        return pred_n 

def pad_leaf_outputs_to_array(leaf_outputs, pad_value=np.nan):
    """
    Convert a list of lists of varying length into a 2D numpy array,
    padding with `pad_value` (default: np.nan).
    
    Parameters
    ----------
    leaf_outputs : list of lists
        Each inner list contains training target values from all leaves
        that a test sample falls into.
    
    Returns
    -------
    padded_array : np.ndarray of shape (n_samples, max_len)
    """
    max_len = max(len(lst) for lst in leaf_outputs)
    n_samples = len(leaf_outputs)

    padded_array = np.full((n_samples, max_len), pad_value, dtype=np.float32)

    for i, row in enumerate(leaf_outputs):
        padded_array[i, :len(row)] = row

    return padded_array   
    
# def _leaf_observation_lookup(tree, x_row, train_leaf_dict):
#     leaf = tree.apply(np.asarray(x_row).reshape(1, -1))[0]
#     return train_leaf_dict.get(leaf, [])

# def _parallel_leaf_lookup(tree_idx, sample_idx, X_test, estimators, leaf_value_store):
#     tree = estimators[tree_idx]
#     leaf_dict = leaf_value_store[tree_idx]
#     x_row = X_test[sample_idx]
#     values = _leaf_observation_lookup(tree, x_row, leaf_dict)
#     return sample_idx, values



# class NodeRandomForestRegressor(RandomForestRegressor):
#     def _store_leaf_training_targets(self, X_train, y_train):
#         self._leaf_value_store_ = []
#         # y_train = np.asarray(y_train)  # Ensure positional indexing
#         for tree in self.estimators_:
#             leaf_ids = tree.apply(X_train)
#             leaf_dict = defaultdict(list)
#             for idx, leaf_id in enumerate(leaf_ids):
#                 leaf_dict[leaf_id].append(y_train[idx])
#             self._leaf_value_store_.append(leaf_dict)

#     def predict(self, X):
#         check_is_fitted(self)
#         X = self._validate_X_predict(X)

#         n_samples = X.shape[0]
#         n_trees = len(self.estimators_)
#         pred_n = [[] for _ in range(n_samples)]

#         results = Parallel(n_jobs=self.n_jobs)(
#             delayed(_parallel_leaf_lookup)(
#                 tree_idx, sample_idx, X, self.estimators_, self._leaf_value_store_
#             )
#             for tree_idx in range(n_trees)
#             for sample_idx in range(n_samples)
#         )

#         for sample_idx, values in results:
#             pred_n[sample_idx].extend(values)

#         return pred_n  # shape: (n_samples,) with combined leaf outputs

#     from sklearn.ensemble import RandomForestRegressor


