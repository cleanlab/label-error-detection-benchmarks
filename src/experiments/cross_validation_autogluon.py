import pandas as pd
import numpy as np
from autogluon.vision import ImagePredictor, ImageDataset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Tuple


def cross_val_predict_autogluon_image_dataset(
    dataset: ImageDataset,
    *,
    n_splits: int = 5,
    model_params: Dict = {"epochs": 1, "holdout_frac": 0.2},
    ngpus_per_trial: int = 1,
    time_limit: int = 7200,
    random_state: int = 123,
) -> Tuple:
    """Run stratified K-folds cross-validation with AutoGluon image model.

    Parameters
    ----------
    dataset : gluoncv.auto.data.dataset.ImageClassificationDataset
      AutoGluon dataset for image classification.

    n_splits : int, default=3
      Number of splits for stratified K-folds cross-validation.

    model_params : Dict, default={"epochs": 1, "holdout_frac": 0.2}
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    ngpus_per_trial : int, default=1
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    time_limit : int, default=7200
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    random_state : int, default=123
      Passed into AutoGluon's `ImagePredictor().fit()` method.

    Returns
    -------
    test_index_pred_probs : np.array
      Predicted probabilities from all test splits in cross-validation procedure.

    test_index_pred_features : np.array
      Predicted features (aka embeddings) from all test splits in cross-validation procedure.

    test_index_images : np.array
      Image file paths from all test splits in cross-validation procedure.

    test_index_labels : np.array
      Labels from all test splits in cross-validation procedure.

    skf_splits : list
      Train/test splits from cross-validation procedure

    cv_models : list
      Models from cross-validation procedure (K models for K-folds cross-validation)

    """

    # stratified K-folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    skf_splits = [
        [train_index, test_index]
        for train_index, test_index in skf.split(X=dataset, y=dataset.label)
    ]

    # save test predictions
    test_index_pred_probs = []
    test_index_pred_features = []
    test_index_images = []  # image file paths
    test_index_labels = []  # image labels

    # save models from each split
    cv_models = []

    # run cross-validation
    for split_num, split in enumerate(skf_splits):

        print("----")
        print(f"Running Cross-Validation on Split: {split_num + 1}")

        # split from stratified K-folds
        train_index, test_index = split

        # init model
        predictor = ImagePredictor(verbosity=0)

        # train model on train indices in this split
        predictor.fit(
            train_data=dataset.iloc[train_index],
            ngpus_per_trial=ngpus_per_trial,
            hyperparameters=model_params,
            time_limit=time_limit,
            random_state=random_state,
        )

        # predict on test indices in this split

        # predicted probabilities for test split
        pred_probs = predictor.predict_proba(
            data=dataset.iloc[test_index], as_pandas=False
        )

        # predicted features (aka embeddings) for test split
        # why does autogluon predict_feature return array of array for the features?
        # need to use stack to convert to 2d array (https://stackoverflow.com/questions/50971123/converty-numpy-array-of-arrays-to-2d-array)
        pred_features = np.stack(
            predictor.predict_feature(data=dataset.iloc[test_index], as_pandas=False)[
                :, 0
            ]
        )

        # save predictions for test indices in this split
        test_index_pred_probs.append(pred_probs)
        test_index_pred_features.append(pred_features)
        test_index_images.append(dataset.iloc[test_index].image.values)
        test_index_labels.append(dataset.iloc[test_index].label.values)

        # save model
        cv_models.append(predictor)

    # combine test predictions from all splits
    test_index_pred_probs = np.vstack(test_index_pred_probs)
    test_index_pred_features = np.vstack(test_index_pred_features)
    test_index_images = np.hstack(test_index_images)
    test_index_labels = np.hstack(test_index_labels)

    return (
        test_index_pred_probs,
        test_index_pred_features,
        test_index_images,
        test_index_labels,
        skf_splits,
        cv_models,
    )
