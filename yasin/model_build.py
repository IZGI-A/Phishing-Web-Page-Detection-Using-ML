import os
import pickle
import random
import sys

import catboost
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def serialize(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def deserialize(path: str):
    with open(path, 'rb') as f:
        temp = pickle.load(f)
    return temp


def check_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, "a") as _:
            pass
        # os.mknod(file_path)
    return None


source_folder_name = "embeddings"
destination_folder_name = "model"

model_name = "xlm-roberta"

legitimate_input = source_folder_name + "/" + "Legitimate/legitimate_out_" + model_name + ".pkl"
phishing_input = source_folder_name + "/" + "Phishing/phishing_out_" + model_name + ".pkl"

SEED = 42

random.seed(SEED)


def main():
    algorithm = sys.argv[1]  # "xgb" or "cat"

    # Write name of pkl file under /embeddings directory.
    embeddingfile = source_folder_name + "/" + sys.argv[2]  # source_folder_name + "/embeddings-" + model_name + ".pkl"

    merged_data = deserialize(embeddingfile)

    target_attribute_index = 768
    y = merged_data[:, target_attribute_index]

    X = np.delete(merged_data, target_attribute_index, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == "xgb":
        model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    elif algorithm == "cat":
        model = catboost.CatBoostClassifier(random_state=42, task_type="GPU")
    else:
        raise Exception(f"algorithm should either be xgb or cat but {algorithm} has given")

    model.fit(X_train, y_train)

    check_file(source_folder_name + "/" + model_name + "_based_model.pkl")
    serialize(model, destination_folder_name + "/" + model_name + "_" + algorithm + "_based_model.pkl")


if __name__ == '__main__':
    main()
