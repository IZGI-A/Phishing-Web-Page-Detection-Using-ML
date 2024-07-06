import os
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier
import torch

class ModelBuilder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self, path):
        
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)

        embeddings = np.array(embeddings)
        print(embeddings.shape)
        print(embeddings)
        np.random.shuffle(embeddings)

        labels = embeddings[:, -1]
        features = embeddings[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        transformer_name = os.path.basename(path).replace("embeddings-", "").replace(".pkl", "")
        return X_train, X_test, y_train, y_test, transformer_name

    def create_model(self, transformer_name, model_name, X_train, y_train):
        
        if model_name == "xgboost":
            model = XGBClassifier(
                n_estimators=10000, learning_rate=0.1, max_depth=2, 
                early_stopping_rounds=100, use_label_encoder=False
            )
            
            model.fit(X_train, y_train, verbose=True, eval_set=[(X_train, y_train)])
            model.save_model(f"model/{transformer_name}_xgboost_model.json")

        # elif model_name == "catboost":
        #     model = CatBoostClassifier(
        #         task_type="GPU", devices='0:1', iterations=10000, 
        #         learning_rate=1, depth=2
        #     )
        #     model.fit(X_train, y_train, verbose=True, eval_set=(X_train, y_train), early_stopping_rounds=100)
        #     model.save_model(f"model/{transformer_name}_catboost_model.cbm", format="cbm")

        elif model_name == "ann":
            model = MLPClassifier(
                hidden_layer_sizes=(500, 200, 10), max_iter=200, activation='relu', 
                solver='adam', random_state=1, verbose=True, n_iter_no_change=30
            )
            model.fit(X_train, y_train)
            with open(f"model/{transformer_name}_ann_model.pkl", "wb") as f:
                pickle.dump(model, f)

        else:
            raise ValueError("Model name is not valid")

    def load_model(self, transformer_name, model_name):
        
        if model_name == "xgboost":
            model = XGBClassifier()
            model.load_model(f"model/{transformer_name}_xgboost_model.json")

        # elif model_name == "catboost":
        #     model = CatBoostClassifier()
        #     model.load_model(f"model/{transformer_name}_catboost_model.cbm", format="cbm")

        elif model_name == "ann":
            with open(f"model/{transformer_name}_ann_model.pkl", "rb") as f:
                model = pickle.load(f)

        else:
            raise ValueError("Model name is not valid")
        
        return model

    def test_model(self, model, X_test, y_test):
        
        y_pred = model.predict(X_test)

        print(f"Testing the model... {model.__class__.__name__}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    def test_specific_model(self, model_name, path, X_test, y_test):
       
        model = self.load_model(path, model_name)
        self.test_model(model, X_test, y_test)

    def save_copy_model(self, path, transformer_name, model_name):
       
        model = self.load_model(path, model_name)
        if model_name == "xgboost":
            model.save_model(f"model/{transformer_name}_xgboost_model.json")
        elif model_name == "catboost":
            model.save_model(f"model/{transformer_name}_catboost_model.cbm", format="cbm")
        elif model_name == "ann":
            with open(f"model/{transformer_name}_ann_model.pkl", "wb") as f:
                pickle.dump(model, f)
        else:
            raise ValueError("Model name is not valid")

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    
    if len(sys.argv) < 3:
        print("Usage: python model_build.py <model_name> <embeddings_file_path>")
        sys.exit(1)

    model_name = sys.argv[1]  # xgboost, catboost, ann
    embeddings_file_path = sys.argv[2]
    
    model_builder = ModelBuilder()
    X_train, X_test, y_train, y_test, transformer_name = model_builder.load_data(embeddings_file_path)
    model_builder.create_model(transformer_name, model_name, X_train, y_train)
    loaded_model = model_builder.load_model(transformer_name, model_name)
    model_builder.test_model(loaded_model, X_test, y_test)
