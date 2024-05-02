import json
import pickle
import argparse
import evaluate
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from datasets import load_from_disk, concatenate_datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Run sentence similarity')
    parser.add_argument('--model_path', type=str, default=None, help='path to the model')
    parser.add_argument('--biosses_path', type=str, default=None, help='path to biosses dataset (on disk)')
    parser.add_argument('--output_dir', type=str, default=None, help='path to save the output files')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    return args

# Best Hyperparameters determined with Grid Search for BERT on biosses for SGDRegressor
_BEST_HYPERPARAMS = {'alpha': 0.0001,
    'eta0': 0.001,
    'loss': 'squared_error',
    'max_iter': 20,
    'penalty': 'l2'
}

def main():

    args = parse_args()
    # Loading Data and models
    biosses = load_from_disk(args.biosses_path)
    text_encoder = AutoModel.from_pretrained(args.model_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    regressor = SGDRegressor(**_BEST_HYPERPARAMS, random_state=args.seed)
    # In BLURB paper (Gu et al.) they say they adopt the splits of the Peng et al. paper, 
    # which uses 80% train and 20% test , so we must concatenate train and validation : 
    biosses_train = concatenate_datasets((biosses["train"],biosses["validation"]),split="train")
    # Preprocess dataset (Tokenization)
    train_dataset = tokenizer(
        text=biosses_train["text_1"],
        text_pair=biosses_train["text_2"],
        padding=True,
        return_tensors='pt'
    ).to('cuda')
    test_dataset = tokenizer(
        text=biosses["test"]["text_1"],
        text_pair=biosses["test"]["text_2"],
        padding=True,
        return_tensors='pt'
    ).to('cuda')
    # Encode text
    X_train = text_encoder(**train_dataset).last_hidden_state[:,0,:] # (batch_size, sequence_length, hidden_size) take only first [CLS] token
    X_train = X_train.to('cpu').detach().numpy()
    y_train = biosses_train["annotator_avg"]
    X_pred = text_encoder(**test_dataset).last_hidden_state[:,0,:] # to be used for prediction that's why y_pred and not y_test
    X_pred = X_pred.to('cpu').detach().numpy()
    y_test = biosses["test"]["annotator_avg"]
    # Train regressor
    regressor.fit(X_train,y_train)
    # Prediction
    y_pred = regressor.predict(X_pred)
    # Evaluation
    pearsonr = evaluate.load("../evaluation/metrics/evaluate_pearsonr.py")
    pearsonr_result = pearsonr.compute(predictions=y_pred,references=y_test)
    train_mse = mean_squared_error(y_train,regressor.predict(X_train))
    test_mse = mean_squared_error(y_test,y_pred)
    # Save results and model
    result_dict = {
        "train_mse": train_mse,
        "test_mse": test_mse
    } | pearsonr_result
    with open(args.output_dir + "/results.json","w") as f:
        json.dump(result_dict,f)
    with open(args.output_dir + "/sgd_regressor_model.pkl","wb") as f:
        pickle.dump(regressor,f)

if __name__ == '__main__':
    main()
