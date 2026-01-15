#This script is taken from https://github.com/kimdanny/Fair-RAG

# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/eval/evaluation.py
from rouge import Rouge
from sklearn.metrics import mean_squared_error, f1_score
from math import sqrt
from tqdm import tqdm

def _postprocess_text_classification(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    return preds, labels


def _postprocess_text_generation(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]

    return preds, labels


def get_metric_fn_f1(all_labels):
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1

    def metric_fn(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = _postprocess_text_classification(
            decoded_preds, decoded_labels
        )
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]

        f1_scores = []
        # Ensure predictions and references are integer-encoded if necessary
        for pred, ref in zip(decoded_preds, decoded_labels):
            score = f1_score([ref], [pred], average="macro")
            f1_scores.append(score)

        return f1_scores

    return metric_fn


def get_metric_fn_accuracy(all_labels):
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1

    def metric_fn(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = _postprocess_text_classification(
            decoded_preds, decoded_labels
        )
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        accuracy_scores = [
            (1 if pred == ref else 0)
            for pred, ref in zip(decoded_preds, decoded_labels)
        ]
        return accuracy_scores

    return metric_fn


def get_metric_fn_mae():
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0

    def metric_fn(decoded_preds, decoded_labels) -> list:
        decoded_preds, decoded_labels = _postprocess_text_classification(
            decoded_preds, decoded_labels
        )
        decoded_preds = [
            create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)
        ]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]
        mae_scores = [
            abs(pred - ref) for pred, ref in zip(decoded_preds, decoded_labels)
        ]
        return mae_scores

    return metric_fn


def get_metric_fn_rmse():
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0

    def metric_fn(decoded_preds, decoded_labels) -> list:
        decoded_preds, decoded_labels = _postprocess_text_classification(
            decoded_preds, decoded_labels
        )
        decoded_preds = [
            create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)
        ]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]

        rmse = sqrt(mean_squared_error(decoded_labels, decoded_preds))
        return rmse

    return metric_fn


def get_metric_fn_rouge_1():
    def metric_fn(predictions, references) -> list:
        rouge = Rouge()
        predictions, references = _postprocess_text_generation(predictions, references)
        scores = rouge.get_scores(predictions, references, avg=False)
        # Extracting ROUGE-1 F1 scores for each prediction-reference pair
        rouge_l_scores = [score["rouge-1"]["f"] for score in scores]
        return rouge_l_scores

    return metric_fn


def get_metric_fn_rouge_L():
    def metric_fn(predictions, references) -> list:
        rouge = Rouge()
        predictions, references = _postprocess_text_generation(predictions, references)
        for i, pred in tqdm(enumerate(predictions)):
            if (
                pred is None or
                (isinstance(pred, float) and math.isnan(pred)) or
                (isinstance(pred, str) and pred.strip() == "")
            ):
                print(f"Empty at index {i}: {repr(pred)}")

        for i, ref in tqdm(enumerate(references)):
            if (
                ref is None or
                (isinstance(ref, float) and math.isnan(ref)) or
                (isinstance(ref, str) and ref.strip() == "")
            ):
                print(f"Empty at index {i}: {repr(ref)}")

        scores = []
        for i, (pred, ref) in tqdm(enumerate(zip(predictions, references))):
            if pred and ref:
                try:
                    score = rouge.get_scores(pred, ref)[0]
                    scores.append(score)
                except Exception as e:
                    print(f"Error at index {i}: {e}")
                    scores.append({"rouge-l": {"f": 0.0}})
            else:
                scores.append({"rouge-l": {"f": 0.0}})
                
        rouge_l_scores = [score["rouge-l"]["f"] for score in scores]
        # scores = rouge.get_scores(predictions, references, avg=False)
        # # Extracting ROUGE-L F1 scores for each prediction-reference pair
        # rouge_l_scores = [score["rouge-l"]["f"] for score in scores]
        return rouge_l_scores

    return metric_fn
