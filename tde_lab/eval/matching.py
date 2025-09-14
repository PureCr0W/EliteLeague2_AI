import re
from rouge_score import rouge_scorer

def exact_match(pred, truth):
    return pred.strip() == truth.strip()

def substring_match(pred, truth):
    return truth.strip() in pred

def rougeL(pred, truth):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(truth, pred)["rougeL"].fmeasure
