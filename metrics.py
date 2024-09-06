import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score

# Initialize global variables for storing metrics across batches
global_bleu4, global_meteor, global_rouge, global_cider = [], [], [], []

# Initialize COCO evaluation metrics
rouge_scorer = Rouge()
cider_scorer = Cider()

def convert_to_dict_format(preds, gts):
    """
    Converts the lists of predictions and ground truths into the dictionary format required by pycocoevalcap.
    Keys are the indices in the list.
    """
    res = {idx: [pred] for idx, pred in enumerate(preds)}
    gts_dict = {idx: [gt] for idx, gt in enumerate(gts)}
    return res, gts_dict

def metrics(preds, gts):
    """
    preds: list of generated outputs (list of strings)
    gts: list of ground truth answers (list of strings)
    """
    global global_bleu4, global_meteor, global_rouge, global_cider

    # Convert predictions and ground truths to dictionary format
    preds_dict, gts_dict = convert_to_dict_format(preds, gts)

    # BLEU-4 Calculation
    bleu4_scores = [
        sentence_bleu([gt.split()], pred.split(), smoothing_function=SmoothingFunction().method1, weights=(0.25, 0.25, 0.25, 0.25))
        for pred, gt in zip(preds, gts)
    ]
    global_bleu4.extend(bleu4_scores)

    # NLTK METEOR Calculation (replacing pycocoeval METEOR)
    meteor_scores = [
        meteor_score([gt.split()], pred.split()) for pred, gt in zip(preds, gts)
    ]
    global_meteor.extend(meteor_scores)

    # ROUGE-L Calculation
    _, rouge_scores = rouge_scorer.compute_score(gts_dict, preds_dict)
    global_rouge.extend(rouge_scores)

    # CIDEr Calculation
    _, cider_scores = cider_scorer.compute_score(gts_dict, preds_dict)
    global_cider.extend(cider_scores)

def tally_metrics():
    """
    Computes the final average of all metrics accumulated in the global variables.
    """
    global global_bleu4, global_meteor, global_rouge, global_cider
    
    # Calculate the average of all stored scores
    avg_bleu4 = sum(global_bleu4) / len(global_bleu4) if global_bleu4 else 0
    avg_meteor = sum(global_meteor) / len(global_meteor) if global_meteor else 0
    avg_rouge = sum(global_rouge) / len(global_rouge) if global_rouge else 0
    avg_cider = sum(global_cider) / len(global_cider) if global_cider else 0

    # Print the final metric scores
    print(f"BLEU-4: {avg_bleu4:.4f}")
    print(f"METEOR: {avg_meteor:.4f}")
    print(f"ROUGE-L: {avg_rouge:.4f}")
    print(f"CIDEr: {avg_cider:.4f}")

    # Clear global metrics for the next evaluation
    global_bleu4.clear()
    global_meteor.clear()
    global_rouge.clear()
    global_cider.clear()
