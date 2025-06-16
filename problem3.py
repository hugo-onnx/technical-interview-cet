import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def load_summary(file_path: str) -> str:
    """Load a summary from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def evaluate_summaries(reference: str, candidate: str) -> dict:
    """
    Compare reference and candidate summaries using BLEU and ROUGE-L.

    Args:
        reference (str): Ground truth summary.
        candidate (str): Generated summary.

    Returns:
        dict: BLEU and ROUGE-L scores.
    """
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()

    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(reference, candidate)['rougeL'].fmeasure

    return {
        'BLEU': round(bleu, 4), # precision / exact matches
        'ROUGE-L': round(rouge_l, 4) # covered reference
    }

# Paths to the summaries
reference_path = os.path.join('data', 'summary-1-flan-ul2--article1.txt') # ground truth
candidate_path = os.path.join('data', 'summary-2-flan-ul2--article1.txt') # generated summary

# Read summaries
reference_summary = load_summary(reference_path)
candidate_summary = load_summary(candidate_path)

# Evaluate
scores = evaluate_summaries(reference_summary, candidate_summary)
print("Evaluation scores:", scores)
