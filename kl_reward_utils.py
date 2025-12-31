from verl.utils.reward_score import default_compute_score

def kl_compute_score(data_source, solution_str, ground_truth, **kwargs):
    """
    Custom compute score function that handles empty data_source by defaulting to 'math_dapo'.
    """
    if not data_source:
        data_source = "math_dapo"
    return default_compute_score(data_source, solution_str, ground_truth, **kwargs)
