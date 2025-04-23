class ModelEvaluator:
    def __init__(self, tolerance=3, min_improvement=0.005):
        self.tolerance = tolerance
        self.min_improvement = min_improvement
        self.counter = 0
        self.best_score = 0

    def check_for_early_stopping(self, score):
        if score >= self.best_score + self.min_improvement:
            self.best_score = score
            self.counter = 0

        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True

        return False
