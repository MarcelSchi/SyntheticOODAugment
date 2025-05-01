from app.training.load_config import ConfigLoader
from app.evaluation.register_evaluation_metrices import Evaluation_Registry, auto_register_evaluation_metrices


class EvaluationMetric:
    def __init__(self, config_loader: ConfigLoader, evaluation_type):
        """
        Automatically registering evaluation metrics helps to add new evaluation types dynamically.
        Depending on the model and the way to interpret data, new methods can be useful (e.g AUC, ...)
        """
        auto_register_evaluation_metrices(module_path="app.evaluation")

        self.config = config_loader.get_config()
        # check if the defined lable in the configuration file exists. If not: list all metrices that are defined.
        if evaluation_type not in Evaluation_Registry:
            available_evaluations = ", ".join(Evaluation_Registry.keys())
            raise ValueError(
                f"Evaluation '{evaluation_type}' not registered. "
                f"Please check all available evaluation metrics: {available_evaluations}."
            )
        evaluation_class = Evaluation_Registry[evaluation_type]
        self.evaluation_metric = evaluation_class(self.config)

    def get_evaluation_metric(self):
        return self.evaluation_metric.evaluate_model
