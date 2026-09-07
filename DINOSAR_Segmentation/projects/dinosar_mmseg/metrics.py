from collections import OrderedDict

import numpy as np
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class ClasswiseIoUMetric(IoUMetric):
    """IoU metric that returns per-class scalars in addition to mIoU."""

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f"results are saved to {self.output_dir}")
            return OrderedDict()

        results = tuple(zip(*results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.dataset_meta["classes"]
        summary = OrderedDict(
            {metric: np.round(np.nanmean(values) * 100, 2) for metric, values in ret_metrics.items()}
        )
        metrics = OrderedDict()
        for key, val in summary.items():
            metric_key = key if key == "aAcc" else "m" + key
            metrics[metric_key] = round(float(val), 2)

        ret_metrics.pop("aAcc", None)
        class_metrics = OrderedDict(
            {metric: np.round(values * 100, 2) for metric, values in ret_metrics.items()}
        )
        class_metrics.update({"Class": class_names})
        class_metrics.move_to_end("Class", last=False)

        table = PrettyTable()
        for key, val in class_metrics.items():
            table.add_column(key, val)
        print_log("per class results:", logger)
        print_log("\n" + table.get_string(), logger=logger)

        for class_idx, class_name in enumerate(class_names):
            for metric_name, metric_values in class_metrics.items():
                if metric_name == "Class":
                    continue
                metrics[f"{metric_name}/{class_name}"] = round(float(metric_values[class_idx]), 2)

        return metrics
