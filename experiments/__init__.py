# every step that main.py can run, by name
from preprocess.pipeline import Preprocess
from experiments.common_cohort import CommonCohortAudit
from experiments.task_comparison import TaskComparison
from experiments.alarm_timing import AlarmTiming
from experiments.explanation import Explanation
from experiments.robustness import Robustness
from experiments.cadence import Cadence
from experiments.supplements import Supplements
from experiments.review import Review
from experiments.refits import TargetedRefits

EXPERIMENTS = {c.name: c for c in [
    Preprocess,
    CommonCohortAudit,
    TaskComparison,
    AlarmTiming,
    Explanation,
    Robustness,
    Cadence,
    Supplements,
    Review,
    TargetedRefits,
]}
