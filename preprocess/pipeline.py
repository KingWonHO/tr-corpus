# cohort, design folds, four-role memberships and task inputs from the Zenodo corpus
from experiments.base import Experiment, split_arg
from preprocess.cohort import CohortStep
from preprocess.roles import RolesStep
from preprocess.tasks import TasksStep

STEPS = {'cohort': CohortStep, 'roles': RolesStep, 'tasks': TasksStep}


# preprocessing: results/preprocess/{cohort,roles,tasks}
class Preprocess(Experiment):
    name = 'preprocess'
    help = 'cohort, design folds, four-role memberships and task inputs from the Zenodo corpus'

    @staticmethod
    def add_arguments(ap):
        ap.add_argument('--steps', default=','.join(STEPS), help='comma-separated subset of cohort,roles,tasks (run in this order)')

    def run(self):
        steps = split_arg(self.args.steps)
        unknown = set(steps) - set(STEPS)
        if unknown:
            raise SystemExit('unknown preprocessing step(s): ' + ','.join(sorted(unknown)))
        for name in STEPS:
            if name in steps:
                print('STEP', name, flush=True)
                STEPS[name](self.paths).run()
