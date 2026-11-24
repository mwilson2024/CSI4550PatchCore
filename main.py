"""Main menu for training and inference."""

import importlib
import os
from multiprocessing import freeze_support
from pathlib import Path

from config import (categoryOptions, conditionOptions, checkpointRoot, conditionsRoot, dataRoot, resultsRoot)


projectRoot = Path(__file__).resolve().parent
trainingStage = ('Train models (keep existing checkpoints)', 'train')
pipelineStages = (('Prepare degraded and enhanced images', 'prep'), ('Evalate baseline and morphology', 'eval'),
                  ('Evaluate brightness adaptive fusion', 'fusion'),
                  ('Run threshold calibration experiments', 'calibrate_thresholds'),
                  ('Create baseline and morphology charts', 'plot_results'), ('Create fusion charts', 'plot_fusion'),
                  ('Create calibation charts', 'plot_calibration'))
demoStages = (('Standalone baseline training and testing demo', 'baseline'),
              ('Preview image transforations', 'experiments'),
              ('Inspect predictions on dark images', 'run_experiment'))


def choose(prompt, choices, default):
    while True:
        answer = input(f'{prompt} [{default}]: ').strip().lower() or default
        if answer in choices:
            return answer
        print('Pick one of these: ' + ', '.join(choices))


def askYesNo(prompt):
    return choose(prompt + ' (y/n)', ('y', 'yes', 'n', 'no'), 'y') in ('y', 'yes')


def chooseCategories():
    print('\nImages to process:')
    print('  0. All six categories (all images within each stage\'s split)')
    for number, category in enumerate(categoryOptions, 1):
        print(f'  {number}. {category}')
    choices = ['0', 'all'] + [str(n) for n in range(1, len(categoryOptions) + 1)] + categoryOptions
    answer = choose('Choose a category number or name', choices, '0')
    if answer in ('0', 'all'):
        return list(categoryOptions)
    return [categoryOptions[int(answer) - 1] if answer.isdigit() else answer]


def userImageInference():
    print('\nPick the model for this image.')
    for number, category in enumerate(categoryOptions, 1):
        print(f'  {number}. {category}')
    print('  0. Back')
    answer = choose('Model category', ['0'] + [str(n) for n in range(1, len(categoryOptions) + 1)] + categoryOptions, '1')
    if answer == '0':
        return
    category = categoryOptions[int(answer) - 1] if answer.isdigit() else answer
    while True:
        value = input('Paste the image file path (Enter to cancel): ').strip()
        if not value:
            return
        path = Path(value.strip('\"\'')).expanduser()
        if path.is_file():
            break
        print(f'File not found: {path}. Try another image path.')
    try:
        importlib.import_module('predict_image').predictImage(path, category)
    except KeyboardInterrupt:
        print('\nImage run stopped. Back to the menu.')
    except Exception as error:
        print(f'\nImage run failed: {type(error).__name__}: {error}')


def checkInputs(module, categories):
    """Check the data and checkpoints before loading the model."""
    required = []
    if module in ('prep', 'train', 'baseline', 'experiments'):
        for category in categories:
            required.append(dataRoot / category / ('train/good' if module in ('train', 'baseline') else 'test'))
    if module in ('eval', 'fusion', 'calibrate_thresholds', 'run_experiment'):
        conditions = conditionOptions
        if module == 'fusion':
            conditions = ('original', 'original_gamma', 'dark', 'dark_gamma')
        elif module == 'run_experiment':
            conditions = ('dark',)
        for category in categories:
            required.append(checkpointRoot / category / 'patchcore.ckpt')
            required.extend(conditionsRoot / condition / category / 'test' for condition in conditions)
            if module == 'calibrate_thresholds':
                required.extend(resultsRoot / category / name for name in ('split_manifest.json', 'threshold.txt'))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing inputs:\n  ' + '\n  '.join(missing)
                                + '\nRun preparation/training first; calibration also needs baseline evaluation.')


def runStage(stage, categories):
    title, module = stage
    print(f'\n--- {title} ---\nCategories: {", ".join(categories)}', flush=True)
    checkInputs(module, categories)
    # Save plots without opening a window.
    os.environ['MPLBACKEND'] = 'Agg'
    importlib.import_module(module).main(categories=categories)
    print(f'Done: {title}', flush=True)


def runStages(stages, categories, guided=False):
    completed, skipped, failed = [], [], []
    for stage in stages:
        if guided and not askYesNo(f'\nDo you want to {stage[0].lower()}?'):
            skipped.append(stage[0])
            continue
        try:
            runStage(stage, categories)
        except KeyboardInterrupt:
            print('\nRun stopped. Back to the menu.')
            break
        except Exception as error:
            print(f'\nFailed at {stage[0]}: {type(error).__name__}: {error}')
            failed.append(stage[0])
            if not guided:
                print('Stopping here. The next parts need these results.')
                break
        else:
            completed.append(stage[0])
    print(f'\nDone: {len(completed)}, skipped: {len(skipped)}, failed: {len(failed)}')


def main():
    # Use the project folder for relative paths.
    os.chdir(projectRoot)
    print('PatchCore project')
    print('Michael Wilson')
    while True:
        print('\n1. Training\n2. Inference - guided (ask before each part)\n3. Inference - full pipeline\n4. Inference - run one part\n5. Older standalone demos (includes a training demo)\n6. Inference - my own image\n0. Exit')
        action = choose('Choose an option', ('0', '1', '2', '3', '4', '5', '6'), '2')
        if action == '0':
            return
        if action == '6':
            userImageInference()
            continue
        stages = (trainingStage,) if action == '1' else pipelineStages
        if action in ('4', '5'):
            available = demoStages if action == '5' else pipelineStages
            for number, (title, _) in enumerate(available, 1):
                print(f'  {number}. {title}')
            print('  0. Back')
            answer = choose('Choose a part', [str(n) for n in range(len(available) + 1)], '0')
            if answer == '0':
                continue
            stages = (available[int(answer) - 1],)
        categories = chooseCategories()
        runStages(stages, categories, guided=action == '2')


if __name__ == '__main__':
    freeze_support()
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print('\nClosed.')
