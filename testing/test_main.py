import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from result_files import combineCategories


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def testCategoryDefaultAndNamedCategory(self):
        with patch('builtins.input', side_effect=['']):
            self.assertEqual(main.chooseCategories(), main.categoryOptions)
        with patch('builtins.input', side_effect=['invalid', '6']):
            self.assertEqual(main.chooseCategories(), ['zipper'])
        with patch('builtins.input', side_effect=['metal_nut']):
            self.assertEqual(main.chooseCategories(), ['metal_nut'])

    def testGuidedSkipsAndPassesSelectedCategory(self):
        with patch('builtins.input', side_effect=['n', 'y']), patch.object(main, 'runStage') as run:
            main.runStages(main.pipelineStages[:2], ['wood'], guided=True)
            run.assert_called_once_with(main.pipelineStages[1], ['wood'])

    def testFullPipelineOrderAndStopOnFailure(self):
        with patch.object(main, 'runStage') as run:
            main.runStages(main.pipelineStages, ['bottle'])
            self.assertEqual([c.args[0] for c in run.call_args_list], list(main.pipelineStages))
        with patch.object(main, 'runStage', side_effect=RuntimeError('test failure')) as run:
            main.runStages(main.pipelineStages, ['bottle'])
            self.assertEqual(run.call_count, 1)

    def testMenuOnePartAndExit(self):
        with patch('builtins.input', side_effect=['4', '4', 'screw', '0']), patch.object(main, 'runStage') as run:
            main.main()
            run.assert_called_once_with(main.pipelineStages[3], ['screw'])

    def testTrainingMenuRunsOnlyTraining(self):
        with patch('builtins.input', side_effect=['1', 'wood', '0']), patch.object(main, 'runStage') as run:
            main.main()
            run.assert_called_once_with(main.trainingStage, ['wood'])

    def testFullInferenceNeverTrains(self):
        with patch('builtins.input', side_effect=['3', '0', '0']), patch.object(main, 'runStage') as run:
            main.main()
            modules = [c.args[0][1] for c in run.call_args_list]
            self.assertIn('eval', modules)
            self.assertIn('fusion', modules)
            self.assertNotIn('train', modules)
            self.assertNotIn('baseline', modules)
            for call in run.call_args_list:
                self.assertEqual(call.args[1], main.categoryOptions)

    def testMissingPrerequisitesDoNotLoadModel(self):
        with patch.object(main, 'checkInputs', side_effect=FileNotFoundError('missing')), patch.object(main.importlib, 'import_module') as load:
            with self.assertRaises(FileNotFoundError):
                main.runStage(main.pipelineStages[2], ['bottle'])
            load.assert_not_called()

    def testCombinedResultsKeepOtherCategories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for category, value in [('bottle', '0.9'), ('cable', '0.8')]:
                (root / category).mkdir()
                (root / category / 'scores.csv').write_text(f'category,score\n{category},{value}\n')
            combineCategories(root, 'scores.csv', 'scores_all.csv')
            (root / 'bottle/scores.csv').write_text('category,score\nbottle,0.95\n')
            combineCategories(root, 'scores.csv', 'scores_all.csv')
            with (root / 'scores_all.csv').open(newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), [
                    {'category': 'bottle', 'score': '0.95'},
                    {'category': 'cable', 'score': '0.8'},
                ])


if __name__ == '__main__':
    unittest.main()
