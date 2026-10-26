"""Root API compatibility and import boundaries used by desktop startup."""

import importlib
import pickle
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
# This is the existing public contract, including exports supplied by packages
# such as io. Resolving an export must return the original object, not a proxy.
PUBLIC_MODULES = {
    "mlcookbook.core.dataset": ("Dataset",),
    "mlcookbook.core.schema": ("VarType",),
    "mlcookbook.preprocessing.config": ("PreprocessingConfig",),
    "mlcookbook.io": ("load_dataset",),
    "mlcookbook.statistics.association": ("CorrelationResult", "correlation"),
    "mlcookbook.statistics.categorical": ("ChiSquareResult", "chi_square"),
    "mlcookbook.statistics.groups": ("AnovaResult", "anova"),
    "mlcookbook.statistics.scan": ("AssociationScanResult", "association_scan"),
    "mlcookbook.regression.linear": ("LinearRegressionResult", "linear_regression"),
    "mlcookbook.regression.tree": ("RegressionTreeResult", "decision_tree_regression"),
    "mlcookbook.regression.forest": ("RandomForestRegressionResult", "random_forest_regression"),
    "mlcookbook.regression.svm": ("SVMRegressionResult", "svm_regression"),
    "mlcookbook.regression.neural_network": ("NeuralNetworkRegressionResult", "neural_network_regression"),
    "mlcookbook.classification.logistic": ("LogisticRegressionResult", "logistic_regression"),
    "mlcookbook.classification.tree": ("DecisionTreeResult", "decision_tree"),
    "mlcookbook.classification.forest": ("RandomForestResult", "random_forest"),
    "mlcookbook.classification.svm": ("SVMResult", "svm"),
    "mlcookbook.classification.neural_network": ("NeuralNetworkResult", "neural_network"),
    "mlcookbook.clustering.kmeans": ("KMeansResult", "kmeans"),
    "mlcookbook.dimensionality.pca": ("PCAResult", "pca"),
    "mlcookbook.comparison": ("RegressionComparisonResult", "compare_regressors"),
    "mlcookbook.llm": ("TextGenerationResult", "generate_text"),
    "mlcookbook.wearable": ("WearableEvaluationResult", "WearableImportReport", "evaluate_wearable",
                            "load_wearable_study"),
}
PUBLIC_NAMES = {name for names in PUBLIC_MODULES.values() for name in names}


def _run(code):
    completed = subprocess.run([sys.executable, "-c", dedent(code)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_package_import_and_discovery_do_not_initialize_analyses_or_optional_runtimes():
    _run("""
        import sys
        import mlcookbook

        assert mlcookbook.__version__ == '1.0.0'
        assert set(mlcookbook.__all__) <= set(dir(mlcookbook))
        assert not any(name in vars(mlcookbook) for name in mlcookbook.__all__)
        assert not hasattr(mlcookbook, 'unknown_recipe')
        try:
            mlcookbook.unknown_recipe
        except AttributeError as error:
            assert 'mlcookbook' in str(error) and 'unknown_recipe' in str(error)
        else:
            raise AssertionError('Unknown exports must raise AttributeError')
        blocked = ('numpy', 'pandas', 'scipy', 'sklearn', 'pyarrow', 'torch', 'transformers', 'PySide6')
        assert not any(name == prefix or name.startswith(prefix + '.')
                       for name in sys.modules for prefix in blocked)
        assert not any(name.startswith('mlcookbook.') for name in sys.modules)
    """)


def test_dataset_import_loads_its_existing_core_without_other_recipe_or_runtime_imports():
    _run("""
        import sys
        from mlcookbook import Dataset
        from mlcookbook.core.dataset import Dataset as OriginalDataset
        import mlcookbook

        assert Dataset is OriginalDataset
        assert mlcookbook.Dataset is Dataset
        assert 'pandas' in sys.modules
        blocked = ('scipy', 'sklearn', 'torch', 'transformers', 'PySide6', 'mlcookbook.classification',
                   'mlcookbook.regression', 'mlcookbook.wearable', 'mlcookbook.llm')
        assert not any(name == prefix or name.startswith(prefix + '.')
                       for name in sys.modules for prefix in blocked)
    """)


def test_every_public_export_and_wildcard_import_keep_the_original_objects():
    import mlcookbook

    assert len(PUBLIC_NAMES) == 44
    assert len(mlcookbook.__all__) == len(PUBLIC_NAMES)
    assert set(mlcookbook.__all__) == PUBLIC_NAMES
    namespace = {}
    exec("from mlcookbook import *", namespace)
    assert set(namespace) - {"__builtins__"} == PUBLIC_NAMES
    for module_name, names in PUBLIC_MODULES.items():
        module = importlib.import_module(module_name)
        for name in names:
            original = getattr(module, name)
            assert getattr(mlcookbook, name) is original, name
            assert namespace[name] is original, name
            assert pickle.loads(pickle.dumps(original)) is original, name


def test_public_classifier_still_runs_and_serializes_without_loading_optional_runtimes():
    _run("""
        import json
        import sys
        import pandas as pd
        from mlcookbook import Dataset, LogisticRegressionResult, logistic_regression

        frame = pd.DataFrame({'x': list(range(80)), 'label': ['low'] * 40 + ['high'] * 40})
        result = logistic_regression(Dataset(frame), target='label', cv_folds=0)
        assert isinstance(result, LogisticRegressionResult)
        assert result.test_metrics.n_obs == 20
        assert result.test_metrics.accuracy >= .9
        json.dumps(result.to_dict())
        assert not any(name == prefix or name.startswith(prefix + '.')
                       for name in sys.modules for prefix in ('torch', 'transformers', 'PySide6'))
    """)
