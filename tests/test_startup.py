"""Desktop startup paints first and initializes analytical work without blocking Qt."""

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_single_entry_point_paints_and_handles_events_without_scientific_imports():
    # conftest imports numpy/pandas for core fixtures, so the import boundary
    # must be checked in a genuinely fresh process rather than this test runner.
    code = textwrap.dedent("""
        import importlib.abc
        import runpy
        import sys

        blocked = ('numpy', 'pandas', 'scipy', 'sklearn', 'pyarrow', 'torch',
                   'transformers', 'mlcookbook.core')
        class ForbidScientificImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
                    raise AssertionError('Scientific import before shell paint: ' + fullname)
        sys.meta_path.insert(0, ForbidScientificImports())
        import app
        from PySide6.QtCore import QEvent, QObject, QTimer

        state = {'paint': False, 'heartbeat': False, 'finished': False}
        keep_alive = []
        def check_and_quit():
            if not (state['paint'] and state['heartbeat']) or state['finished']:
                return
            state['finished'] = True
            try:
                controller = state['controller']
                assert controller.key == 'data' and controller.window.isVisible()
                assert controller.screen.view is None and controller.job is None
                controller.show_front_page()
                assert controller.key == 'recipes'
                controller.show_data_page()
                assert controller.key == 'data'
                assert not any(module == name or module.startswith(name + '.')
                               for module in sys.modules for name in blocked)
                print('shell painted; timer responded; scientific imports absent')
                app.QApplication.instance().quit()
            except BaseException:
                import traceback
                traceback.print_exc()
                app.QApplication.instance().exit(7)
        class ObservePaint(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint:
                    state['paint'] = True
                    QTimer.singleShot(0, check_and_quit)
                return False
        def startup(controller, options):
            state['controller'] = controller
            observer = ObservePaint()
            keep_alive.append(observer)
            controller.screen.installEventFilter(observer)
            controller.screen.update()
            def heartbeat():
                state['heartbeat'] = True
                check_and_quit()
            QTimer.singleShot(0, heartbeat)
            QTimer.singleShot(10000, lambda: app.QApplication.instance().exit(8))
        app._open_startup = startup
        sys.argv = ['mlcookbook']
        try:
            runpy.run_module('mlcookbook', run_name='__main__')
        except SystemExit as exit:
            assert exit.code == 0, exit.code
        assert state['finished'] and state['paint'] and state['heartbeat']
    """)
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "shell painted; timer responded; scientific imports absent" in completed.stdout


def test_real_first_recipe_loads_scientific_modules_off_the_gui_thread_without_optional_inference():
    code = textwrap.dedent("""
        import importlib.abc
        import runpy
        import sys
        import threading

        gui_thread = threading.get_ident()
        scientific = ('numpy', 'pandas', 'scipy', 'sklearn', 'pyarrow', 'mlcookbook.core')
        optional = ('torch', 'transformers')
        imports = []
        class ObserveImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in optional):
                    raise AssertionError('Unrequested inference import: ' + fullname)
                if any(fullname == name or fullname.startswith(name + '.') for name in scientific):
                    imports.append((fullname, threading.get_ident()))
        sys.meta_path.insert(0, ObserveImports())
        import app
        from PySide6.QtCore import QEvent, QObject, QTimer
        from gui.recipes import RECIPES

        state = {'painted': False, 'started': False, 'finished': False}
        keep_alive = []
        def finish():
            controller = state['controller']
            if controller.job is not None:
                QTimer.singleShot(1, finish)
                return
            try:
                assert state['painted'] and controller.key == 'linear-regression'
                assert controller._runtime_initialized and controller.screen._report == {}
                assert controller.screen.view['rows'] == 0
                assert imports and all(thread != gui_thread for _, thread in imports), imports
                assert not any(module == name or module.startswith(name + '.')
                               for module in sys.modules for name in optional)
                state['finished'] = True
                print('real first recipe initialized in worker; no inference imports or analysis')
                app.QApplication.instance().quit()
            except BaseException:
                import traceback
                traceback.print_exc()
                app.QApplication.instance().exit(7)
        def open_recipe():
            controller = state['controller']
            controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == 'linear-regression'))
            QTimer.singleShot(1, finish)
        class ObservePaint(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint and not state['started']:
                    state['painted'] = state['started'] = True
                    QTimer.singleShot(0, open_recipe)
                return False
        def startup(controller, options):
            state['controller'] = controller
            observer = ObservePaint()
            keep_alive.append(observer)
            controller.screen.installEventFilter(observer)
            controller.screen.update()
            QTimer.singleShot(20000, lambda: app.QApplication.instance().exit(8))
        app._open_startup = startup
        sys.argv = ['mlcookbook']
        try:
            runpy.run_module('mlcookbook', run_name='__main__')
        except SystemExit as exit:
            assert exit.code == 0, exit.code
        assert state['finished']
    """)
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "real first recipe initialized in worker; no inference imports or analysis" in completed.stdout


def test_real_main_retires_its_window_while_qapplication_is_still_alive():
    code = textwrap.dedent("""
        import threading
        import weakref
        import app
        from PySide6.QtCore import QCoreApplication, QThread, QTimer
        from PySide6.QtWidgets import QWidget
        from shiboken6 import isValid

        gui_thread = threading.get_ident()
        state = {}
        def startup(controller, options):
            assert controller.key == 'data' and controller.window.isVisible()
            state['window'] = weakref.ref(controller.window)
            # An analysis candidate can be constructed but not yet attached.
            # Keep its wrapper strongly referenced to prove explicit Qt cleanup.
            state['hidden'] = QWidget()
            assert state['hidden'].parent() is None and not state['hidden'].isVisible()
            def hidden_destroyed():
                state['hidden_destroyed'] = True
                state['hidden_destroyed_on'] = threading.get_ident()
                state['hidden_application_alive'] = (app.QApplication.instance() is not None
                                                     and not QCoreApplication.closingDown())
            state['hidden'].destroyed.connect(hidden_destroyed)
            def destroyed():
                state['destroyed'] = True
                state['destroyed_on'] = threading.get_ident()
                instance = app.QApplication.instance()
                state['application_alive'] = instance is not None and not QCoreApplication.closingDown()
                state['gui_affinity'] = instance is not None and QThread.currentThread() == instance.thread()
            controller.window.destroyed.connect(destroyed)
            QTimer.singleShot(0, controller.window.close)
        app._open_startup = startup
        result = app.main([])
        assert result == 0
        assert state.get('destroyed') and state['destroyed_on'] == gui_thread
        assert state['application_alive'] and state['gui_affinity']
        assert state['window']() is None or not isValid(state['window']())
        assert state.get('hidden_destroyed') and state['hidden_destroyed_on'] == gui_thread
        assert state['hidden_application_alive'] and not isValid(state['hidden'])
        print('real main returned after GUI-thread window deletion with QApplication alive')
    """)
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "real main returned after GUI-thread window deletion with QApplication alive" in completed.stdout


def test_real_main_close_during_cold_initialization_waits_for_worker_then_deletes_window():
    code = textwrap.dedent("""
        import threading
        import app
        from PySide6.QtCore import QCoreApplication, QTimer
        from gui.recipes import RECIPES

        entered, closing, release, finished = (threading.Event() for _ in range(4))
        gui_thread = threading.get_ident()
        state = {}
        original_prepare = app._prepare_runtime
        def prepare():
            state['prepared_on'] = threading.get_ident()
            entered.set()
            if not release.wait(15):
                raise TimeoutError('The test did not release cold initialization')
            original_prepare()
            finished.set()
        app._prepare_runtime = prepare
        def release_after_close():
            if closing.wait(15):
                release.set()
        releaser = threading.Thread(target=release_after_close)
        releaser.start()
        def startup(controller, options):
            assert not controller._runtime_initialized
            def destroyed():
                state['worker_finished_before_window_deletion'] = finished.is_set()
                state['window_deleted_on'] = threading.get_ident()
                state['application_alive'] = app.QApplication.instance() is not None and not QCoreApplication.closingDown()
            controller.window.destroyed.connect(destroyed)
            controller.open_recipe(next(recipe for recipe in RECIPES if recipe.key == 'linear-regression'))
            def close_after_worker_entered():
                if not entered.is_set():
                    QTimer.singleShot(1, close_after_worker_entered)
                    return
                assert controller.job is not None and not finished.is_set()
                closing.set()
                controller.window.close()
            QTimer.singleShot(0, close_after_worker_entered)
            QTimer.singleShot(20000, lambda: app.QApplication.instance().exit(8))
        app._open_startup = startup
        try:
            result = app.main([])
        finally:
            release.set()
            releaser.join(5)
        assert result == 0
        assert entered.is_set() and closing.is_set() and finished.is_set()
        assert state['prepared_on'] != gui_thread
        assert state.get('worker_finished_before_window_deletion')
        assert state['window_deleted_on'] == gui_thread and state['application_alive']
        assert not releaser.is_alive()
        print('real main waited for cold worker and retired window before QApplication shutdown')
    """)
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=45)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "real main waited for cold worker and retired window before QApplication shutdown" in completed.stdout


@pytest.fixture(scope="module")
def qt():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qt):
    from PySide6.QtCore import QCoreApplication, QEvent

    import app

    controller = app.MLCookbookApp()
    controller._runtime_initialized = False
    controller.window.show()
    yield controller
    controller.wait_for_job()
    qt.processEvents()
    controller.window.close()
    controller.window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _finish(controller, qt):
    deadline = time.monotonic() + 30
    while controller.job is not None and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(.005)
    qt.processEvents()
    assert controller.job is None


def _recipe(key):
    from gui.recipes import RECIPES

    return next(recipe for recipe in RECIPES if recipe.key == key)


def test_runtime_initialization_retains_the_view_and_keeps_qt_responsive(controller, qt, monkeypatch):
    from PySide6.QtCore import QThread, QTimer

    import app

    entered, release = threading.Event(), threading.Event()
    worker_threads, factory_threads, heartbeat = [], [], []
    previous = controller.screen
    previous.markers.setText("?, -999")
    factory = app.SCREEN_FACTORIES["linear-regression"]

    def prepare():
        worker_threads.append(QThread.currentThread())
        entered.set()
        if not release.wait(10):
            raise TimeoutError("The test did not release initialization")

    def create(*args, **kwargs):
        factory_threads.append(QThread.currentThread())
        return factory(*args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("Opening an unconfigured recipe must not load examples or train")

    monkeypatch.setattr(app, "_prepare_runtime", prepare)
    monkeypatch.setitem(app.SCREEN_FACTORIES, "linear-regression", create)
    monkeypatch.setattr(app, "_run_report", unexpected)
    _, name, target, positive = app.DEMOS["linear-regression"]
    monkeypatch.setitem(app.DEMOS, "linear-regression", (unexpected, name, target, positive))
    try:
        controller.open_recipe(_recipe("linear-regression"))
        assert entered.wait(2)
        assert controller.screen is previous and controller.key == "data"
        assert not controller._runtime_initialized
        assert not factory_threads
        assert "Preparing analysis runtime" in previous.status.text()
        assert not previous.load_button.isEnabled()
        assert all(not button.isEnabled() for button in controller.window.navigation_buttons.values())
        assert not controller.window.open_experiment_button.isEnabled()
        assert not controller.window.save_experiment_button.isEnabled()
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        qt.processEvents()
        assert heartbeat == [True]
        assert controller.job is not None
    finally:
        release.set()
        _finish(controller, qt)
    assert worker_threads and all(thread != qt.thread() for thread in worker_threads)
    assert factory_threads == [qt.thread()]
    assert controller._runtime_initialized
    assert controller.key == "linear-regression" and controller.screen is not previous
    assert not controller.screen._report and not controller.screen.run_button.isEnabled()
    assert controller.data_options["markers"] == "?, -999"
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())
    assert controller.window.open_experiment_button.isEnabled()


def test_failed_runtime_initialization_preserves_work_and_can_be_retried(controller, qt, monkeypatch):
    import app

    previous = controller.screen
    previous.markers.setText("unknown")
    attempts = []

    def prepare():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("Runtime initialization could not complete")

    monkeypatch.setattr(app, "_prepare_runtime", prepare)
    controller.open_recipe(_recipe("linear-regression"))
    _finish(controller, qt)
    assert controller.screen is previous and controller.key == "data"
    assert previous.markers.text() == "unknown"
    assert not controller._runtime_initialized
    assert "Runtime initialization could not complete" in previous.status.text()
    assert previous.load_button.isEnabled()
    assert all(button.isEnabled() for button in controller.window.navigation_buttons.values())
    controller.open_recipe(_recipe("linear-regression"))
    _finish(controller, qt)
    assert len(attempts) == 2 and controller._runtime_initialized
    assert controller.key == "linear-regression"
    assert not controller.screen._report


@pytest.mark.parametrize("fail", [False, True], ids=["ready", "initialization-failed"])
def test_study_startup_waits_for_recipe_initialization_without_running_analysis(controller, qt, monkeypatch,
                                                                              tmp_path, fail):
    from PySide6.QtCore import QThread

    import app

    entered, release = threading.Event(), threading.Event()
    loaded = []
    manifest = tmp_path / "study with spaces" / "export.json"

    def prepare():
        entered.set()
        if not release.wait(10):
            raise TimeoutError("The test did not release initialization")
        if fail:
            raise RuntimeError("Cannot initialize study runtime")

    def load(path):
        loaded.append((path, controller.key, QThread.currentThread()))

    def unexpected(*args, **kwargs):
        pytest.fail("Startup handoffs must not train or unseal the test")

    monkeypatch.setattr(app, "_prepare_runtime", prepare)
    monkeypatch.setattr(controller, "load_wearable_study", load)
    monkeypatch.setattr(app, "_run_report", unexpected)
    options, qt_arguments = app._startup_options(["--study", str(manifest), "-platform", "windows"])
    assert qt_arguments == ["-platform", "windows"]
    try:
        app._open_startup(controller, options)
        assert entered.wait(2)
        qt.processEvents()
        assert not loaded and controller.key == "data"
    finally:
        release.set()
        _finish(controller, qt)
    if fail:
        assert loaded == [] and controller.key == "data"
        assert not controller._runtime_initialized
        assert "Cannot initialize study runtime" in controller.screen.status.text()
    else:
        assert loaded == [(str(manifest.resolve()), "wearable-study", qt.thread())]
        assert controller.screen.request()["params"]["final_test"] is False
        assert controller.screen._report == {}
        assert controller.imported is None


@pytest.mark.parametrize("action,module", [("csv", "mlcookbook.io"), ("experiment", "mlcookbook.experiments")])
def test_first_file_action_imports_its_core_in_the_worker(controller, qt, monkeypatch, tmp_path, action, module):
    import builtins

    from PySide6.QtCore import QThread

    imported_on = []
    original_import = builtins.__import__

    def observe_import(name, *args, **kwargs):
        if name == module:
            imported_on.append(QThread.currentThread())
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observe_import)
    if action == "csv":
        path = tmp_path / "own measurements.csv"
        path.write_text("identifier,value\n0001,2.5\n0002,3.5\n", encoding="utf-8")
        controller.load_file(str(path), ",", "utf-8-sig")
    else:
        controller.open_experiment(str(tmp_path / "missing.mlcb"))
    _finish(controller, qt)
    assert imported_on and all(thread != qt.thread() for thread in imported_on)
    assert controller.key == "data" and controller.screen.load_button.isEnabled()
    if action == "csv":
        assert controller.imported is not None and controller._runtime_initialized
        assert controller.imported[0].df["identifier"].tolist() == ["0001", "0002"]
    else:
        assert controller.imported is None
        assert "Unable to complete" in controller.screen.status.text()
    assert controller.window.open_experiment_button.isEnabled()
