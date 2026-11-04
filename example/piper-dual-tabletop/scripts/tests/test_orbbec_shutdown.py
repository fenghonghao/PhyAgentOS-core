"""Exercise a full image queue with no reader, without opening a camera."""
import importlib.util
import multiprocessing as mp
from pathlib import Path


PATCH_SCRIPT = Path(__file__).parents[1] / "patch_orbbec_shutdown.py"
spec = importlib.util.spec_from_file_location("patch_orbbec_shutdown", PATCH_SCRIPT)
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

WORKER = '''def worker(frame_queue, ready, stop):
    """Run pyorbbecsdk in a clean child process with no Dora imports."""
    frame_queue.put(b"x" * (2 * 1024 * 1024))
    ready.set()
    stop.wait(5)
'''


def exit_without_reader(source):
    namespace = {}
    exec(compile(source, "worker.py", "exec"), namespace)
    ctx = mp.get_context("fork")
    frames = ctx.Queue(maxsize=2)
    ready, stop = ctx.Event(), ctx.Event()
    process = ctx.Process(target=namespace["worker"], args=(frames, ready, stop))
    process.start()
    try:
        assert ready.wait(5), "worker did not enqueue the image"
        stop.set()
        process.join(timeout=1)
        return process.exitcode
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        frames.close()


def test_original_worker_hangs_when_parent_stops_reading():
    assert exit_without_reader(WORKER) is None


def test_patched_worker_exits_normally_and_patch_is_idempotent(tmp_path):
    target = tmp_path / "forge_devices_orbbec_camera" / "isolated_backend.py"
    target.parent.mkdir()
    target.write_text(WORKER)
    patcher.patch(tmp_path)
    patched = target.read_text()
    patcher.patch(tmp_path)
    assert target.read_text() == patched
    assert exit_without_reader(patched) == 0
