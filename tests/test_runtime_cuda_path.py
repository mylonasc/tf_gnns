import types

from tf_gnns.commons.tensorflow import runtime_cuda_path as rcp


def test_major_from_version_string_parses_and_rejects():
    assert rcp._major_from_version_string("12.4") == 12
    assert rcp._major_from_version_string("CUDA 13.0") == 13
    assert rcp._major_from_version_string("unknown") is None


def test_detect_system_cuda_major_from_nvidia_smi(monkeypatch):
    def fake_run(cmd, capture_output, text, check):
        del capture_output, text, check
        if cmd == ["nvidia-smi"]:
            return types.SimpleNamespace(stdout="CUDA Version: 13.1", stderr="")
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(rcp.subprocess, "run", fake_run)
    assert rcp._detect_system_cuda_major() == 13


def test_detect_system_cuda_major_falls_back_to_nvcc(monkeypatch):
    def fake_run(cmd, capture_output, text, check):
        del capture_output, text, check
        if cmd == ["nvidia-smi"]:
            raise RuntimeError("missing")
        if cmd == ["nvcc", "--version"]:
            return types.SimpleNamespace(stdout="Cuda compilation tools, release 12.8", stderr="")
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(rcp.subprocess, "run", fake_run)
    assert rcp._detect_system_cuda_major() == 12


def test_discover_pip_nvidia_lib_paths_deduplicates(monkeypatch):
    monkeypatch.setattr(rcp.site, "getsitepackages", lambda: ["/a", "/b"])

    def fake_glob(pattern):
        if pattern.startswith("/a"):
            return ["/a/nvidia/cublas/lib", "/a/nvidia/cudnn/lib"]
        return ["/a/nvidia/cublas/lib", "/b/nvidia/cusolver/lib"]

    monkeypatch.setattr(rcp.glob, "glob", fake_glob)
    assert rcp._discover_pip_nvidia_lib_paths() == [
        "/a/nvidia/cublas/lib",
        "/a/nvidia/cudnn/lib",
        "/b/nvidia/cusolver/lib",
    ]


def test_patch_ld_library_path_success(monkeypatch):
    monkeypatch.setattr(rcp, "_PATCH_ALREADY_RUN", False)
    monkeypatch.delenv(rcp._PATCH_MARKER_ENV, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib:/opt/lib")
    monkeypatch.setattr(rcp, "_detect_system_cuda_major", lambda: 13)
    monkeypatch.setattr(rcp, "_detect_tf_build_cuda_major", lambda: 12)
    monkeypatch.setattr(rcp, "_discover_pip_nvidia_lib_paths", lambda: ["/x/lib", "/y/lib"])

    changed = rcp.maybe_patch_ld_library_path_for_tensorflow()
    assert changed is True
    assert rcp._PATCH_MARKER_ENV in rcp.os.environ
    assert rcp.os.environ["LD_LIBRARY_PATH"] == "/x/lib:/y/lib:/usr/lib:/opt/lib"


def test_patch_ld_library_path_noop_paths(monkeypatch):
    monkeypatch.setattr(rcp, "_PATCH_ALREADY_RUN", False)
    monkeypatch.delenv(rcp._PATCH_MARKER_ENV, raising=False)
    monkeypatch.setattr(rcp, "_detect_system_cuda_major", lambda: 12)
    monkeypatch.setattr(rcp, "_detect_tf_build_cuda_major", lambda: 12)
    monkeypatch.setattr(rcp, "_discover_pip_nvidia_lib_paths", lambda: ["/x/lib"])
    assert rcp.maybe_patch_ld_library_path_for_tensorflow() is False


def test_patch_ld_library_path_idempotent(monkeypatch):
    monkeypatch.setattr(rcp, "_PATCH_ALREADY_RUN", False)
    monkeypatch.setenv(rcp._PATCH_MARKER_ENV, "1")
    assert rcp.maybe_patch_ld_library_path_for_tensorflow() is False
