from pathlib import Path
import subprocess, hashlib


def test_panda_overlay_regeneration_is_byte_identical(tmp_path):
    root=Path(__file__).parents[1]; out=root/'assets/industrial/derived/franka_panda_tarzan/panda_grasp_overlay.xml'
    before=out.read_bytes(); subprocess.run(['python3','scripts/build_panda_grasp_overlay.py'],cwd=root,check=True,capture_output=True)
    assert out.read_bytes()==before
    assert hashlib.sha256((root/'assets/industrial/franka_emika_panda/panda.xml').read_bytes()).hexdigest()
