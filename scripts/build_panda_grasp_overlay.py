from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; src=ROOT/'assets/industrial/franka_emika_panda/panda.xml'; out=ROOT/'assets/industrial/franka_emika_panda/panda_grasp_overlay.xml'
xml=src.read_text(); marker='<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'; site='<site name="grasp_site" pos="0 0 0.1029" quat="1 0 0 0"/>'
out.write_text(xml.replace(marker,marker+site)); print(out)
