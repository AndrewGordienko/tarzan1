from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'assets/industrial/universal_robots_ur10e/ur10e.xml'; OUTDIR=ROOT/'assets/industrial/derived/ur10e_2f140_proxy'; OUT=OUTDIR/'ur10e_2f140.xml'

def main():
    OUTDIR.mkdir(parents=True,exist_ok=True); xml=SRC.read_text().replace('meshdir="assets"','meshdir="../../universal_robots_ur10e/assets"').replace('<body name="base" quat="0 0 0 -1" childclass="ur10e">','<body name="base" pos="0 0 0.75" quat="0 0 0 -1" childclass="ur10e">')
    marker='<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>'
    tool='''<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>
                  <body name="tool_body" pos="0 0.1 0" quat="-1 1 0 0">
                    <inertial mass="1.025" pos="0 0 0.055" diaginertia="0.003 0.003 0.002"/>
                    <geom name="tool_palm" type="box" pos="0 0 0.035" size="0.045 0.04 0.035" rgba="0.12 0.16 0.19 1"/>
                    <body name="left_jaw"><joint name="left_jaw_joint" type="slide" axis="0 1 0" range="0.005 0.075" damping="15" armature="0.01" solreflimit="0.002 1"/><geom name="left_pad" type="box" pos="0 0 0.11" size="0.025 0.005 0.04" mass="0.02" condim="4" friction="2.0 0.02 0.001" rgba="0.12 0.55 0.75 1"/><geom name="left_support_lip" type="box" pos="0 -0.008 0.148" size="0.025 0.010 0.004" mass="0.01" friction="1.5 0.01 0.001" rgba="0.12 0.55 0.75 1"/></body>
                    <body name="right_jaw"><joint name="right_jaw_joint" type="slide" axis="0 1 0" range="-0.075 -0.005" damping="15" armature="0.01" solreflimit="0.002 1"/><geom name="right_pad" type="box" pos="0 0 0.11" size="0.025 0.005 0.04" mass="0.02" condim="4" friction="2.0 0.02 0.001" rgba="0.12 0.55 0.75 1"/><geom name="right_support_lip" type="box" pos="0 0.008 0.148" size="0.025 0.010 0.004" mass="0.01" friction="1.5 0.01 0.001" rgba="0.12 0.55 0.75 1"/></body>
                    <site name="grasp_site" pos="0 0 0.11" size="0.005" rgba="0 1 0 1"/>
                  </body>'''
    xml=xml.replace(marker,tool).replace('  <actuator>','''  <equality><joint name="jaw_symmetry" joint1="left_jaw_joint" joint2="right_jaw_joint" polycoef="0 -1 0 0 0" solref="0.002 1"/></equality>

  <actuator>''').replace('</actuator>','''  <position name="left_jaw_actuator" joint="left_jaw_joint" kp="3000" ctrlrange="0.005 0.075" forcerange="-62.5 62.5"/>
    <position name="right_jaw_actuator" joint="right_jaw_joint" kp="3000" ctrlrange="-0.075 -0.005" forcerange="-62.5 62.5"/>
  </actuator>''')
    OUT.write_text(xml); files=sorted((ROOT/'assets/industrial/universal_robots_ur10e').rglob('*')); manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}; manifest[str(OUT.relative_to(ROOT))]=hashlib.sha256(OUT.read_bytes()).hexdigest(); (OUTDIR/'manifest.json').write_text(json.dumps({"schema":"ur10e_2f140_proxy_v1","upstream_commit":"71f066ad0be9cd271f7ed58c030243ef157af9f4","files":manifest,"model_label":"simulation-equivalent geometry; not OEM digital twin"},indent=2)+'\n'); print(OUT)
if __name__=='__main__':main()
