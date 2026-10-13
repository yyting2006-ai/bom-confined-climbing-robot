"""Render recorded MuJoCo states at their original control time scale."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from paper_env import load_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=720)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    motion = np.load(a.motion, allow_pickle=False)
    trace = json.loads(a.trace.read_text(encoding='utf-8'))
    dt = float(motion['dt'])
    qpos, qvel = motion['qpos'], motion['qvel']
    assert len(qpos) == len(trace) + 1 == len(qvel)
    assert abs(len(trace) * dt - trace[-1]['time_s']) < 1e-9
    original = load_model(a.model)
    visual_tree = ET.parse(a.model)
    visual_root = visual_tree.getroot()
    assets = visual_root.find('asset')
    if assets is None:
        assets = ET.SubElement(visual_root, 'asset')
    ET.SubElement(assets, 'texture', type='skybox', builtin='gradient',
                  rgb1='1 1 1', rgb2='1 1 1', width='256', height='256')
    visual_text = ET.tostring(visual_root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(visual_text)
    for field in ('qpos0','body_mass','body_inertia','geom_pos','geom_quat',
                  'geom_size','jnt_pos','actuator_gear','eq_data'):
        np.testing.assert_array_equal(getattr(model,field),getattr(original,field))
    model.vis.global_.offwidth = a.width
    model.vis.global_.offheight = a.height
    model.vis.global_.fovy = 3.5
    model.vis.headlight.ambient[:] = .45
    model.vis.headlight.diffuse[:] = .55
    model.vis.map.fogstart = 1000.
    model.vis.map.fogend = 1001.
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.82, 0, 2.4]
    camera.distance = 100
    camera.azimuth = -90
    camera.elevation = 0
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    renderer = mujoco.Renderer(model, height=a.height, width=a.width)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    selected = set(np.linspace(0, len(qpos)-1, 6, dtype=int).tolist())
    contact_sheet = []
    mp4 = a.output / 'trajectory.mp4'
    try:
        with imageio.get_writer(mp4, fps=1/dt, codec='libx264', quality=8, macro_block_size=None) as writer:
            for i, (q, v) in enumerate(zip(qpos, qvel)):
                data.qpos[:] = q
                data.qvel[:] = v
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, a.width, 88), fill=(248, 249, 251))
                draw.text((18, 10), a.label, font=font, fill=(22, 35, 50))
                gain = trace[i-1]['com_gain_m'] if i else 0.
                draw.text((18, 45), f't = {i*dt:.2f} s    COM gain = {gain:.3f} m', font=font, fill=(22,35,50))
                writer.append_data(np.asarray(frame))
                if i in selected:
                    frame.save(a.output / f'frame_{i:04d}.png')
                    contact_sheet.append(frame.resize((480,360)))
    finally:
        renderer._mjr_context.free()
        if renderer._gl_context is not None:
            renderer._gl_context.free()
    sheet = Image.new('RGB', (1440,720), 'white')
    for i, frame in enumerate(contact_sheet):
        sheet.paste(frame, ((i%3)*480, (i//3)*360))
    sheet.save(a.output / 'contact_sheet.png')
    record = dict(model=str(a.model.resolve()), model_sha256=sha(a.model),
                  motion=str(a.motion.resolve()), motion_sha256=sha(a.motion),
                  trace=str(a.trace.resolve()), trace_sha256=sha(a.trace),
                  renderer='MuJoCo saved generalized states; white skybox and near-orthographic distant camera',
                  visual_xml_sha256=hashlib.sha256(visual_text.encode('utf-8')).hexdigest(),
                  geometry_invariance_checked=True,vertical_field_of_view_deg=3.5,
                  fog_disabled_for_distant_camera=True,
                  video_sha256=sha(mp4), frames=len(qpos), fps=1/dt,
                  simulated_duration_s=len(trace)*dt,
                  encoded_duration_s=len(qpos)*dt,
                  extra_endpoint_frame_s=dt, camera=dict(lookat=camera.lookat.tolist(),
                  distance=camera.distance, azimuth=camera.azimuth,elevation=camera.elevation),
                  label=a.label, outcome=trace[-1])
    (a.output / 'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),frames=len(qpos),fps=1/dt)))


if __name__ == '__main__':
    main()
