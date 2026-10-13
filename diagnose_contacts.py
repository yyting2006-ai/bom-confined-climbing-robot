"""Numerical contact study using an already saved development action sequence."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    saved=np.load(a.motion,allow_pickle=False)
    actions=saved['actions'];control_dt=float(saved['dt'])
    options=[('package',.002,None,None),('stiff_4ms',.002,.004,[.95,.99,.001]),
             ('stiff_8ms',.002,.008,[.95,.99,.001]),('stiff_8ms_fine',.001,.008,[.95,.99,.001]),
             ('stiff_4ms_fine',.001,.004,[.95,.99,.001])]
    rows=[]
    for label,dt,tc,imp in options:
        root=ET.fromstring(a.source.read_text(encoding='utf-8'))
        root.find('option').set('timestep',str(dt))
        root.find('option').set('iterations','100')
        root.find('option').set('tolerance','1e-10')
        if tc is not None:
            for g in root.findall('.//geom'):
                g.set('solref',f'{tc} 1');g.set('solimp',' '.join(map(str,imp)))
            for c in root.findall('./equality/connect'):
                c.set('solref',f'{tc} 1');c.set('solimp',' '.join(map(str,imp)))
        xml=ET.tostring(root,encoding='unicode')
        (a.output/f'{label}.xml').write_text(xml,encoding='utf-8')
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        d.qpos[:]=saved['qpos'][0];d.qvel[:]=saved['qvel'][0];mujoco.mj_forward(m,d)
        masses=np.flatnonzero(m.body_mass>0)
        initial_com=float(np.average(d.xipos[masses,2],weights=m.body_mass[masses]))
        initial_root=float(d.qpos[1]);max_pen=0.;max_record=None;start=time.perf_counter()
        for step,action in enumerate(actions):
            d.ctrl[:]=0;d.ctrl[[0,2,4,6]]=action
            for raw in range(round(control_dt/dt)):
                mujoco.mj_step(m,d)
                # Contact positions/distances are available at the solver evaluation state.
                for contact in d.contact:
                    g1,g2=int(contact.geom1),int(contact.geom2)
                    if (m.geom_bodyid[g1]==0) == (m.geom_bodyid[g2]==0):continue
                    penetration=max(0.,-float(contact.dist))
                    if penetration>max_pen:
                        max_pen=penetration
                        max_record=dict(control_step=step,raw_step=raw,simulation_time=float(d.time),
                                        geoms=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in [g1,g2]])
            mujoco.mj_forward(m,d)
        row=dict(variant=label,timestep_s=dt,contact_time_constant=tc,contact_impedance=imp,
                 max_raw_solver_penetration_m=max_pen,max_contact=max_record,
                 com_gain_m=float(np.average(d.xipos[masses,2],weights=m.body_mass[masses]))-initial_com,
                 root_gain_m=float(d.qpos[1])-initial_root,
                 warning_counts=[int(w.number) for w in d.warning],elapsed_seconds=time.perf_counter()-start,
                 simulation_time=float(d.time))
        rows.append(row);print(json.dumps(row,allow_nan=False),flush=True)
    report=dict(status='DEVELOPMENT_CONTACT_DIAGNOSTIC',source=str(a.source),motion=str(a.motion),
                purpose='Identical open-loop actions; contact numerical sensitivity, not method performance',rows=rows)
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
