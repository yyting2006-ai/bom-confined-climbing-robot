"""Draw scale-preserving x-z sections directly from compiled MuJoCo geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from paper_env import build_model, load_model
from parallel_eight_rod_morphology import GEOM_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    physics=json.loads((ROOT/'configs/contact_4ms.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,4,figsize=(7.15,4.15),sharex=True,sharey=True)
    records=[]
    for index,(scene,ax) in enumerate(zip(('parallel','tilted','stepped','curved'),axes)):
        xml=a.output/f'{scene}.xml'
        metadata=build_model(xml,scene,[1.]*8,physics=physics)
        model=load_model(xml)
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        walls=[]
        for gid in range(model.ngeom):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid)
            if not name.startswith('wall_'):
                continue
            corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[gid]
            points=(corners@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid])[:,[0,2]]
            points=points[ConvexHull(points).vertices]
            ax.add_patch(Polygon(points,facecolor='#c9cdd2',edgecolor='#6f7781',linewidth=.55))
            walls.append(dict(name=name,projected_polygon_xz_m=points.tolist()))
        ax.axhspan(-.15,0,color='#e6e8eb',zorder=0)
        ax.axhline(0,color='#6f7781',linewidth=.6)
        rods=[]
        for i,name in enumerate(GEOM_NAMES):
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
            center=data.geom_xpos[gid][[0,2]]
            direction=data.geom_xmat[gid].reshape(3,3)[[0,2],2]
            radius,half=model.geom_size[gid,:2]
            ends=np.array([center-half*direction,center+half*direction])
            normal=np.array([-direction[1],direction[0]])*radius
            color=model.geom_rgba[gid,:3]
            ax.add_patch(Polygon([ends[0]+normal,ends[1]+normal,ends[1]-normal,ends[0]-normal],color=color,zorder=3))
            for end in ends:
                ax.add_patch(Circle(end,radius,color=color,zorder=3))
            rods.append(dict(rod=i,endpoints_xz_m=ends.tolist(),radius_m=float(radius)))
        if scene!='parallel':
            ax.axhline(2.1,color='#46586d',linestyle=(0,(3,3)),linewidth=.65)
        ax.set_title(f'({chr(97+index)}) {scene.title()}',fontsize=9.5,pad=8)
        ax.set_xlim(-.35,2.0)
        ax.set_ylim(-.15,4.2)
        ax.set_aspect('equal')
        ax.set_xticks([0,1,2])
        ax.set_yticks([0,1,2,3,4])
        ax.set_xlabel('$x$ (m)',labelpad=2)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=2,pad=2,labelsize=8)
        records.append(dict(scene=scene,model_sha256=metadata['model_sha256'],walls=walls,rods=rods))
    axes[0].set_ylabel('$z$ (m)',labelpad=3)
    fig.subplots_adjust(left=.068,right=.992,bottom=.13,top=.90,wspace=.20)
    fig.savefig(a.output/'environments.pdf',metadata={'Title':'Confined climbing simulation geometry'})
    fig.savefig(a.output/'environments.png',dpi=250)
    plt.close(fig)
    contract=dict(source='compiled nominal MuJoCo model geometry',
                  horizontal_limits_m=[-.35,2.0],vertical_limits_m=[-.15,4.2],
                  aspect='equal',dashed_line='onset of varying right-wall profiles at z=2.1 m',
                  full_wall_top_m=10.,records=records,
                  pdf_sha256=hashlib.sha256((a.output/'environments.pdf').read_bytes()).hexdigest())
    (a.output/'figure_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(a.output))))


if __name__=='__main__':
    main()
