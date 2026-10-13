"""Eight-link climbing environment with explicit control-rate metrics."""
from pathlib import Path
from collections import deque
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import gymnasium as gym

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT
from parallel_eight_rod_morphology import EightRodMorphology,NOMINAL_LENGTHS_M,generate_model,GEOM_NAMES
import scene_builder as walls

SCENES=('parallel','tilted','stepped','curved')
SOURCE_XML=PROJECT/'models/parallel_original_8joint_4act_forceclean.xml'


def load_model(path):
    """Load exact MJCF using a virtual filesystem for Unicode Windows paths."""
    path=Path(path)
    assets={p.name:p.read_bytes() for p in path.parent.glob('*.png')}
    return mujoco.MjModel.from_xml_string(path.read_text(encoding='utf-8'),assets=assets)


def save_new(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False)


def bounds(model,data):
    lows=[];highs=[]
    for name in GEOM_NAMES:
        gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
        direction=data.geom_xmat[gid].reshape(3,3)[:,2]
        span=abs(direction[2])*model.geom_size[gid,1]+model.geom_size[gid,0]
        lows.append(data.geom_xpos[gid,2]-span);highs.append(data.geom_xpos[gid,2]+span)
    return min(lows),max(highs)


def build_model(destination,scene,vector,*,physics=None,terrain_start_m=2.1):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    if mujoco.__version__!='2.3.3':raise RuntimeError('Original global-coordinate MJCF requires MuJoCo 2.3.3')
    vector=np.asarray(vector,float)
    if scene not in SCENES or vector.shape!=(8,) or not np.isfinite(vector).all() or np.any(vector<.9) or np.any(vector>1.1):
        raise ValueError('Invalid scene or rod length scales')
    generate_model(SOURCE_XML,destination,EightRodMorphology(*map(float,NOMINAL_LENGTHS_M*vector)),gap_m=1.79)
    original=load_model(destination);data=mujoco.MjData(original);mujoco.mj_forward(original,data)
    bottom,_=bounds(original,data)
    tree=ET.parse(destination);root=tree.getroot();walls._translate_robot_global(root,.03-bottom)
    world=root.find('worldbody');right=world.find("geom[@name='wall_right']");template=dict(right.attrib);world.remove(right)
    face=walls.CORRIDOR_CENTER_X_M+1.79/2
    if scene=='parallel':
        sections=[dict(name='wall_right',start_xz_m=[face,0.],end_xz_m=[face,10.])]
    elif scene=='tilted':
        sections=[dict(name='wall_right',start_xz_m=[face,0.],end_xz_m=[face,2.1]),
                  dict(name='wall_right_tilted',start_xz_m=[face,2.1],end_xz_m=[face-.15,10.])]
    else:sections=walls._wall_sections(scene)
    if not 1.7<=terrain_start_m<=3.:
        raise ValueError('Terrain onset outside the development geometry range')
    if scene!='parallel' and terrain_start_m!=2.1:
        shift=float(terrain_start_m)-2.1
        for section in sections:
            for key in ('start_xz_m','end_xz_m'):
                z=section[key][1]
                if 2.1<=z<10.:
                    section[key][1]=float(z+shift)
    for section in sections:walls._make_wall_box(world,template,section)
    if physics:
        if 'timestep_s' in physics:root.find('option').set('timestep',str(physics['timestep_s']))
        root.find('option').set('iterations',str(physics.get('iterations',100)))
        root.find('option').set('tolerance',str(physics.get('tolerance',1e-10)))
        solref=' '.join(map(str,physics['solref']))
        solimp=' '.join(map(str,physics['solimp']))
        for geom in root.findall('.//geom'):
            geom.set('solref',solref);geom.set('solimp',solimp)
        for equality in root.findall('./equality/connect'):
            equality.set('solref',solref);equality.set('solimp',solimp)
    root.set('model','bom_eight_link_'+scene)
    tree.write(destination,encoding='utf-8',xml_declaration=True)
    model=load_model(destination);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    assert np.allclose(original.body_mass,model.body_mass,rtol=0,atol=1e-9)
    assert (model.nq,model.nv,model.nu)==(10,10,8)
    bottom,top=bounds(model,data)
    record=dict(scene=scene,rod_length_scales=vector.tolist(),rod_lengths_m=(NOMINAL_LENGTHS_M*vector).tolist(),
                bottom_m=bottom,top_m=top,right_wall_sections=sections,
                active_actuator_ids=[0,2,4,6],gravity=model.opt.gravity.tolist(),
                model_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                original_scene_xml_recovered=False,
                initialization='Existing physical robot rod lengths; not manuscript equal-rod 45-degree formula',
                physics_overrides=physics or {},official_result=False)
    record['scene_geometry']=dict(terrain_start_m=float(terrain_start_m))
    save_new(destination.with_suffix('.json'),record)
    return record


class PaperClimbEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,model_path,seed_base=2026090700,*,action_repeat=20,
                 horizon_raw_steps=4500,target_gain_m=1.,discount=.99,reward_height='root'):
        self.model=load_model(model_path)
        self.data=mujoco.MjData(self.model)
        self.active_ids=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in walls.ACTIVE_ACTUATORS])
        assert self.active_ids.tolist()==[0,2,4,6]
        self.rod_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in GEOM_NAMES]
        self.rod_index={g:i for i,g in enumerate(self.rod_ids)}
        self.env_ids={i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]==0}
        self.wall_ids={i for i in self.env_ids if mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i).startswith('wall_')}
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(56,),np.float32)
        self.action_space=gym.spaces.Box(-1.,1.,(4,),np.float32)
        if action_repeat<1 or horizon_raw_steps%action_repeat:
            raise ValueError('Horizon must contain whole positive control steps')
        if not 0<discount<=1 or target_gain_m<=0:
            raise ValueError('Invalid discount or target height')
        self.action_repeat=int(action_repeat);self.horizon_raw_steps=int(horizon_raw_steps)
        self.target_gain_m=float(target_gain_m);self.discount=float(discount)
        if reward_height not in ('root','com'):raise ValueError('Unknown reward height')
        self.reward_height=reward_height
        self.control_dt=float(self.model.opt.timestep)*self.action_repeat
        self.seed_base=int(seed_base);self.episode=0
        self.hold_samples=int(math.ceil(.52/self.control_dt))+1
        self.wall_contact_steps=0;self.control_steps=0;self.history=deque(maxlen=self.hold_samples)
        self.mass_ids=np.flatnonzero(self.model.body_mass>0)
        self.closure_site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'closure_endpoint')
        self.contact_stats={}

    def com_height(self):
        return float(np.average(self.data.xipos[self.mass_ids,2],weights=self.model.body_mass[self.mass_ids]))

    def observe(self):
        theta=np.arctan2(np.sin(self.data.qpos[2:]),np.cos(self.data.qpos[2:]))
        positions=self.data.geom_xpos[self.rod_ids][:,[0,2]].reshape(-1)
        velocity=[];jacp=np.zeros((3,self.model.nv));jacr=np.zeros_like(jacp)
        for gid in self.rod_ids:
            mujoco.mj_jacGeom(self.model,self.data,jacp,jacr,gid)
            velocity.extend((jacp@self.data.qvel)[[0,2]])
        normal=np.zeros(8);weighted_height=np.zeros(8);wall_contact=False;force=np.zeros(6)
        left_contact=right_contact=ground_contact=terrain_contact=False
        max_penetration=0.
        for contact_id in range(self.data.ncon):
            contact=self.data.contact[contact_id];a,b=int(contact.geom1),int(contact.geom2)
            gid=a if a in self.rod_index and b in self.env_ids else b if b in self.rod_index and a in self.env_ids else None
            if gid is None:continue
            max_penetration=max(max_penetration,-float(contact.dist))
            mujoco.mj_contactForce(self.model,self.data,contact_id,force)
            n=max(float(force[0]),0.)
            if n<=1e-6:continue
            i=self.rod_index[gid];normal[i]+=n;weighted_height[i]+=n*float(contact.pos[2])
            wall_contact=wall_contact or a in self.wall_ids or b in self.wall_ids
            other=b if gid==a else a
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,other)
            left_contact=left_contact or name.startswith('wall_left')
            right_contact=right_contact or name.startswith('wall_right')
            ground_contact=ground_contact or name=='floor'
            terrain_contact=terrain_contact or name.startswith(('wall_right_step_','wall_right_curve_','wall_right_tilted'))
        flags=normal>0;heights=np.divide(weighted_height,normal,out=np.zeros(8),where=flags)
        result=np.concatenate((theta,positions,np.array(velocity),flags.astype(float),heights)).astype(np.float32)
        assert result.shape==(56,)
        closure=self.closure_error()
        self.contact_stats=dict(wall_contact=bool(wall_contact),dual_wall_contact=bool(left_contact and right_contact),
                                ground_contact=bool(ground_contact),terrain_contact=bool(terrain_contact),
                                control_sample_penetration_m=max_penetration,closure_error_m=closure)
        return result,wall_contact

    def closure_error(self):
        # The equality anchor on body 0 and the chain endpoint coincide in a closed linkage.
        anchor=self.data.xpos[1]+self.data.xmat[1].reshape(3,3)@self.model.eq_data[0,:3]
        return float(np.linalg.norm(self.data.site_xpos[self.closure_site]-anchor))

    def reset(self,*,seed=None,options=None):
        physical_seed=int(seed) if seed is not None else self.seed_base+self.episode%32
        super().reset(seed=physical_seed);self.episode+=1
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]+=self.np_random.normal(0,.002,self.model.nq)
        self.data.qvel[:]+=self.np_random.normal(0,.0005,self.model.nv)
        mujoco.mj_forward(self.model,self.data)
        self.initial_com=self.com_height();self.initial_root=float(self.data.qpos[1])
        self.raw_step=0;self.control_steps=0;self.wall_contact_steps=0;self.episode_return=0.;self.history.clear()
        self.discounted_return=0.;self.peak_gain=0.;self.max_penetration=0.;self.max_closure=0.
        self.dual_steps=0;self.ground_steps=0;self.terrain_steps=0
        self.episode_seed=physical_seed
        obs,_=self.observe()
        return obs,dict(physical_seed=physical_seed)

    def step(self,action):
        action=np.asarray(action,float)
        if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('Expected finite 4D motor action')
        before=float(self.data.qpos[1]) if self.reward_height=='root' else self.com_height()
        before_time=float(self.data.time)
        warning_before=np.array([w.number for w in self.data.warning],dtype=int)
        self.data.ctrl[:]=0;self.data.ctrl[self.active_ids]=np.clip(action,-1,1)
        mujoco.mj_step(self.model,self.data,nstep=self.action_repeat)
        mujoco.mj_forward(self.model,self.data)
        self.raw_step+=self.action_repeat;self.control_steps+=1
        finite=bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        warning_after=np.array([w.number for w in self.data.warning],dtype=int)
        clock_ok=abs(float(self.data.time)-before_time-self.control_dt)<1e-8
        if not finite or not clock_ok or np.any(warning_after[4:]>warning_before[4:]):
            raise RuntimeError('Simulator numerical failure; inspect state and warnings')
        after=float(self.data.qpos[1]) if self.reward_height=='root' else self.com_height()
        reward=(after-before)/self.control_dt
        obs,wall_contact=self.observe();self.wall_contact_steps+=int(wall_contact)
        gain=self.com_height()-self.initial_com;self.history.append((gain,wall_contact));self.episode_return+=reward
        self.discounted_return+=self.discount**(self.control_steps-1)*reward
        self.peak_gain=max(self.peak_gain,gain)
        self.max_penetration=max(self.max_penetration,self.contact_stats['control_sample_penetration_m'])
        self.max_closure=max(self.max_closure,self.contact_stats['closure_error_m'])
        self.dual_steps+=int(self.contact_stats['dual_wall_contact'])
        self.ground_steps+=int(self.contact_stats['ground_contact'])
        self.terrain_steps+=int(self.contact_stats['terrain_contact'])
        terminated=False;truncated=self.raw_step>=self.horizon_raw_steps
        info=dict(physical_seed=self.episode_seed,com_gain_m=gain,root_gain_m=float(self.data.qpos[1])-self.initial_root,
                  discounted_return=self.discounted_return,**self.contact_stats)
        if truncated:
            info.update(completion_ratio=float(np.clip(gain/self.target_gain_m,0,1)),
                        wall_contact_fraction=self.wall_contact_steps/self.control_steps,
                        dual_wall_contact_fraction=self.dual_steps/self.control_steps,
                        ground_contact_fraction=self.ground_steps/self.control_steps,
                        terrain_contact_fraction=self.terrain_steps/self.control_steps,
                        max_control_sample_penetration_m=self.max_penetration,max_control_sample_closure_m=self.max_closure,
                        peak_com_gain_m=self.peak_gain,
                        episode_return=self.episode_return,
                        sampled_height_hold_success=bool(len(self.history)==self.hold_samples and all(g>=self.target_gain_m and c for g,c in self.history)),
                        hold_metric=f'{self.hold_samples} control samples spanning {(self.hold_samples-1)*self.control_dt:.3f}s')
        return obs,float(reward),terminated,truncated,info

    def close(self):
        pass
