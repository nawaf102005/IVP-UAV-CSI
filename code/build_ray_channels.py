"""Independent Sionna RT propagation data for fixed corridor/urban clusters.

The released NPZs permit all policy evaluations without installing Sionna.
Regeneration needs sionna-rt==2.2.0; no measured-channel claim is made.
Training/calibration/test indices are disjoint and frozen before policy tests.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def generate(name,out,seed_offset=0,ntrain=128,ncal=64,ntest=256,all_future=False):
    import sionna.rt as rt
    import mitsuba as mi
    import drjit as dr
    dr.set_thread_count(1)
    scene_name='munich' if name=='munich' else 'simple_street_canyon'
    scene=rt.load_scene(getattr(rt.scene,scene_name),merge_shapes=False)
    boxes=[]
    for shape in scene.mi_scene.shapes():
        bb=shape.bbox();lo=np.array(bb.min).reshape(3);hi=np.array(bb.max).reshape(3)
        if hi[2]>3:boxes.append((lo,hi))
    if name=='munich':
        targets=np.array([[-180,-160],[80,-160],[-180,100],[80,100]])
        options=np.array([[x,y,5.] for x in np.arange(-350,351,10) for y in np.arange(-350,351,10)])
        clear=np.array([p for p in options if all(np.any((p<lo-1)|(p>hi+1)) for lo,hi in boxes)])
        base=np.array([clear[np.argmin(np.linalg.norm(clear[:,:2]-t,axis=1))] for t in targets])
        anchor=base.copy();anchor[:,2]=120.;jitter=np.array([35.,25.,15.])
    else:
        base=np.array([[-73,-45,5],[-23,45,5],[24,-45,5],[74,45,5]],float)
        anchor=np.c_[base[:,0],np.zeros(4),np.full(4,35. if name=='canyon_low' else 80.)]
        jitter=np.array([15.,6.,7.])
    total=ntrain+ncal+ntest
    seed={'canyon_low':960100,'canyon_high':960200,'munich':960300}[name]+seed_offset
    rng=np.random.default_rng(seed)
    positions=anchor[None,:,:]+rng.uniform(-1,1,(total,4,3))*jitter
    # 20 m/s along the corridor for one 20-ms frame; recompute geometry and phase.
    future=positions[0 if all_future else ntrain+ncal:].copy();future[:,:,0]+=.4
    points=np.concatenate([positions.reshape(-1,3),future.reshape(-1,3)])
    scene.frequency=3.5e9
    scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='tr38901',polarization='V')
    scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
    solver=rt.PathSolver(deterministic=True);start=time.perf_counter()
    gain=np.zeros((len(points),4));narrow=np.zeros_like(gain);path_count=np.zeros_like(gain,dtype=int)
    for tx_index,tx_position in enumerate(base):
        tx=rt.Transmitter(name='bs',position=tx_position);scene.add(tx);tx.look_at(anchor[tx_index])
        for start_rx in range(0,len(points),128):
            batch=points[start_rx:start_rx+128]
            for i,p in enumerate(batch):scene.add(rt.Receiver(name=f'uav{i}',position=p))
            paths=solver(scene,max_depth=3,samples_per_src=10000,max_num_paths_per_src=20000,
                         los=True,specular_reflection=True,diffuse_reflection=False,
                         refraction=False,diffraction=True,edge_diffraction=True,seed=seed+tx_index)
            aa,_=paths.cir(out_type='numpy');a=aa[:,0,0,0,:,0]
            sl=slice(start_rx,start_rx+len(batch))
            gain[sl,tx_index]=np.sum(np.abs(a)**2,axis=-1)
            narrow[sl,tx_index]=np.abs(np.sum(a,axis=-1))**2
            path_count[sl,tx_index]=np.count_nonzero(np.abs(a)>0,axis=-1)
            for i in range(len(batch)):scene.remove(f'uav{i}')
            del paths,aa,a
            dr.flush_malloc_cache()
            print(name,tx_index,min(start_rx+128,len(points)),len(points),round(time.perf_counter()-start,2),flush=True)
        scene.remove('bs')
    first=4*total
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'{name}.npz',base=base,anchor=anchor,positions=positions,future_positions=future,
                        incoherent=gain[:first].reshape(total,4,4),coherent=narrow[:first].reshape(total,4,4),
                        incoherent_future=gain[first:].reshape(len(future),4,4),coherent_future=narrow[first:].reshape(len(future),4,4),
                        path_count=path_count[:first].reshape(total,4,4),building_boxes=np.array(boxes))
    scene_root=Path(getattr(rt.scene,scene_name)).parent
    hashes={str(p.relative_to(scene_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(scene_root.rglob('*')) if p.is_file()}
    meta=dict(engine='Sionna RT 2.2.0',variant=mi.variant(),scene=scene_name,condition=name,seed=seed,
              train=ntrain,calibration=ncal,test=ntest,split='first train; next calibration; remaining test',
              carrier_hz=3.5e9,max_depth=3,rays_per_source=10000,bs_antenna='single tr38901 element, fixed boresight to own anchor',
              uav_antenna='single isotropic element',polarization='vertical',
              propagation='LoS, specular reflection, diffraction and edge diffraction; no diffuse or refraction',
              static_gain='incoherent path-power sum',motion_gain='coherent baseband path sum, geometry recomputed after 0.4 m',
              velocity_m_s=20.,frame_s=.020,all_future=all_future,runtime_s=time.perf_counter()-start,
              zero_power_links=int((gain<=0).sum()),zero_coherent_links=int((narrow<=0).sum()),
              log_floor_gain=1e-20,scene_asset_sha256=hashes)
    (out/f'{name}.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['canyon_low','canyon_high','munich','all'],default='all')
    p.add_argument('--output',default='../ray_data')
    p.add_argument('--fresh-test',action='store_true',help='New seed block, 256 test frames; reuse separately frozen training/calibration.')
    p.add_argument('--paired-future',action='store_true',help='New seed block (+17000), with initial/endpoint pairs for all training, calibration and test frames.')
    a=p.parse_args()
    if a.fresh_test and a.paired_future:p.error('Choose only one regeneration protocol')
    for name in (['canyon_low','canyon_high','munich'] if a.scene=='all' else [a.scene]):
        generate(name,Path(a.output),17000 if a.paired_future else 7000 if a.fresh_test else 0,0 if a.fresh_test else 128,0 if a.fresh_test else 64,all_future=a.paired_future)
