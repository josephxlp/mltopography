from params import define_parameters
from uvars import model_dir, pq_path, model_pattern, WDIR,tsp_tilenames,yaml_pattern
from os import listdir 
from glob import glob 
import time 
from utilspred import prediction_workflow
from concurrent.futures import ProcessPoolExecutor 
# make this parallel

print(tsp_tilenames)
tilename = "N13E103"
tilenames = [tilename]
tilenamex = sorted(listdir(WDIR))

rnd_seed,roi,mx,fcolydi,fcolyid, fcolref, fcolX,catboost_params, nboost,fcolY,FTCOLSC = define_parameters()
keys_to_keep = FTCOLSC 
nboost = 1000 + 100
tvar = "all"#roi
seed = 42#42, 123
mfiles = glob(model_pattern)
modelpaths = [i for i in mfiles if f'_{nboost}_' in i];print(len(modelpaths))
modelpaths = [i for i in mfiles if f'_{seed}_' in i];print(len(modelpaths))
modelpaths = [i for i in modelpaths if str(tvar) in i];print(len(modelpaths))
#modelpaths
assert len(modelpaths) == 2, 'Wrong models'

if __name__ == '__main__':
    ta = time.perf_counter()
    tilenames = tilenames#
    raster_out_list = []
    for i in range(len(tilenames)):
        print('%%'*20)
        print(f'tile:{tilenames[i]}')
        prediction_workflow(yaml_pattern,tilename,keys_to_keep,fcolydi,fcolyid, fcolref,fcolY,fcolX,modelpaths)

    tb = time.perf_counter() - ta 
    print(f'[INFO]: run.time = {tb/60} min(s)')


