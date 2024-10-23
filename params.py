
def define_parameters():
    roi = 'all'  # TLS,#MDT #RNG ,#ALL #ALL-r=[call all]
    mx = 'gpu' # gpu,cpu
    fcolydi = 'ldtm'
    fcolyid = 'zdif'
    fcolref = 'tdem_dem_filled'
    nboost = 10#0
    od_wait = 10#0
    rnd_seed = 123#42[x] 
    fcolX = ['egm08', 'egm96', 'tdem_hem', fcolref]  # ,'edem_demw84'

    catboost_params = {
      'iterations': nboost,
      #'learning_rate': 0.1,
      'depth': 16,
      'loss_function': 'RMSE',
      'eval_metric': 'RMSE',
      'random_seed': rnd_seed,#
      'od_type': 'Iter',
      'od_wait': od_wait,
      'task_type': 'GPU',  # Change to 'GPU' if using GPU
      'devices': '0:1'  # Uncomment and adjust if using specific GPU devices
    }
    nboost = nboost + od_wait
    fcolY = [fcolydi, fcolyid]
    FTCOLSC = fcolX + [fcolydi]  # fcolYc ,
    return rnd_seed,roi,mx,fcolydi,fcolyid, fcolref, fcolX,catboost_params, nboost,fcolY,FTCOLSC

MKD_tiles = ['N09E106','N10E104','N10E105','N10E106'] # one tile to process 
TLS_tiles = ['N13E103']
RNG_tiles = []
roi_tiles = MKD_tiles + TLS_tiles
