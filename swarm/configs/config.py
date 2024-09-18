from yacs.config import CfgNode as CN

_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.ROOT_DIR = "~/Desktop/projects/dataset/AffectNet/"
_C.PATHS.FEATS_DIR = "~/Desktop/projects/dataset/AffectNet/"
_C.PATHS.VIZ_DIR = "~/Desktop/projects/dataset/AffectNet/"
_C.PATHS.RES_DIR = "~/Desktop/projects/dataset/AffectNet/"

_C.DATASET = CN()
_C.DATASET.CITY = "Tampa"

_C.FEATS = CN()
_C.FEATS.FS = 18
_C.FEATS.WINDOW = 10 
_C.FEATS.OVERLAP = 0.75
_C.FEATS.COL = "0.75"
_C.FEATS.COLS = ["0.75"]
_C.FEATS.CLS_FEATS = ""
_C.FEATS.SMOTE = False
_C.FEATS.PCA = False
_C.FEATS.NUM_FEATS = 2
_C.FEATS.FEATS_LST = ['info', "psd"]

_C.FEATS_PARAM = CN()
_C.FEATS_PARAM.PSD_BW = 2
_C.FEATS_PARAM.PSD_NPERSEG = 0.5

_C.FLAGS = CN()
_C.FLAGS.VIZ = True
_C.FLAGS.LABELS = True

_C.TRAIN = CN()
_C.TRAIN.CLS = ""
_C.TRAIN.MODE = ""
_C.TRAIN.SMOTE_SAMPLING = [1., 1., 1.]

_C.GRID_SEARCH = CN()
_C.GRID_SEARCH.FEAT_LST = ['a', 'b']
_C.GRID_SEARCH.CLS_LST = ['a', 'b']
_C.GRID_SEARCH.COL_LST = ['a', 'b']

_C.CLS_PARAMS = CN()
_C.CLS_PARAMS.N_CLASS= 5
_C.CLS_PARAMS.COV_TYPE = ""
_C.CLS_PARAMS.MAX_ITER = 19


# COMMENTS
_C.COMMENTS = "TEST"




def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()