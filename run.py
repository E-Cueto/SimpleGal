import numpy as np


### Astronomy
import astropy.cosmology as co

### Statistical tools

import scipy.interpolate as si
from scipy.integrate import simpson as simps

### Gotta Go FAST
from numba import jit, njit, vectorize, prange
import time as Ti
### System manipulation
import sys
import os 
sys.path.append('/home/vdm981_alumni_ku_dk/modi_mount/Astraeus/astraeus/dynamic_IMF/yield_table_FINAL/output_data_original')
sys.path.append('/home/vdm981_alumni_ku_dk/modi_mount/ToyModel/ToyModel/')
sys.path.append(r'/home/vdm981_alumni_ku_dk/modi_mount/ToyModel/ToyModel/src/')


from utils import *
from StarFormation import *
from fescClouds import *
from fescISM import *
from HaloProperties import *
from SNfeedback import *
from IMF import *
from Luminosity import *
from output import *
from simparam import *
from Galaxy import *

n = 64

# Which z=0 halo masses to run simulations for
#Halo_Masses = 10 ** np.arange(9,10.01,0.02)
Halo_Masses = 10 ** np.arange(10,14.51,0.02)

# Halo mass growth parameter. halo mass is proportional to (1+z)**0.25 * exp(beta * z). default is -0.75
beta = np.array([-0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55
                ]) 
#beta = np.array([-0.95, -0.90, -0.85, -0.80
#                ])

# How many runs for each scenario to run
Nexp = 10

# Which cloud mass limits to run with
minMs = [[1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],#  0,  1
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],#  2,  3
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],#  4,  5
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],#  6,  7
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],#  8,  9
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 10, 11
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 12, 13
         [1e4,1e4,1e4],                       # 14  
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 15, 16
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 17, 18
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 19, 20
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 21, 22
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 23, 24
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 25, 26
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 27, 28
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 29, 30 
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 31, 32
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 33, 34
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 35, 36
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 37, 38
         [1e4,1e4],[1e4,1e4], [1e4,1e4],[1e4,1e4],# 39, 40
         [1e4,1e4],[1e4,1e4], [1e4,1e4],[1e4,1e4],# 41, 42
         [1e4,1e4],[1e4,1e4], [1e4,1e4],[1e4,1e4],# 43, 44
         [1e4,1e4],[1e4,1e4], [1e4,1e4],[1e4,1e4],# 45, 46
         [1e4,1e4],[1e4,1e4], [1e4,1e4],[1e4,1e4],# 47, 48
         [1e4,1e4],[1e4,1e4], [1e4,1e4],[1e4,1e4],# 49, 50
         [1e4,1e4,1e4,1e4], [1e4,1e4,1e4,1e4],# 63, 50
        ]
maxMs = [[1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],#  0,  1
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],#  2,  3
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],#  4,  5
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],#  6,  7
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],#  8,  9
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 10, 11
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 12, 13
         [1e8,1e8,1e8],                       # 14  
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 15, 16
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 17, 18
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 19, 20
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 21, 22
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 23, 24
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 25, 26
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 27, 28
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 29, 30 
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 31, 32
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 33, 34
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 35, 36
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 37, 38
         [1e8,1e8],[1e8,1e8], [1e8,1e8],[1e8,1e8],# 39, 40
         [1e8,1e8],[1e8,1e8], [1e8,1e8],[1e8,1e8],# 41, 42
         [1e8,1e8],[1e8,1e8], [1e8,1e8],[1e8,1e8],# 43, 44
         [1e8,1e8],[1e8,1e8], [1e8,1e8],[1e8,1e8],# 45, 46
         [1e8,1e8],[1e8,1e8], [1e8,1e8],[1e8,1e8],# 47, 48
         [1e8,1e8],[1e8,1e8], [1e8,1e8],[1e8,1e8],# 49, 50
         [1e8,1e8,1e8,1e8], [1e8,1e8,1e8,1e8],# 63, 50
        ]

# Which IMF formulations to run simulations for. Options are 'salpeter', 'evolving', 'evolving2', 'salpeter_cloud', 'evolving_cloud'
IMFs       = [['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],#  0,  1
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],#  2,  3
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],#  4,  5
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],#  6,  7
              ['salpeter_cloud', 'salpeter_cloud', 'salpeter_cloud', 'salpeter_cloud'], ['salpeter_cloud', 'salpeter_cloud', 'salpeter_cloud', 'salpeter_cloud'],#  8,  9
              ['salpeter'      , 'salpeter'      , 'salpeter'      , 'salpeter'      ], ['salpeter'      , 'salpeter'      , 'salpeter'      , 'salpeter'      ],# 10, 11
              ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ], ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ],# 12, 13
              ['evolving'      , 'salpeter'      , 'salpeter_cloud'],                                                                                            # 14
              ['evolving2'     , 'evolving2'     , 'evolving2'     , 'evolving2'     ], ['evolving2'     , 'evolving2'     , 'evolving2'     , 'evolving2'     ],# 15, 16
              ['evolving2'     , 'evolving2'     , 'evolving2'     , 'evolving2'     ], ['evolving2'     , 'evolving2'     , 'evolving2'     , 'evolving2'     ],# 17, 18
              ['evolving2'     , 'evolving2'     , 'evolving2'     , 'evolving2'     ], ['evolving2'     , 'evolving2'     , 'evolving2'     , 'evolving2'     ],# 19, 20
              ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ], ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ],# 21, 22
              ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ], ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ],# 23, 24
              ['salpeter'      , 'salpeter'      , 'salpeter'      , 'salpeter'      ], ['salpeter'      , 'salpeter'      , 'salpeter'      , 'salpeter'      ],# 25, 26
              ['salpeter'      , 'salpeter'      , 'salpeter'      , 'salpeter'      ], ['salpeter'      , 'salpeter'      , 'salpeter'      , 'salpeter'      ],# 27, 28
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],# 29, 30
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],# 31, 32
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],# 33, 34
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],# 35, 36
              ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud', 'evolving_cloud', 'evolving_cloud'],# 37, 38
              ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'],[ 'evolving_cloud', 'evolving_cloud'],# 39, 40
              ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'],[ 'evolving_cloud', 'evolving_cloud'],# 42, 42
              ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'],[ 'evolving_cloud', 'evolving_cloud'],# 43, 44
              ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'],[ 'evolving_cloud', 'evolving_cloud'],# 45, 46
              ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'],[ 'evolving_cloud', 'evolving_cloud'],# 47, 48
              ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'], ['evolving_cloud', 'evolving_cloud'],[ 'evolving_cloud', 'evolving_cloud'],# 49, 50
              ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ], ['evolving'      , 'evolving'      , 'evolving'      , 'evolving'      ],# 63,64
             ]               

# Which SNfeedback model to use. options are 'delayed' or '10myr'
SNfeedback = [['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],#  0,  1
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],#  2,  3
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],#  4,  5
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],#  6,  7
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],#  8,  9
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 10, 11
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 12, 13
              ['delayed', 'delayed', 'delayed'],                                                         # 14
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 15, 16
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 17, 18
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 19, 20
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 21, 22
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 23, 24
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 25, 26
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 27, 28
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 29, 30
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 31, 32
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 33, 34
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 35, 36
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 37, 38
              ['delayed', 'delayed'],[ 'delayed', 'delayed'], ['delayed', 'delayed'],[ 'delayed', 'delayed'],# 39, 40
              ['delayed', 'delayed'],[ 'delayed', 'delayed'], ['delayed', 'delayed'],[ 'delayed', 'delayed'],# 42, 42
              ['delayed', 'delayed'],[ 'delayed', 'delayed'], ['delayed', 'delayed'],[ 'delayed', 'delayed'],# 43, 44
              ['delayed', 'delayed'],[ 'delayed', 'delayed'], ['delayed', 'delayed'],[ 'delayed', 'delayed'],# 45, 46
              ['delayed', 'delayed'],[ 'delayed', 'delayed'], ['delayed', 'delayed'],[ 'delayed', 'delayed'],# 47, 48
              ['delayed', 'delayed'],[ 'delayed', 'delayed'], ['delayed', 'delayed'],[ 'delayed', 'delayed'],# 49, 50
              ['delayed', 'delayed', 'delayed', 'delayed'], ['delayed', 'delayed', 'delayed', 'delayed'],# 63, 64
             ]
 
# Galaxy-wide star formation efficiency and SN wind coupling efficiency
fstar = [[0.010, 0.010, 0.010, 0.010], [0.010, 0.010, 0.010, 0.010],#  0,  1
         [0.010, 0.010, 0.010, 0.010], [0.010, 0.010, 0.010, 0.010],#  2,  3
         [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025],#  4,  5
         [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025],#  6,  7
         [0.010, 0.010, 0.025, 0.025], [0.010, 0.010, 0.025, 0.025],#  8,  9
         [0.010, 0.010, 0.025, 0.025], [0.010, 0.010, 0.025, 0.025],# 10, 11
         [0.010, 0.010, 0.025, 0.025], [0.010, 0.010, 0.025, 0.025],# 12, 13
         [0.010, 0.010, 0.010],                                     # 14
         [0.010, 0.010, 0.025, 0.025], [0.010, 0.010, 0.025, 0.025],# 15, 16
         [0.010, 0.010, 0.025, 0.025], [0.010, 0.010, 0.025, 0.025],# 17, 18
         [0.025, 0.025, 0.025, 0.025], [0.010, 0.010, 0.010, 0.010],# 19, 20
         [0.010, 0.010, 0.025, 0.025], [0.010, 0.010, 0.025, 0.025],# 21, 22
         [0.025, 0.025, 0.025, 0.025], [0.010, 0.010, 0.010, 0.010],# 23, 24
         [0.010, 0.010, 0.025, 0.025], [0.010, 0.010, 0.025, 0.025],# 25, 26
         [0.025, 0.025, 0.025, 0.025], [0.010, 0.010, 0.010, 0.010],# 27, 28
         [0.020, 0.020, 0.020, 0.020], [0.020, 0.020, 0.020, 0.020],# 29, 30
         [0.020, 0.020, 0.020, 0.020], [0.020, 0.020, 0.020, 0.020],# 31, 32
         [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025],# 33, 34
         [0.030, 0.030, 0.030, 0.030], [0.030, 0.030, 0.030, 0.030],# 35, 36
         [0.030, 0.030, 0.030, 0.030], [0.030, 0.030, 0.030, 0.030],# 37, 38
         [0.010, 0.010],[ 0.010, 0.010], [0.010, 0.010],[ 0.010, 0.010],# 39, 40
         [0.015, 0.015],[ 0.015, 0.015], [0.015, 0.015],[ 0.015, 0.015],# 41, 42
         [0.015, 0.015],[ 0.015, 0.015], [0.015, 0.015],[ 0.015, 0.015],# 43, 44
         [0.015, 0.015],[ 0.015, 0.015], [0.015, 0.015],[ 0.015, 0.015],# 45, 46
         [0.020, 0.020],[ 0.020, 0.020], [0.020, 0.020],[ 0.020, 0.020],# 47, 48
         [0.030, 0.030],[ 0.030, 0.030], [0.030, 0.030],[ 0.030, 0.030],# 49, 50
         [0.025, 0.025, 0.025, 0.025], [0.010, 0.010, 0.010, 0.010],# 63, 64
        ]
fwind = [[0.1,  0.1,  0.1,  0.1], [0.2,  0.2,  0.2,  0.2],#  0,  1
         [0.3,  0.3,  0.3,  0.3], [0.4,  0.4,  0.4,  0.4],#  2,  3
         [0.1,  0.1,  0.1,  0.1], [0.2,  0.2,  0.2,  0.2],#  4,  5
         [0.3,  0.3,  0.3,  0.3], [0.4,  0.4,  0.4,  0.4],#  6,  7
         [0.2,  0.2,  0.2,  0.2], [0.4,  0.4,  0.4,  0.4],#  8,  9
         [0.2,  0.2,  0.2,  0.2], [0.4,  0.4,  0.4,  0.4],# 10, 11
         [0.2,  0.2,  0.2,  0.2], [0.4,  0.4,  0.4,  0.4],# 12, 13
         [0.4,  0.4,  0.4],                               # 14
         [0.2,  0.2,  0.2,  0.2], [0.4,  0.4,  0.4,  0.4],# 15, 16
         [0.1,  0.1,  0.1,  0.1], [0.3,  0.3,  0.3,  0.3],# 17, 18
         [0.1,  0.2,  0.3,  0.4], [0.1,  0.2,  0.3,  0.4],# 19, 20
         [0.1,  0.1,  0.1,  0.1], [0.3,  0.3,  0.3,  0.3],# 21, 22
         [0.1,  0.2,  0.3,  0.4], [0.1,  0.2,  0.3,  0.4],# 23, 24
         [0.1,  0.1,  0.1,  0.1], [0.3,  0.3,  0.3,  0.3],# 25, 26
         [0.1,  0.2,  0.3,  0.4], [0.1,  0.2,  0.3,  0.4],# 27, 28
         [0.1,  0.3,  0.3,  0.3], [0.1,  0.4,  0.4,  0.4],# 29, 30
         [0.2,  0.2,  0.2,  0.1], [0.2,  0.3,  0.4,  0.1],# 31, 32
         [0.2,  0.3,  0.4,  0.1], [0.2,  0.3,  0.4,  0.1],# 33, 34
         [0.1,  0.2,  0.3,  0.4], [0.1,  0.2,  0.3,  0.4],# 35, 36
         [0.1,  0.2,  0.3,  0.4], [0.1,  0.2,  0.3,  0.4],# 37, 38
         [0.2,  0.3],[  0.4,  0.1], [0.2,  0.3],[  0.4,  0.1],# 39, 40
         [0.1,  0.2],[  0.3,  0.4], [0.1,  0.2],[  0.3,  0.4],# 41, 42
         [0.1,  0.2],[  0.3,  0.4], [0.1,  0.2],[  0.3,  0.4],# 43, 44
         [0.1,  0.2],[  0.3,  0.4], [0.1,  0.2],[  0.3,  0.4],# 45, 46
         [0.1,  0.2],[  0.3,  0.4], [0.1,  0.2],[  0.3,  0.4],# 47, 48
         [0.1,  0.2],[  0.3,  0.4], [0.1,  0.2],[  0.3,  0.4],# 49, 50
         [0.1,  0.2,  0.3,  0.4], [0.1,  0.2,  0.3,  0.4],# 63, 64
        ]

# Maximum cloud mass scaling factor. if 0 then no cloud mass limit scaling is done. Typical values between 2e7-1.2e8
cloudMassLimit = [[ 2e7,  4e7,  8e7, 12e7], [ 2e7,  4e7,  8e7, 12e7],#  0,  1
                  [ 2e7,  4e7,  8e7, 12e7], [ 2e7,  4e7,  8e7, 12e7],#  2,  3
                  [ 2e7,  4e7,  8e7, 12e7], [ 2e7,  4e7,  8e7, 12e7],#  4,  5
                  [ 2e7,  4e7,  8e7, 12e7], [ 2e7,  4e7,  8e7, 12e7],#  6,  7
                  [ 4e7, 12e7,  4e7, 12e7], [ 4e7, 12e7,  4e7, 12e7],#  8,  9
                  [ 4e7, 12e7,  4e7, 12e7], [ 4e7, 12e7,  4e7, 12e7],# 10, 11
                  [ 4e7, 12e7,  4e7, 12e7], [ 4e7, 12e7,  4e7, 12e7],# 12, 13
                  [  2e7,   2e7,   2e7],                                       # 14
                  [ 4e7, 12e7,  4e7, 12e7], [ 4e7, 12e7,  4e7, 12e7],# 15, 16
                  [ 4e7, 12e7,  4e7, 12e7], [ 4e7, 12e7,  4e7, 12e7],# 17, 18
                  [ 2e7,  2e7,  2e7,  2e7], [ 2e7,  2e7,  2e7,  2e7],# 19, 20
                  [ 4e7, 12e7,  4e7, 12e7], [ 4e7, 12e7,  4e7, 12e7],# 21, 22
                  [ 2e7,  2e7,  2e7,  2e7], [ 2e7,  2e7,  2e7,  2e7],# 23, 24
                  [ 4e7, 12e7,  4e7, 12e7], [ 4e7, 12e7,  4e7, 12e7],# 25, 26
                  [ 2e7,  2e7,  2e7,  2e7], [ 2e7,  2e7,  2e7,  2e7],# 27, 28
                  [ 3e7,  2e7,  3e7,  4e7], [ 5e7,  2e7,  3e7,  4e7],# 29, 30             
                  [ 2e7,  3e7,  4e7,  2e7], [ 5e7,  5e7,  5e7,  4e7],# 31, 32
                  [ 3e7,  3e7,  3e7,  3e7], [ 5e7,  5e7,  5e7,  5e7],# 33, 34
                  [ 2e7,  2e7,  2e7,  2e7], [ 4e7,  4e7,  4e7,  4e7],# 35, 36
                  [ 3e7,  3e7,  3e7,  3e7], [ 5e7,  5e7,  5e7,  5e7],# 37, 38
                  [ 3e7,  3e7],[  3e7,  3e7], [ 5e7,  5e7],[  5e7,  5e7],# 39, 40
                  [ 2e7,  2e7],[  2e7,  2e7], [ 4e7,  4e7],[  4e7,  4e7],# 41, 42
                  [ 3e7,  3e7],[  3e7,  3e7], [ 5e7,  5e7],[  5e7,  5e7],# 43, 44
                  [ 8e7,  8e7],[  8e7,  8e7], [12e7, 12e7],[ 12e7, 12e7],# 45, 46
                  [ 8e7,  8e7],[  8e7,  8e7], [12e7, 12e7],[ 12e7, 12e7],# 47, 48
                  [ 8e7,  8e7],[  8e7,  8e7], [12e7, 12e7],[ 12e7, 12e7],# 49, 50
                  [ 8e7,  8e7,  8e7,  8e7], [ 8e7,  8e7,  8e7,  8e7],# 63, 64
                 ]
# Halo Mass where simulations start
MhaloIni = [[7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],#  0,  1
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],#  2,  3
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],#  4,  5
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],#  6,  7
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],#  8,  9
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 10, 11
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 12, 13
            [7.4,  7.4,  7.4],                               # 14
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 15, 16
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 17, 18
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 19, 20
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 21, 22
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 23, 24
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 25, 26
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 27, 28
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 29, 30
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 31, 32
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 33, 34
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 35, 36
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 37, 38
            [7.4,  7.4],[  7.4,  7.4], [7.4,  7.4],[  7.4,  7.4],# 39, 40
            [7.4,  7.4],[  7.4,  7.4], [7.4,  7.4],[  7.4,  7.4],# 41, 42
            [7.4,  7.4],[  7.4,  7.4], [7.4,  7.4],[  7.4,  7.4],# 43, 44
            [7.4,  7.4],[  7.4,  7.4], [7.4,  7.4],[  7.4,  7.4],# 45, 46
            [7.4,  7.4],[  7.4,  7.4], [7.4,  7.4],[  7.4,  7.4],# 47, 48
            [7.4,  7.4],[  7.4,  7.4], [7.4,  7.4],[  7.4,  7.4],# 49, 50
            [7.4,  7.4,  7.4,  7.4], [7.4,  7.4,  7.4,  7.4],# 63, 64
           ]

# Cloud star formation efficiency. either a number for constant SFE or None for dynamic SFE
SFEcloud = [[None, None, None, None], [None, None, None, None],#  0,  1
            [None, None, None, None], [None, None, None, None],#  2,  3
            [None, None, None, None], [None, None, None, None],#  4,  5
            [None, None, None, None], [None, None, None, None],#  6,  7
            [None, None, None, None], [None, None, None, None],#  8,  9
            [None, None, None, None], [None, None, None, None],# 10, 11
            [None, None, None, None], [None, None, None, None],# 12, 13
            [None, None, None],                                # 14
            [None, None, None, None], [None, None, None, None],# 15, 16
            [None, None, None, None], [None, None, None, None],# 17, 18
            [None, None, None, None], [None, None, None, None],# 19, 20
            [None, None, None, None], [None, None, None, None],# 21, 22
            [None, None, None, None], [None, None, None, None],# 23, 24
            [None, None, None, None], [None, None, None, None],# 25, 26
            [None, None, None, None], [None, None, None, None],# 27, 28
            [None, None, None, None], [None, None, None, None],# 29, 30
            [None, None, None, None], [None, None, None, None],# 31, 32
            [None, None, None, None], [None, None, None, None],# 33, 34
            [None, None, None, None], [None, None, None, None],# 35, 36
            [None, None, None, None], [None, None, None, None],# 37, 38
            [None, None],[ None, None], [None, None],[ None, None],# 39, 40
            [None, None],[ None, None], [None, None],[ None, None],# 41, 42
            [None, None],[ None, None], [None, None],[ None, None],# 43, 44
            [None, None],[ None, None], [None, None],[ None, None],# 45, 46
            [None, None],[ None, None], [None, None],[ None, None],# 47, 48
            [None, None],[ None, None], [None, None],[ None, None],# 49, 50
            [None, None, None, None], [None, None, None, None],# 63, 64
           ]

# Cloud Density. Can be 1e2,1e3,1e4,1e5, or False for non-constant
cloudDensity = [[False, False, False, False], [False, False, False, False],#  0,  1
                [False, False, False, False], [False, False, False, False],#  2,  3
                [False, False, False, False], [False, False, False, False],#  4,  5
                [False, False, False, False], [False, False, False, False],#  6,  7
                [False, False, False, False], [False, False, False, False],#  8,  9
                [False, False, False, False], [False, False, False, False],# 10, 11
                [False, False, False, False], [False, False, False, False],# 12, 13
                [False, False, False],                                     # 14
                [False, False, False, False], [False, False, False, False],# 15, 16
                [False, False, False, False], [False, False, False, False],# 17, 18
                [False, False, False, False], [False, False, False, False],# 19, 20
                [False, False, False, False], [False, False, False, False],# 21, 22
                [False, False, False, False], [False, False, False, False],# 23, 24
                [False, False, False, False], [False, False, False, False],# 25, 26
                [False, False, False, False], [False, False, False, False],# 27, 28
                [False, False, False, False], [False, False, False, False],# 29, 30
                [False, False, False, False], [False, False, False, False],# 31, 32
                [False, False, False, False], [False, False, False, False],# 33, 34
                [False, False, False, False], [False, False, False, False],# 35, 36
                [False, False, False, False], [False, False, False, False],# 37, 38 
                [False, False], [False, False], [False, False],[ False, False],# 39, 40
                [False, False], [False, False], [False, False],[ False, False],# 41, 42
                [False, False], [False, False], [False, False],[ False, False],# 43, 44
                [False, False], [False, False], [False, False],[ False, False],# 45, 46
                [False, False], [False, False], [False, False],[ False, False],# 47, 48
                [False, False], [False, False], [False, False],[ False, False],# 49, 50
                [False, False, False, False], [False, False, False, False],# 63, 64
               ]
 
# SFE Turnover. True or False
turnover = [[True, True, True, True], [True, True, True, True],#  0,  1
            [True, True, True, True], [True, True, True, True],#  2,  3
            [True, True, True, True], [True, True, True, True],#  4,  5
            [True, True, True, True], [True, True, True, True],#  6,  7
            [True, True, True, True], [True, True, True, True],#  8,  9
            [True, True, True, True], [True, True, True, True],# 10, 11
            [True, True, True, True], [True, True, True, True],# 12, 13
            [True, True, True],                                # 14
            [True, True, True, True], [True, True, True, True],# 15, 16
            [True, True, True, True], [True, True, True, True],# 17, 18
            [True, True, True, True], [True, True, True, True],# 19, 20
            [True, True, True, True], [True, True, True, True],# 21, 22
            [True, True, True, True], [True, True, True, True],# 23, 24
            [True, True, True, True], [True, True, True, True],# 25, 26
            [True, True, True, True], [True, True, True, True],# 27, 28
            [True, True, True, True], [True, True, True, True],# 29, 30
            [True, True, True, True], [True, True, True, True],# 31, 32
            [True, True, True, True], [True, True, True, True],# 33, 34
            [True, True, True, True], [True, True, True, True],# 35, 36
            [True, True, True, True], [True, True, True, True],# 37, 38
            [True, True],[ True, True], [True, True],[ True, True],# 39, 40
            [True, True],[ True, True], [True, True],[ True, True],# 41, 42
            [True, True],[ True, True], [True, True],[ True, True],# 43, 44
            [True, True],[ True, True], [True, True],[ True, True],# 45, 46
            [True, True],[ True, True], [True, True],[ True, True],# 47, 48
            [True, True],[ True, True], [True, True],[ True, True],# 49, 50
            [True, True, True, True], [True, True, True, True],# 63, 64
           ]
            
# Extra bit for file names:
extra = ['_smooth2','_smooth2','_smooth2','_smooth2']
# Output directory
outDir = 'Galaxy_data/'

# Append existing files or create new ones
append = False
print('Running stochastic sims',flush=True)
run(       IMFs[n], Halo_Masses, minMs[n], maxMs[n], fstar[n], fwind[n], SNfeedback[n], MhaloIni[n], SFEcloud.copy()[n], cloudMassLimit[n], cloudDensity[n], turnover[n], beta, Nexp, outDir, append, extra)
#print('Running smooth sims',flush=True)
#run_smooth(IMFs[n], Halo_Masses, minMs[n], maxMs[n], fstar[n], fwind[n], SNfeedback[n], MhaloIni[n], SFEcloud.copy()[n], cloudMassLimit[n], beta, outDir, append)