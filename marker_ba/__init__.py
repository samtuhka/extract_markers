import cppimport
cppimport.set_quiet(False)
# A hack required to get relative imports working
# with cppimport
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
try:
    cppimport.imp('marker_ba')
#    cppimport.imp('_theia')
finally:
    sys.path.pop(0)
from marker_ba import *
#import _theia as theia
