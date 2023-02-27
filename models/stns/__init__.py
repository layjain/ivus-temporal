from . import base
from . import rigid_stn

def get_stn(args):
    if args.transform=="rigid":
        return rigid_stn.RigidSTN()