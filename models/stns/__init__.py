from . import base
from . import rigid_stn

def get_stn(args):
    if args.transform=="Rigid":
        return rigid_stn.RigidSTN()
    else:
        raise NotImplementedError(f"STN not implemented for {args.transform}")