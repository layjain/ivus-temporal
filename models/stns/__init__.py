from . import base
from . import rigid_stn
from . import rotation_stn

def get_stn(args):
    if args.transform=="Rigid":
        return rigid_stn.RigidSTN()
    elif args.transform=="Rotation":
        return rotation_stn.RotationSTN()
    else:
        raise NotImplementedError(f"STN not implemented for {args.transform}")