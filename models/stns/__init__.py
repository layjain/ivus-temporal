from . import base
from . import rigid_stn
from . import rotation_stn
from . import parameters

def get_stn(args):
    if set(args.transforms)==set(["Translation", "Rotation"]):
        return rigid_stn.RigidSTN()
    elif args.transforms==["Rotation"]:
        return rotation_stn.RotationSTN()
    else:
        raise NotImplementedError(f"STN not implemented for transforms:\n {args.transforms}")