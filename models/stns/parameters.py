class Parameters(object):

    def __init__(self, translation=None, rotation=None):
        if (translation is None) and (rotation is None):
            raise ValueError("Parameters Empty")

        self.translation=translation
        self.rotation=rotation
        if self.rotation.device!=self.translation.device:
            raise ValueError("Rotations and Translations can't be on different devices")

        self.device = self.rotation.device

    def __repr__(self) -> str:
        ret = f"Parameters(T={list(self.translation)}, R={list(self.rotation)}) at {self.device}"
        return ret