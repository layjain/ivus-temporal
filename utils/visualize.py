import wandb

class Visualize(object):
    def __init__(self, args):
        self.args = args
        self._init = False

    def wandb_init(self, model):
        if not self._init:
            self._init = True
            wandb.init(project=self.args.project_name, group=args.group_number, config=self.args, name=self.args.name)
            wandb.watch(model)

    def log(self, key_vals):
        return wandb.log(key_vals)
