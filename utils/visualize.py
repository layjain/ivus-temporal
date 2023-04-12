import wandb

class Visualize(object):
    def __init__(self, args):
        self.args = args
        self._init = False

    def wandb_init(self, model, log_freq=1000):
        if not self._init:
            self._init = True
            wandb.init(project=self.args.project_name, group=self.args.group_number, config=self.args, name=self.args.wandb_name)
            wandb.watch(model, log='all', log_freq=log_freq)

    def log(self, key_vals):
        return wandb.log(key_vals)
