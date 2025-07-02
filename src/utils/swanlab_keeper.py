import swanlab

class SwanlabKeeper:
    def __init__(self, config):

        self.config = config
        self.enable = config["logging"]["enable"]

        if self.enable == True:
            self.run = swanlab.init(
                project=config["app"],
                experiment_name=config["logging"]["run_name"],
                logdir=config["folder"],
                config=self.config,
            )
        else:
            print("Swanlab is disabled. No run initialized.")
    def log(self, global_step,step_log):
        self.run.log(step_log,step=global_step,)
        pass