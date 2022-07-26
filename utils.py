import matplotlib.pyplot as plt

def plot_lr(lr, epoch):
    fig, ax = plt.subplots()
    ax.plot(range(epoch), lr)
    ax.set_xlabel('epochs')
    ax.set_ylabel('learning rate')
    return fig

class LambdaLR:
    def __init__(self):
        self.main = 'Sets the learning rate of each parameter group to the initial lr times a given function.'
        self.lr_lambda = 'lr_lambda (function or list) – A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.'

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]

class MultiplicativeLR:
    def __init__(self):
        self.main = 'Multiply the learning rate of each parameter group by the factor given in the specified function.'
        self.lr_lambda = 'lr_lambda (function or list) – A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups.'

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]

class StepLR:
    def __init__(self):
        self.main = 'Decays the learning rate of each parameter group by gamma every step_size epochs.'
        self.step_size = 'Period of learning rate decay.'
        self.gamma = ' Multiplicative factor of learning rate decay. Default: 0.1.'

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]

class MultiStepLR:
    def __init__(self):
        self.main = 'Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.'
        self.milestones = 'List of epoch indices. Must be increasing.'
        self.gamma = ' Multiplicative factor of learning rate decay. Default: 0.1.'

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]


class ExponentialLR:
    def __init__(self):
        self.main = 'Decays the learning rate of each parameter group by gamma every epoch.'
        self.gamma = 'Multiplicative factor of learning rate decay.'

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]

class CosineAnnealingLR:
    def __init__(self):
        self.main = 'Set the learning rate of each parameter group using a cosine annealing schedule, where eta_max is set to the initial lr and T_cur is the number of epochs since the last restart in SGDR:' 
        self.T_max = 'Maximum number of iterations.'
        self.eta_min= 'Minimum learning rate. Default: 0.'

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]

class CyclicLR:
    def __init__(self):
        self.main = 'Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).'
        self.base_lr = 'Initial learning rate which is the lower boundary in the cycle for each parameter group.'
        self.max_lr= 'Upper learning rate boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_lr - base_lr). The lr at any cycle is the sum of base_lr and some scaling of the amplitude; therefore max_lr may not actually be reached depending on scaling function.'
        self.step_size_up= 'Number of training iterations in the increasing half of a cycle. Default: 2000'
        self.step_size_down= 'Number of training iterations in the decreasing half of a cycle. If step_size_down is None, it is set to step_size_up. Default: None'
        self.mode= 'One of {triangular, triangular2, exp_range}. Values correspond to policies detailed above. If scale_fn is not None, this argument is ignored. Default: "triangular"'
        self.gamma= 'Constant in ‘exp_range’ scaling function: gamma**(cycle iterations) Default: 1.0'
        self.scale_fn='Custom scaling policy defined by a single argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0. If specified, then ‘mode’ is ignored. Default: None'
        self.scale_mode="{‘cycle’, ‘iterations’}. Defines whether scale_fn is evaluated on cycle number or cycle iterations (training iterations since start of cycle). Default: ‘cycle’"
        self.cycle_momentum="If True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’. Default: True"
        self.base_momentum="Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled inversely to learning rate; at the peak of a cycle, momentum is ‘base_momentum’ and learning rate is ‘max_lr’. Default: 0.8"
        self.max_momentum="Upper momentum boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_momentum - base_momentum). The momentum at any cycle is the difference of max_momentum and some scaling of the amplitude; therefore base_momentum may not actually be reached depending on scaling function. Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is ‘max_momentum’ and learning rate is ‘base_lr’ Default: 0.9"

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]

class OneCycleLR:
    def __init__(self):
        self.main = 'Sets the learning rate of each parameter group according to the 1cycle learning rate policy.'
        self.max_lr= 'Upper learning rate boundaries in the cycle for each parameter group.'
        self.total_steps= 'The total number of steps in the cycle. Note that if a value is not provided here, then it must be inferred by providing a value for epochs and steps_per_epoch. Default: None'
        self.epochs= 'The number of epochs to train for. This is used along with steps_per_epoch in order to infer the total number of steps in the cycle if a value for total_steps is not provided. Default: None'
        self.steps_per_epoch = 'The number of steps per epoch to train for. This is used along with epochs in order to infer the total number of steps in the cycle if a value for total_steps is not provided. Default: None'
        self.pct_start= 'The percentage of the cycle (in number of steps) spent increasing the learning rate. Default: 0.3'
        self.anneal_strategy ="{‘cos’, ‘linear’} Specifies the annealing strategy: “cos” for cosine annealing, “linear” for linear annealing. Default: ‘cos’"
        self.cycle_momentum="If True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’. Default: True"
        self.base_momentum="Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled inversely to learning rate; at the peak of a cycle, momentum is ‘base_momentum’ and learning rate is ‘max_lr’. Default: 0.85"
        self.max_momentum="Upper momentum boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_momentum - base_momentum). Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is ‘max_momentum’ and learning rate is ‘base_lr’ Default: 0.95"
        self.div_factor="Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25"
        self.final_div_factor="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor"
        self.three_phase="If True, use a third phase of the schedule to annihilate the learning rate according to ‘final_div_factor’ instead of modifying the second phase (the first two phases will be symmetrical about the step indicated by ‘pct_start’)."

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]

class CosineAnnealingWarmRestarts:
    def __init__(self):
        self.main = "Set the learning rate of each parameter group using a cosine annealing schedule, where eta_max is set to the initial lr, T_cur is the number of epochs since the last restart and T_i is the number of epochs between two warm restarts in SGDR:"
        self.T_0 = "Number of iterations for the first restart."
        self.T_mult = "A factor increases T_i after a restart. Default: 1."
        self.eta_min = "Minimum learning rate. Default: 0."

    def get_description(self, key):
        attrs = list(self.__dict__.keys())
        if key in attrs:
            return self.__dict__[key]


