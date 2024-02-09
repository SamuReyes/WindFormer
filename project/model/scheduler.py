from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class LinearLR(_LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose="deprecated"):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                'Starting multiplicative factor expected to be greater than 0 and less or equal to 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                'Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]


class WarmUpScheduler(object):
    """
    Warm up scheduler for changing learning rate at the beginning of training
    Need to call WarmUpScheduler behind lr_scheduler instance in Pytorch.

    Args:
        optimizer: Optimizer = Wrapped optimizer in Pytorch.
        lr_scheduler: _LRScheduler = Wrapped lr_scheduler in Pytorch.
        warmup_steps: int = The number of iterations for warmup_scheduler_pytorch.
        warmup_start_lr: list or tuple or float = The start learning rate of warmup_scheduler_pytorch
                                                  for optimizer param_groups.
        len_loader: int = The length of dataloader.
        warmup_mode: str ='linear'.
        verbose: bool = If True, prints a message to stdout for each update.

    Example:
       '>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)                                               '
       '>>> lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)                    '
       '>>> data_loader = torch.utils.data.DataLoader(...)                                                        '
       '>>> warmup_scheduler_pytorch = WarmUpScheduler(optimizer, lr_scheduler, len_loader=len(data_loader),      '
       '>>>                                    warmup_steps=64, warmup_start_lr=0.01)                             '
       '>>> for epoch in range(10):                                                                               '
       '>>>     for batch in data_loader:                                                                         '
       '>>>         train(...)                                                                                    '
       '>>>         validate(...)                                                                                 '
       '>>>         warmup_scheduler_pytorch.step()                                                               '
    """

    def __init__(self, optimizer, lr_scheduler, warmup_steps: int, warmup_start_lr,
                 len_loader: int = 1, warmup_mode: str = 'linear', verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f'{type(optimizer).__name__} is not an Optimizer in pytorch')
        self.optimizer = optimizer

        # Attach lr_scheduler
        if not isinstance(lr_scheduler, (_LRScheduler, ReduceLROnPlateau)):
            raise TypeError(
                f'{type(lr_scheduler).__name__} is not a lr_scheduler in pytorch')
        self.lr_scheduler = lr_scheduler

        # check whether attribute initial_lr in optimizer.param_group
        for idx, group in enumerate(optimizer.param_groups):
            if 'initial_lr' not in group:
                raise KeyError("param 'initial_lr' is not specified "
                               f"in param_groups[{idx}] when resuming an optimizer")
        self.base_lrs = [group['initial_lr']
                         for group in optimizer.param_groups]

        self.len_loader = len_loader
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode

        if isinstance(warmup_start_lr, (list, tuple)):
            assert len(warmup_start_lr) == len(self.base_lrs), \
                f'The length of warmup_start_lr {len(warmup_start_lr)} ' \
                f'and optimizer.param_group {len(self.base_lrs)} do not correspond'
            self.warmup_start_lrs = warmup_start_lr

        else:
            self.warmup_start_lrs = [warmup_start_lr] * len(self.base_lrs)

        self.last_step = -1
        self.last_epoch = -1
        self._step_count = 0
        self._last_lr = None
        self.__warmup_done = False
        self.__is_ReduceLROnPlateau = isinstance(
            lr_scheduler, ReduceLROnPlateau)
        self.verbose = verbose

        self.step()

    def state_dict(self):
        r"""
        It contains an entry for every variable in self.__dict__
        which is not one of the ('optimizer', 'lr_scheduler').

        Returns:
            the state of the scheduler as a dict.
        """
        return {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}

    def load_state_dict(self, state_dict):
        r"""
        Loads the schedulers state.

        Args:
            state_dict: dict = scheduler state. Should be an object returned from a call to state_dict.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        r"""
        Return last computed learning rate by current warmup_scheduler_pytorch scheduler.
        """
        return self._last_lr

    def get_warmup_lr(self):
        r"""Return warmup_scheduler_pytorch learning rate to upgrade"""
        if self.warmup_mode == 'linear':
            return [warmup_lr + (base_lr - warmup_lr) * (self.last_step / self.warmup_steps)
                    for warmup_lr, base_lr in zip(self.warmup_start_lrs, self.base_lrs)]
        else:
            raise ValueError(f"Now the other warmup_mode is not implemented, "
                             f"there is only 'linear' mode")

    @staticmethod
    def print_lr(is_verbose, group, lr, epoch=None):
        """Display the current learning rate"""
        if is_verbose:
            if epoch is None:
                print(f'Adjusting learning rate '
                      f'of group {group} to {lr:.4e}')
            else:
                print(f'Epoch {epoch:5d}: adjusting learning rate '
                      f'of group {group} to {lr:.4e}')

    @property
    def warmup_done(self):
        r"""Return whether warnup is done"""
        return self.__warmup_done

    @property
    def _new_epoch(self):
        r"""Return whether is a new epoch started now"""
        return self.last_step % self.len_loader == 0

    def _step(self, epoch, metrics):
        r"""For warmup_scheduler_pytorch and lr_scheduler step once"""
        if self.__warmup_done and self._new_epoch:
            if self.__is_ReduceLROnPlateau:
                self.lr_scheduler.step(metrics, epoch)
            else:
                self.lr_scheduler.step(epoch)

        elif (not self.__warmup_done) and (self.last_step <= self.warmup_steps):
            values = self.get_warmup_lr()

            if self.last_step >= self.warmup_steps:
                self.__warmup_done = True

            for idx, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, values)):
                param_group['lr'] = lr
                self.print_lr(self.verbose, idx, lr, epoch)

    def step(self, metrics=None, step=None, epoch=None):
        self._step_count += 1

        if step is None and epoch is None:
            self.last_step += 1
            if self._new_epoch:
                self.last_epoch += 1
            self._step(epoch, metrics)

        elif step is not None and epoch is None:
            self.last_step = step
            self.last_epoch = step // self.len_loader
            self._step(epoch, metrics)

        elif step is None and epoch is not None:
            self.last_step = epoch * self.len_loader
            self.last_epoch = epoch
            self._step(epoch, metrics)

        else:  # if step and epoch
            # step is relative to epoch only here
            self.last_step = step + epoch * self.len_loader
            self.last_epoch = epoch
            self._step(epoch, metrics)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
