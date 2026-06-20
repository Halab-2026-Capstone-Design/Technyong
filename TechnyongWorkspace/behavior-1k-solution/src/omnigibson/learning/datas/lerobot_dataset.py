try:
    from b1k.training.data_loader import BehaviorLeRobotDataset
except ImportError:
    class BehaviorLeRobotDataset:
        def __init__(self, *args, **kwargs):
            pass
