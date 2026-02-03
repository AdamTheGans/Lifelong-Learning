from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import os
import time

@dataclass
class TBLogger:
    run_name: str
    log_dir: str = "runs"
    writer: SummaryWriter | None = None

    def __post_init__(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        full_dir = os.path.join(self.log_dir, f"{self.run_name}_{ts}")
        self.writer = SummaryWriter(full_dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        assert self.writer is not None
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
