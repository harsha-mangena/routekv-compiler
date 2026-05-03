"""Rich-based structured logger for RouteKV."""
import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a rich-formatted logger."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger(name)
