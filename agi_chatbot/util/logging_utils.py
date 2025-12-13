import logging
import logging.config
import os


def initialize_logging(config_path: str = None):
    """Initialize logging using a config file if available, otherwise basicConfig."""
    try:
        if config_path and os.path.exists(config_path):
            logging.config.fileConfig(config_path, disable_existing_loggers=False)
        else:
            # Fallback to a sensible default
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            )
    except Exception:
        # Best-effort only: avoid raising during startup
        logging.basicConfig(level=logging.INFO)


def get_logger(name: str = None):
    return logging.getLogger(name)
