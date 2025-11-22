"""Logging configuration"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level=logging.INFO,
    log_to_file=True,
    log_dir="logs",
    log_filename=None
):
    """
    Configure logging for the RAG system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
        log_dir: Directory for log files
        log_filename: Custom log filename (default: rag_YYYYMMDD_HHMMSS.log)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('rag_system')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (INFO and above, simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above, detailed format)
    if log_to_file:
        Path(log_dir).mkdir(exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'rag_{timestamp}.log'
        
        log_path = Path(log_dir) / log_filename
        
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger
