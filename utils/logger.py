import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "project",
                 log_dir: str = "logs",
                 log_file: str = "project.log",
                 level: int = logging.INFO) -> logging.Logger:
    """
    Создаёт и настраивает логгер с выводом в файл и консоль.

    Параметры
    ----------
    name : str
        Имя логгера (удобно использовать __name__ модуля).
    log_dir : str
        Папка для лог-файлов.
    log_file : str
        Имя файла лога.
    level : int
        Уровень логирования (по умолчанию INFO).

    Возвращает
    ----------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Не дублируем обработчики, если логгер уже настроен
    if not logger.handlers:
        # Создаём папку для логов
        os.makedirs(log_dir, exist_ok=True)

        # Файловый обработчик с ротацией (10 МБ, 3 копии)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, log_file),
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(level)

        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Формат сообщений
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger