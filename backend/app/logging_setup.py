"""設定 log 的格式與層級,讓所有進入點印出來的 log 長得一樣。"""

import logging

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
