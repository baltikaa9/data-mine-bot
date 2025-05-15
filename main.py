import asyncio
import logging

import matplotlib
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiohttp import ClientTimeout

from src.config import BOT_TOKEN
from src.handlers import router

font = {
    'family': 'normal',
    'size': 20
}

matplotlib.rc('font', **font)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    timeout=ClientTimeout(total=60)
)
dp = Dispatcher()

# Подключаем роутер с обработчиками
dp.include_router(router)

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
