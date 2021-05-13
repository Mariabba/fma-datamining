import hashlib
import logging
import time

from aiogram import Bot, Dispatcher, executor
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent

API_TOKEN = ""

logging.basicConfig(level=logging.DEBUG)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.inline_handler()
async def inline_echo(inline_query: InlineQuery):
    # id affects both preview and content,
    # so it has to be unique for each result
    # (Unique identifier for this result, 1-64 Bytes)
    # you can set your unique id's
    # but for example i'll generate it based on text because I know, that
    # only text will be passed in this example
    with open("data/tokens/progress.txt", "r") as f:
        text = f.read()
    text = int(text)
    LEN = 10640
    percent, compl, total = int(text / LEN * 100), text, LEN
    text = f"Status: {percent}%, {compl} out of {total} runs"
    input_content = InputTextMessageContent(text)
    result_id: str = hashlib.md5(text.encode()).hexdigest()
    item = InlineQueryResultArticle(
        id=result_id,
        title=text,
        input_message_content=input_content,
    )
    # don't forget to set cache_time=1 for testing (default is 300s or 5m)
    await bot.answer_inline_query(inline_query.id, results=[item], cache_time=1)


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
