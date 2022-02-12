import asyncio
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import Text
from aiogram.types.bot_command import BotCommand
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from shutil import rmtree
import os

from matplotlib.pyplot import text

from nst1 import nst

from cycle_gan import CycleGanModel

API_TOKEN = '5090014965:AAF-nULwRTfQHtgjBxQMthuaq-JE5K_-HH0'

hellow_str = '''Привет, я бот, который умеет изменять стиль фотографии двумя способами. 
Первый способ долгий и осуществляется моделью nst и работает таким образом: сначала ты скидываешь фотку, на которую хочешь перенести стиль, 
а потом скидываешь фотку стиля. В итоге ты получишь фотку с тем стилем, который скинул. 
Второй способ осуществляется моделью style_gan, и умеет она делать картинку мультяшной. 
Для этого боту нужно скинуть только одну фотку, то есть ту, на которую будет перенесен мультяшный стиль.'''

loop = asyncio.get_event_loop()

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage(), loop=loop)

POOL = ThreadPoolExecutor(max_workers=cpu_count())


CONTENT_IMG_NAME_NST = "content_img_nst.png"
CONTENT_IMG_NAME_GAN = "content_img_gan.png"
STYLE_IMG_NAME = "style_img.png"
RESULT_IMG_NAME_NST = "result_img_nst.png"
RESULT_IMG_NAME_GAN = "result_img_gan.png"


class BotState(StatesGroup):
    initial_state = State()
    waiting_content_nst = State()
    waiting_style_nst = State()
    waiting_content_gan = State()


num_inline_buttons_per_user = {}


async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="/start", description="Начать все заново")
    ]
    await bot.set_my_commands(commands)


def get_keyboard_model_selection(num):
    buttons = [
        types.InlineKeyboardButton(
            text="nst", callback_data=f'model_nst_{num}'),
        types.InlineKeyboardButton(
            text="style_gan", callback_data=f'model_gan_{num}')
    ]
    keyboard = types.InlineKeyboardMarkup(resize_keyboard=True)
    keyboard.add(*buttons)
    keyboard.add(types.InlineKeyboardButton(
        text="Описание бота", callback_data=f'model_cancel_{num}'))
    return keyboard


def get_keyboard_bot_description(num):
    buttons = [
        types.InlineKeyboardButton(
            text="Посмотреть примеры", callback_data=f'init_examples_{num}'),
        types.InlineKeyboardButton(
            text="Выбрать модель", callback_data=f'init_models_{num}')
    ]
    keyboard = types.InlineKeyboardMarkup(resize_keyboard=True)
    keyboard.add(*buttons)
    return keyboard


def get_path_to_session_img_dir(session_id: int) -> str:
    return "images/" + str(session_id) + '/'


async def description_bot(message: types.Message):
    await message.answer(hellow_str, reply_markup=get_keyboard_bot_description(
        num_inline_buttons_per_user[message.chat.id]))


async def description_bot_per_cancel(message: types.Message):
    await message.edit_text(hellow_str, reply_markup=get_keyboard_bot_description(
        num_inline_buttons_per_user[message.chat.id]))


async def choose_model(message: types.Message):
    await message.edit_text(text='Выбери модель', reply_markup=get_keyboard_model_selection(
        num_inline_buttons_per_user[message.chat.id]))


async def choose_model_per_examples(message: types.Message):
    await message.answer(text='Выбери модель', reply_markup=get_keyboard_model_selection(
        num_inline_buttons_per_user[message.chat.id]))


async def make_ban(message: types.Message):
    await message.edit_text('Воспользуйтесь последней вызванной клавиатурой, либо нажмите /start для вызова новой.')


@dp.message_handler(commands="start", state='*')
async def start_bot(message: types.Message):
    await BotState.initial_state.set()
    num_inline_buttons_per_user.setdefault(message.chat.id, -1)
    num_inline_buttons_per_user[message.chat.id] += 1
    await description_bot(message)


@dp.callback_query_handler(Text(startswith="init_"), state='*')
async def callbacks_init(call: types.CallbackQuery):
    action = call.data.split('_')[1]
    num = int(call.data.split('_')[2])
    if num == num_inline_buttons_per_user[call.message.chat.id]:
        if action == 'examples':

            media_nst = types.MediaGroup()
            media_nst.attach_photo(types.InputFile(
                'examples/content_img_nst.png'), 'Контент')
            media_nst.attach_photo(types.InputFile(
                'examples/style_img.png'), 'Стиль')
            media_nst.attach_photo(types.InputFile(
                'examples/result_img_nst.png'), 'Резултат NST')
            await call.message.edit_text('Пример работы NST: контент, стиль и результат соответственно')
            await call.message.answer_media_group(media=media_nst)

            media_gan = types.MediaGroup()
            media_gan.attach_photo(types.InputFile(
                'examples/content_img_gan.png'), 'Контент')
            media_gan.attach_photo(types.InputFile(
                'examples/result_img_gan.png'), 'Резултат GAN')
            await call.message.answer('Пример работы GAN: контент и результат соответственно')
            await call.message.answer_media_group(media=media_gan)

            await choose_model_per_examples(call.message)
        elif action == 'models':
            await choose_model(call.message)
    else:
        await make_ban(call.message)
    await call.answer()


@dp.callback_query_handler(Text(startswith="model_"), state='*')
async def callbacks_init(call: types.CallbackQuery):
    action = call.data.split('_')[1]
    num = int(call.data.split('_')[2])
    if num == num_inline_buttons_per_user[call.message.chat.id]:
        if action == 'nst':
            await call.message.edit_text('Пришли картинку, на которую будет перенесен стиль')
            await BotState.waiting_content_nst.set()
        elif action == 'gan':
            await call.message.edit_text('Пришли картинку, на которую будет перенесен мультяшный стиль')
            await BotState.waiting_content_gan.set()
        elif action == 'cancel':
            await description_bot_per_cancel(call.message)
    else:
        await make_ban(call.message)
    await call.answer()


async def download_image(message: types.Message, file_path: str, need_style=0):
    if "photo" in message:
        photo = message.photo.pop()
        await photo.download(file_path)
        return 1

    elif "document" in message:
        document = message.document
        if document.mime_type.split("/")[0] == "image":
            await document.download(file_path)
            return 1
        else:
            return 0
    else:
        return 0


@dp.message_handler(
    content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT],
    state=[BotState.waiting_style_nst,
           BotState.waiting_content_gan, BotState.waiting_content_nst]
)
async def download_images(message: types.Message, state: FSMContext):

    session_dir = get_path_to_session_img_dir(message.chat.id)
    current_state = await state.get_state()
    if current_state.split(':')[1] == 'waiting_style_nst':
        success_download = await download_image(message, os.path.join(session_dir, STYLE_IMG_NAME))
        if success_download:
            await message.answer(
                'Ждем результаты nst. Это займет 5-7 минут'
            )
            nst_res = nst(session_dir + CONTENT_IMG_NAME_NST, session_dir +
                          STYLE_IMG_NAME, session_dir + RESULT_IMG_NAME_NST)
            await dp.loop.run_in_executor(
                POOL,
                nst_res.predict
            )
            result_img = types.InputFile(
                os.path.join(session_dir, RESULT_IMG_NAME_NST))
            await message.answer_photo(result_img)
            rmtree(session_dir)
            await choose_model_per_examples(message)
        else:
            await message.answer('Пришли изображение, присланный файл не поддается обработке')

    elif current_state.split(':')[1] == 'waiting_content_gan':
        success_download = await download_image(message, os.path.join(session_dir, CONTENT_IMG_NAME_GAN))
        if success_download:
            await message.answer(
                'Ждем результаты gan. Это займет меньше минуты'
            )
            model = CycleGanModel()
            _ = await dp.loop.run_in_executor(
                POOL,
                model.predict,
                os.path.join(session_dir, CONTENT_IMG_NAME_GAN),
                os.path.join(session_dir, RESULT_IMG_NAME_GAN)
            )
            result_img = types.InputFile(
                os.path.join(session_dir, RESULT_IMG_NAME_GAN))

            await message.answer_photo(result_img)
            rmtree(session_dir)
            await choose_model_per_examples(message)
        else:
            await message.answer('Пришли изображение, присланный файл не поддается обработке')

    elif current_state.split(':')[1] == 'waiting_content_nst':
        success_download = await download_image(message, os.path.join(session_dir, CONTENT_IMG_NAME_NST), need_style=1)

        if success_download:
            await BotState.waiting_style_nst.set()
            await message.answer('Теперь пришли картинку стиля')
        else:
            await message.answer('Пришли изображение, присланный файл не поддается обработке')


@dp.message_handler()
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, 'Я не понимаю тебя, для начала работы нажми /start')

if __name__ == '__main__':
    executor.start_polling(
        dp, loop=loop, skip_updates=True)
