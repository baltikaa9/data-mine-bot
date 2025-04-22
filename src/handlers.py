from aiogram import F, types
from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message

from .keyboards import main_kb, get_clustering_methods_kb
from .utils import process_csv, plot_clusters_count

router = Router()

class ClusteringState(StatesGroup):
    waiting_for_method = State()
    waiting_for_clusters = State()
    waiting_for_file = State()
    waiting_for_auto_clusters = State()

# Обработчик /start
@router.message(Command('start'))
async def cmd_start(message: Message):
    await message.answer(
        'Выбери действие:',
        reply_markup=main_kb
    )

# Обработчик кнопки 'Кластеризация' (Reply)
@router.message(F.text == 'Кластеризация')
async def clustering_menu(message: Message):
    await message.answer(
        'Выбери метод кластеризации:',
        reply_markup=get_clustering_methods_kb()
    )

# Обработчик выбора метода
@router.callback_query(F.data.in_(['kmeans', 'gmm', 'hierarchical']))
async def select_method(callback: types.CallbackQuery, state: FSMContext):
    method = callback.data
    await state.update_data(method=method)
    await callback.message.edit_text(
        f'Выбран метод: {method}. Введи число кластеров (от 2 до 20):'
    )
    await state.set_state(ClusteringState.waiting_for_clusters)
    await callback.answer()

# Обработчик методов без указания кластеров (DBSCAN, MeanShift)
@router.callback_query(F.data.in_(['dbscan', 'meanshift']))
async def select_method_no_clusters(callback: types.CallbackQuery, state: FSMContext):
    method = callback.data
    await state.update_data(method=method, clusters=None)
    await callback.message.edit_text(
        f'Выбран метод: {method}. Теперь загрузи CSV-файл:'
    )
    await state.set_state(ClusteringState.waiting_for_file)
    await callback.answer()


# Обработчик ввода числа кластеров
@router.message(ClusteringState.waiting_for_clusters, F.text)
async def handle_clusters_input(message: Message, state: FSMContext):
    if not message.text.isdigit():
        await message.reply('❌ Введи ЧИСЛО, кретин!')
        return

    clusters = int(message.text)
    if clusters < 2 or clusters > 20:
        await message.reply('❌ Число кластеров должно быть от 2 до 20.')
        return

    await state.update_data(clusters=clusters)
    await message.answer('Теперь загрузи CSV-файл:')
    await state.set_state(ClusteringState.waiting_for_file)


@router.callback_query(F.data == 'auto_clusters')
async def handle_auto_clusters(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Загрузи CSV-файл для анализа оптимального числа кластеров:')
    await state.set_state(ClusteringState.waiting_for_auto_clusters)
    await callback.answer()


# Обработчик файла для автоопределения кластеров
@router.message(ClusteringState.waiting_for_auto_clusters, F.document)
async def handle_auto_clusters_file(message: Message, state: FSMContext):
    file = await message.bot.download(message.document)
    file_bytes = file.read()

    # Строим график методом локтя
    image_data, error = plot_clusters_count(file_bytes)

    if error:
        await message.answer(error, reply_markup=main_kb)
    else:
        await message.answer_photo(
            types.BufferedInputFile(image_data, filename='optimal_k.png'),
            caption='График для определения оптимального числа кластеров:',
            reply_markup=main_kb
        )

    await state.clear()

# Обработчик загрузки файла
@router.message(ClusteringState.waiting_for_file, F.document)
async def handle_csv(message: Message, state: FSMContext):
    data = await state.get_data()
    method = data.get('method')
    clusters = data.get('clusters')

    # Скачиваем файл
    file = await message.bot.download(message.document)
    file_bytes = file.read()

    if method in ['kmeans', 'gmm', 'hierarchical']:
        image_data, error = await process_csv(file_bytes, method, clusters)
        caption = f'Кластеризация ({method}, k={clusters}):'
    else:
        image_data, error = await process_csv(file_bytes, method)
        caption = f'Кластеризация ({method}):'

    if error:
        await message.answer(error, reply_markup=main_kb)
    else:
        await message.answer_photo(
            types.BufferedInputFile(image_data, filename='clusters.png'),
            caption=caption,
            reply_markup=main_kb
        )

    await state.clear()
