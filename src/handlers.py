from io import BytesIO

import pandas as pd
from aiogram import F, types
from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from .keyboards import get_main_menu_kb, get_clustering_methods_kb, get_classification_methods_kb
from .states.classification import Classification
from .states.clustering import Clustering
from .utils import process_csv, plot_clusters_count, process_classification, plot_correlation_matrix

router = Router()


# Обработчик /start
@router.message(Command('start'))
async def cmd_start(message: Message):
    await message.answer(
        'Выбери действие:',
        reply_markup=get_main_menu_kb()
    )

# Обработчик кнопки 'Кластеризация' (Reply)
@router.message(F.text == 'Кластеризация')
async def clustering_menu(message: Message):
    await message.answer(
        'Выбери метод кластеризации:',
        reply_markup=get_clustering_methods_kb()
    )


@router.callback_query(F.data.in_(['kmeans', 'gmm', 'hierarchical']))
async def select_method(callback: types.CallbackQuery, state: FSMContext):
    method = callback.data
    await state.update_data(method=method)
    await callback.message.edit_text(
        f'Выбран метод: {method}. Введи число кластеров (от 2 до 20):'
    )
    await state.set_state(Clustering.waiting_for_clusters)
    await callback.answer()

# Обработчик методов без указания кластеров (DBSCAN, MeanShift)
@router.callback_query(F.data.in_(['dbscan', 'meanshift']))
async def select_method_no_clusters(callback: types.CallbackQuery, state: FSMContext):
    method = callback.data
    await state.update_data(method=method, clusters=None)
    await callback.message.edit_text(
        f'Выбран метод: {method}. Теперь загрузи CSV-файл:'
    )
    await state.set_state(Clustering.waiting_for_file)
    await callback.answer()


# Обработчик ввода числа кластеров
@router.message(Clustering.waiting_for_clusters, F.text)
async def handle_clusters_input(message: Message, state: FSMContext):
    if not message.text.isdigit():
        await message.reply('❌ Введите ЧИСЛО!')
        return

    clusters = int(message.text)
    if clusters < 2 or clusters > 20:
        await message.reply('❌ Число кластеров должно быть от 2 до 20.')
        return

    await state.update_data(clusters=clusters)
    await message.answer('Теперь загрузи CSV-файл:')
    await state.set_state(Clustering.waiting_for_file)


@router.callback_query(F.data == 'auto_clusters')
async def handle_auto_clusters(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Загрузи CSV-файл для анализа оптимального числа кластеров:')
    await state.set_state(Clustering.waiting_for_auto_clusters)
    await callback.answer()


# Обработчик файла для автоопределения кластеров
@router.message(Clustering.waiting_for_auto_clusters, F.document)
async def handle_auto_clusters_file(message: Message, state: FSMContext):
    file = await message.bot.download(message.document)
    file_bytes = file.read()

    # Строим график методом локтя
    image_data, error = plot_clusters_count(pd.read_csv(BytesIO(file_bytes)))

    if error:
        await message.answer(error, reply_markup=get_main_menu_kb())
    else:
        await message.answer_photo(
            types.BufferedInputFile(image_data, filename='optimal_k.png'),
            caption='График для определения оптимального числа кластеров:',
            reply_markup=get_main_menu_kb()
        )

    await state.clear()

# Обработчик загрузки файла
@router.message(Clustering.waiting_for_file, F.document)
async def handle_csv(message: Message, state: FSMContext):
    data = await state.get_data()
    method = data.get('method')
    clusters = data.get('clusters')

    # Скачиваем файл
    file = await message.bot.download(message.document)
    file_bytes = file.read()

    try:
        df = pd.read_csv(BytesIO(file_bytes))
        corr_image = await plot_correlation_matrix(df)
        if corr_image:
            await message.answer_photo(
                types.BufferedInputFile(corr_image, filename="correlation.png"),
                caption="Матрица корреляции признаков"
            )
    except Exception as e:
        await message.answer(f"⚠️ Не удалось построить матрицу корреляции: {str(e)}")

    if method in ['kmeans', 'gmm', 'hierarchical']:
        image_data, error, silhouette = await process_csv(df, method, clusters)
        caption = f'Кластеризация ({method}, k={clusters})'
    else:
        image_data, error, silhouette = await process_csv(df, method)
        caption = f'Кластеризация ({method})'

    caption += f'\nОценка силуэта: {silhouette}'

    if error:
        await message.answer(error, reply_markup=get_main_menu_kb())
    else:
        await message.answer_photo(
            types.BufferedInputFile(image_data, filename='clusters.png'),
            caption=caption,
            reply_markup=get_main_menu_kb()
        )

    await state.clear()


# Обработчик кнопки "Классификация"
@router.message(F.text == 'Классификация')
async def classification_menu(message: Message):
    await message.answer(
        'Выбери метод классификации:',
        reply_markup=get_classification_methods_kb()
    )


# Обработчик выбора метода классификации
@router.callback_query(F.data.in_(['logreg', 'random_forest', 'svm', 'knn']))
async def select_classification_method(callback: types.CallbackQuery, state: FSMContext):
    await state.update_data(method=callback.data)
    await callback.message.edit_text("Теперь загрузи CSV-файл с данными")
    await state.set_state(Classification.waiting_for_file)
    await callback.answer()


# Обработчик ввода целевой колонки
@router.message(Classification.waiting_for_file, F.document)
async def handle_target_column(message: Message, state: FSMContext):
    file = await message.bot.download(message.document)
    await state.update_data(file=file)
    await message.answer("Введи название целевой колонки")
    await state.set_state(Classification.waiting_for_target)


# Обработчик файла для классификации
@router.message(Classification.waiting_for_target, F.text)
async def handle_classification_file(message: Message, state: FSMContext):
    data = await state.get_data()
    method = data['method']

    target = message.text.strip()

    file_bytes = data['file'].read()

    try:
        df = pd.read_csv(BytesIO(file_bytes))
        corr_image = await plot_correlation_matrix(df)
        if corr_image:
            await message.answer_photo(
                types.BufferedInputFile(corr_image, filename="correlation.png"),
                caption="Матрица корреляции признаков:"
            )
    except Exception as e:
        await message.answer(f"⚠️ Не удалось построить матрицу корреляции: {str(e)}")

    image_data, error, accuracy = await process_classification(
        file_bytes,
        target,
        method=method
    )

    if error:
        await message.answer(error, reply_markup=get_main_menu_kb())
    else:
        # Отправляем распределение классов
        await message.answer_photo(
            types.BufferedInputFile(image_data[1], filename="pca_distribution.png"),
            caption=f"Распределение классов (Точность: {accuracy:.2f})"
        )

        # Отправляем матрицу ошибок
        await message.answer_photo(
            types.BufferedInputFile(image_data[0], filename="confusion_matrix.png"),
            caption="Матрица ошибок:"
        )

    await state.clear()
