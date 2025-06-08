from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

def get_main_menu_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text='Кластеризация')],
            [KeyboardButton(text='Классификация')],
        ],
        resize_keyboard=True
    )


def get_clustering_methods_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text='𝑘-средних', callback_data='kmeans'),
                InlineKeyboardButton(text='Гауссова смесь', callback_data='gmm'),
            ],
            [
                InlineKeyboardButton(text='DBSCAN', callback_data='dbscan'),
                InlineKeyboardButton(text='Иерархическая', callback_data='hierarchical'),
            ],
            [
                InlineKeyboardButton(text='Средний сдвиг', callback_data='meanshift'),
                InlineKeyboardButton(text='🔍 Определить оптимальное k', callback_data='auto_clusters'),
            ],
        ]
    )

def get_online_retail() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text='Загрузить набор данных "Online Retail"', callback_data='online_retail'),
            ]
        ]
    )

def get_customer_behavior() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text='Загрузить набор данных "Customer Behavior"', callback_data='customer_behavior'),
            ]
        ]
    )

def get_classification_methods_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Логистическая регрессия', callback_data='logreg'),
             InlineKeyboardButton(text='Случайный лес', callback_data='random_forest')],
            [InlineKeyboardButton(text='SVM', callback_data='svm'),
             InlineKeyboardButton(text='KNN', callback_data='knn')],
            [InlineKeyboardButton(text='Назад', callback_data='back')]
        ]
    )
