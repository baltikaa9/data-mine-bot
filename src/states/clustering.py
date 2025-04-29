from aiogram.fsm.state import StatesGroup, State


class Clustering(StatesGroup):
    waiting_for_method = State()
    waiting_for_clusters = State()
    waiting_for_file = State()
    waiting_for_auto_clusters = State()