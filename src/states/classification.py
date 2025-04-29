from aiogram.fsm.state import StatesGroup, State


class Classification(StatesGroup):
    waiting_for_method = State()
    waiting_for_file = State()
    waiting_for_target = State()
