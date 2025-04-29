from io import BytesIO
from typing import Any, Coroutine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


async def process_csv(
        file_bytes: bytes,
        method: str,
        n_clusters: int | None = None
) -> tuple[bytes | None, str | None]:
    '''Обрабатывает CSV и возвращает PNG-график кластеров.'''
    try:
        # Чтение данных
        df: pd.DataFrame = pd.read_csv(BytesIO(file_bytes))
        X: np.ndarray = df.select_dtypes(include=[np.number]).values

        if X.shape[0] < 3:
            return None, '❌ Нужно минимум 3 строки данных'

        optimal_k, ks, bic = find_optimal_clusters(X)

        # Выбор модели
        model: BaseEstimator
        match method:
            case 'kmeans':
                model = KMeans(n_clusters=n_clusters or optimal_k)
            case 'gmm':
                model = GaussianMixture(n_components=n_clusters or optimal_k)
            case 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=8)
            case 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters or optimal_k)
            case 'meanshift':
                model = MeanShift()
            case _:
                return None, '❌ Неизвестный метод'

        # Кластеризация
        clusters: np.ndarray = model.fit_predict(X)

        # Визуализация
        fig: plt.Figure = plt.figure(figsize=(10, 6))
        ax: plt.Axes = fig.add_subplot(111)

        # PCA для многомерных данных
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis: np.ndarray = pca.fit_transform(X)
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
        else:
            X_vis = X
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1] if X.shape[1] > 1 else '')

        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax.legend(*scatter.legend_elements(), title='Кластеры')
        plt.title(f'Кластеризация ({method})')

        # Конвертация в байты
        buf: BytesIO = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue(), None

    except Exception as e:
        return None, f'❌ Ошибка: {str(e)}'

def plot_clusters_count(file_bytes: bytes):
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        X = df.select_dtypes(include=[np.number]).values

        if X.shape[0] < 3:
            return None, "❌ Нужно минимум 3 строки данных"

        _, ks, bic = find_optimal_clusters(X)

        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ks, bic, marker='o')
        ax.set_xlabel('Количество кластеров')
        ax.set_ylabel('BIC')
        ax.set_title('Метод локтя для определения оптимального k')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue(), None

    except Exception as e:
        return None, f"❌ Ошибка: {str(e)}"


def find_optimal_clusters(data: np.ndarray, max_k: int = 10) -> tuple[int, np.ndarray, list]:
    bic = []
    ks = np.arange(1, max_k)
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full')
        gmm.fit(data)
        bic.append(gmm.bic(data))

    return ks[np.argmin(bic)], ks, bic


from sklearn.decomposition import PCA


async def process_classification(
        file_bytes: bytes,
        target_column: str,
        test_size: float = 0.2,
        method: str = 'logreg'
) -> tuple[None, str, float] | tuple[list[bytes], None, float | int]:
    """Обработка CSV для классификации + визуализация"""
    try:
        # Чтение данных
        df = pd.read_csv(BytesIO(file_bytes))

        # Проверка целевой колонки
        if target_column not in df.columns:
            return None, f"❌ Колонка '{target_column}' не найдена", 0.0

        # Выделение признаков и целевой переменной
        X = df.drop(target_column, axis=1).select_dtypes(include=[np.number])
        y = df[target_column]

        # 1. Обработка пропусков
        initial_rows = X.shape[0]
        X = X.dropna()
        y = y[X.index]
        dropped_na = initial_rows - X.shape[0]

        # 2. Удаление выбросов через Z-score
        z_scores = zscore(X)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        X = X[filtered_entries]
        y = y[filtered_entries]
        dropped_outliers = filtered_entries.size - np.sum(filtered_entries)

        # Проверка минимального размера данных
        if X.shape[0] < 50:
            return None, "❌ После очистки осталось слишком мало данных (<50 строк)", 0.0, {}

        # 3. Нормализация данных
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size)

        # Обучение модели
        model = None
        if method == 'logreg':
            model = LogisticRegression(max_iter=1000)
        elif method == 'random_forest':
            model = RandomForestClassifier()
        elif method == 'svm':
            model = SVC()
        elif method == 'knn':
            model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Создаём два отдельных изображения
        images = []

        # 1. Матрица ошибок
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax1)
        ax1.set_title("Матрица ошибок")
        ax1.set_xlabel("Предсказанные классы")
        ax1.set_ylabel("Истинные классы")
        buf1 = BytesIO()
        plt.savefig(buf1, format='png', bbox_inches='tight')
        buf1.seek(0)
        images.append(buf1.getvalue())
        plt.close(fig1)

        # 2. Распределение классов через PCA
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        scatter = ax2.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y_pred,
            cmap='viridis',
            alpha=0.6
        )
        ax2.set_title("Распределение классов (PCA)")
        ax2.set_xlabel("Главная компонента 1")
        ax2.set_ylabel("Главная компонента 2")
        plt.colorbar(scatter, ax=ax2, label="Класс")
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        images.append(buf2.getvalue())
        plt.close(fig2)

        return images, None, accuracy

    except Exception as e:
        return None, f"❌ Ошибка: {str(e)}", 0.0
