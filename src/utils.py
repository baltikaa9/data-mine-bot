from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


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
