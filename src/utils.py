from io import BytesIO
from typing import Any
from typing import Coroutine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna()

    for col in df.columns:
        if col == 'Страна': continue
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]

    return df_clean

async def process_csv(
        df: pd.DataFrame,
        method: str,
        n_clusters: int | None = None
) -> tuple[bytes | None, str | None, float | None]:
    '''Обрабатывает CSV и возвращает PNG-график кластеров.'''
    try:
        # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        # df = df.select_dtypes(include=[np.number])

        df['Страна'] = pd.factorize(df['Country'])[0]
        print(df)
        country_mapping = df[['Country', 'Страна']].drop_duplicates().reset_index(drop=True)
        print(country_mapping)

        proc_df = preprocess_data(df.select_dtypes(include=[np.number]))

        print(df)

        X: np.ndarray = proc_df.select_dtypes(include=[np.number]).values

        if X.shape[0] < 3:
            return None, '❌ Нужно минимум 3 строки данных', None

        if not n_clusters:
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
                return None, '❌ Неизвестный метод', None

        # Кластеризация
        clusters: np.ndarray = model.fit_predict(X)

        silhouette = silhouette_score(X, clusters)

        # Визуализация
        fig: plt.Figure = plt.figure(figsize=(16, 8))
        ax1: plt.Axes = fig.add_subplot(121)
        ax2: plt.Axes = fig.add_subplot(122)

        # PCA для многомерных данных
        # if X.shape[1] > 2:
        #     pca = PCA(n_components=2)
        #     X_pca: np.ndarray = pca.fit_transform(X)
        #
        #     # Для Component 1
        #     ax.set_xlabel('$PCA_1$')
        #
        #     # Для Component 2
        #     ax.set_ylabel('$PCA_2$')
        # else:
        #     X_pca = X
        #     ax.set_xlabel(df.columns[0])
        #     ax.set_ylabel(df.columns[1] if X.shape[1] > 1 else '')

        ax1.set_xlabel(proc_df.columns[0])
        ax1.set_ylabel(proc_df.columns[1])

        ax2.set_xlabel(proc_df.columns[0])
        ax2.set_ylabel(proc_df.columns[2])

        scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax1.legend(*scatter1.legend_elements(), title='Кластеры', bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=4, mode="expand", borderaxespad=0.)

        scatter2 = ax2.scatter(X[:, 0], X[:, 2], c=clusters, cmap='viridis', alpha=0.6)
        ax2.legend(*scatter2.legend_elements(), title='Кластеры', bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncols=4, mode="expand", borderaxespad=0.)

        # Конвертация в байты
        buf: BytesIO = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue(), None, silhouette

    except Exception as e:
        return None, f'❌ Ошибка: {str(e)}', None

def plot_clusters_count(df: pd.DataFrame):
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df = preprocess_data(df, numeric_cols)
        X = df.select_dtypes(include=[np.number]).values

        if X.shape[0] < 3:
            return None, '❌ Нужно минимум 3 строки данных'

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
        return None, f'❌ Ошибка: {str(e)}'


def find_optimal_clusters(data: np.ndarray, max_k: int = 10) -> tuple[int, np.ndarray, list]:
    bic = []
    ks = np.arange(1, max_k)
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full')
        gmm.fit(data)
        bic.append(gmm.bic(data))

    return ks[np.argmin(bic)], ks, bic




async def process_classification(
        file_bytes: bytes,
        target_column: str,
        test_size: float = 0.2,
        method: str = 'logreg'
) -> tuple[None, str, float] | tuple[list[bytes], None, float | int]:
    '''Обработка CSV для классификации + визуализация'''
    try:
        # Чтение данных
        df = pd.read_csv(BytesIO(file_bytes))

        # Проверка целевой колонки
        if target_column not in df.columns:
            return None, f'❌ Колонка \'{target_column}\' не найдена', 0.0

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
            return None, '❌ После очистки осталось слишком мало данных (<50 строк)', 0.0, {}

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
        sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
        # ax1.set_title('Матрица ошибок')
        ax1.set_xlabel('Предсказанные классы')
        ax1.set_ylabel('Истинные классы')
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
            cmap='bwr',
            alpha=0.6
        )
        # ax2.set_title('Распределение классов (PCA)')
        ax2.set_xlabel('$PCA_1$')
        ax2.set_ylabel('$PCA_2$')
        plt.colorbar(scatter, ax=ax2, label='Класс')
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        images.append(buf2.getvalue())
        plt.close(fig2)

        return images, None, accuracy

    except Exception as e:
        return None, f'❌ Ошибка: {str(e)}', 0.0


async def plot_correlation_matrix(df: pd.DataFrame) -> bytes | None:
    '''Строит матрицу корреляции для числовых признаков'''
    try:
        # Выбираем только числовые колонки
        df['Страна'] = pd.factorize(df['Country'])[0]
        print(df)
        country_mapping = df[['Country', 'Страна']].drop_duplicates().reset_index(drop=True)
        print(country_mapping)

        # df = preprocess_data(df.select_dtypes(include=[np.number]))

        # print(df)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        # Строим матрицу корреляции
        corr = numeric_df.corr()
        print(corr)

        # Визуализация
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=ax,
            annot_kws={"size": 20},
            # mask=np.triu(np.ones_like(corr, dtype=bool))
        )  # Скрываем верхний треугольник
        # ax.set_title('Матрица корреляции признаков')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Конвертация в байты
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    except Exception as e:
        print(f'Ошибка при построении матрицы корреляции: {str(e)}')
        return None

