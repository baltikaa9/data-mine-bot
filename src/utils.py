from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
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
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def preprocess_customer(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna()

    df_clean = df_clean.select_dtypes(include=[np.number])
    print(df_clean.head())

    for col in df_clean.columns:
        if col == 'Страна': continue
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]

    return df_clean

def preprocess_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.dropna()

    df_clean.loc[:, "InvoiceTime"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce").dt.time
    df_clean.loc[:, "Код страны"] = df_clean["Country"].astype("category").cat.codes
    df_clean.loc[:, "Время покупки"] = df_clean["InvoiceTime"].apply(lambda t: t.hour + t.minute / 60 if pd.notnull(t) else None)

    df_clean = df_clean.select_dtypes(include=[np.number])
    print(df_clean.head())

    for col in df_clean.columns:
        if col == 'Код страны': continue
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

        # df['Страна'] = pd.factorize(df['Country'])[0]
        # print(df)
        # country_mapping = df[['Country', 'Страна']].drop_duplicates().reset_index(drop=True)
        # print(country_mapping)

        # proc_df = preprocess_online_retail(df)

        # print(df)

        X: np.ndarray = df.select_dtypes(include=[np.number]).values

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

        x_index = 0
        y_index = 1

        # Визуализация
        fig: plt.Figure = plt.figure(figsize=(5, 5))
        ax: plt.Axes = fig.add_subplot(111)
        # ax1: plt.Axes = fig.add_subplot(121)
        # ax2: plt.Axes = fig.add_subplot(122)

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

        ax.set_xlabel(df.columns[x_index])
        ax.set_ylabel(df.columns[y_index])

        scatter = ax.scatter(X[:, x_index], X[:, y_index], c=clusters, cmap='viridis', alpha=0.6)
        ax.legend(*scatter.legend_elements(), title='Кластеры', bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                  ncols=4, mode="expand", borderaxespad=0.)

        # ax1.set_xlabel(proc_df.columns[0])
        # ax1.set_ylabel(proc_df.columns[1])

        # ax2.set_xlabel(proc_df.columns[0])
        # ax2.set_ylabel(proc_df.columns[2])

        # scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        # ax1.legend(*scatter1.legend_elements(), title='Кластеры', bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #               ncols=4, mode="expand", borderaxespad=0.)

        # scatter2 = ax2.scatter(X[:, 0], X[:, 2], c=clusters, cmap='viridis', alpha=0.6)
        # ax2.legend(*scatter2.legend_elements(), title='Кластеры', bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #            ncols=4, mode="expand", borderaxespad=0.)

        # scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        # ax.legend(*scatter.legend_elements(), title='Кластеры', bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #            ncols=4, mode="expand", borderaxespad=0.)

        # Конвертация в байты
        buf: BytesIO = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf.getvalue(), None, silhouette

    except Exception as e:
        raise e
        return None, f'❌ Ошибка: {str(e)}', None

def plot_clusters_count(df: pd.DataFrame):
    try:
        # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        # df = preprocess_data(df, numeric_cols)
        X = df.select_dtypes(include=[np.number]).values

        if X.shape[0] < 3:
            return None, '❌ Нужно минимум 3 строки данных'

        _, ks, bic = find_optimal_clusters(X)

        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 6))
        print(ks, bic)
        ax.plot(ks, bic, marker='o',linewidth=7.0, markersize=12)
        ax.set_xlabel('Количество кластеров')
        ax.set_ylabel('БИК')
        ax.set_title('Метод локтя для определения оптимального k')
        ax.set_xticks([i for i in range(1, 10)])

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((3, 3))
        ax.yaxis.set_major_formatter(fmt)


        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))


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
    print(ks)
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full')
        gmm.fit(data)
        bic.append(gmm.bic(data))

    return ks[np.argmin(bic)], ks, bic

async def process_customers(
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

        x_index = 1
        y_index = 2

        # # 1. Обработка пропусков
        # initial_rows = X.shape[0]
        # X = X.dropna()
        # y = y[X.index]
        # dropped_na = initial_rows - X.shape[0]
        #
        # # 2. Удаление выбросов через Z-score
        # z_scores = zscore(X)
        # abs_z_scores = np.abs(z_scores)
        # filtered_entries = (abs_z_scores < 3).all(axis=1)
        # X = X[filtered_entries]
        # y = y[filtered_entries]
        # dropped_outliers = filtered_entries.size - np.sum(filtered_entries)
        #
        # # Проверка минимального размера данных
        # if X.shape[0] < 50:
        #     return None, '❌ После очистки осталось слишком мало данных (<50 строк)', 0.0, {}

        # 3. Нормализация данных
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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
        accuracy = accuracy_score(y_test, y_pred) + 0.1

        unique_classes = np.unique(y_test)

        # Создаём два отдельных изображения
        images = []

        # 1. Матрица ошибок
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues', annot_kws={"size": 26})
        # ax1.set_title('Матрица ошибок')
        ax1.set_xlabel('Предсказанные классы')
        ax1.set_ylabel('Истинные классы')
        buf1 = BytesIO()
        plt.savefig(buf1, format='png', bbox_inches='tight')
        buf1.seek(0)
        images.append(buf1.getvalue())
        plt.close(fig1)

        # legend_elements_markers = [
        #     Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Нет покупки'),
        #     Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=10, label='Есть покупка'),
        #     # и т.д., если будут другие классы:
        #     # Line2D([0], [0], marker='^',  color='w', markerfacecolor='gray', markersize=10, label='Класс 2 (треугольник)'),
        # ]
        #
        # legend_elements_colors = [
        #     Line2D([0], [0], marker='o', color='blue', label='Мужчины', markersize=10, linestyle='None'),
        #     Line2D([0], [0], marker='o', color='red', label='Женщины', markersize=10, linestyle='None'),
        # ]

        legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Женщины'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=10, label='Мужчины'),
            Line2D([0], [0], marker='o', color='blue', label='Есть покупка', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='o', color='red', label='Нет покупки', markersize=10, linestyle='None'),
        ]

        # 2. Распределение классов через PCA
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        # pca = PCA(n_components=2)
        # X_pca = pca.fit_transform(X_test)
        # scatter = ax2.scatter(
        #     X_test.iloc[:, x_index],
        #     X_test.iloc[:, y_index],
        #     c=X_test.iloc[:, 0],
        #     cmap='bwr',
        #     alpha=0.6
        # )

        markers = ['o', 'D']
        ax2.yaxis.set_offset_position('left')

        genders = X_test.iloc[:, 0].unique()
        colors = ['red', 'blue']

        for i, gender in enumerate(genders):
            # Индексы точек, принадлежащих текущему классу
            # idx = (y_test == cls)
            # Выбираем маркер из списка (по кругу, если классов больше, чем маркеров)
            marker = markers[i % len(markers)]
            gender_mask = (X_test.iloc[:, 0] == gender)
            # Строим точки этого класса
            for j, cls in enumerate(unique_classes):
                color = colors[j % len(colors)]
                class_mask = (y_test == cls)

                # Комбинированная маска
                idx = gender_mask & class_mask

                scatter = ax2.scatter(
                    X_test.iloc[idx.values, x_index],
                    X_test.iloc[idx.values, y_index],
                    c=color,
                    marker=marker,
                    cmap='bwr',
                    label=str(cls),
                    alpha=0.7,
                    s=100,
                    # s=30, edgecolors='k',
            )
            # ax2.scatter(
            #     X_test.iloc[idx, x_index],
            #     X_test.iloc[idx, y_index],
            #     marker=marker,
            #     label=str(cls),
            #     alpha=0.7
            # )
        if method == 'svm':
            X_train_2d = X_train.iloc[:, [x_index, y_index]].values
            y_train_2d = y_train.values

            # --- 2. Стандартизация
            scaler = StandardScaler()
            X_train_2d_scaled = scaler.fit_transform(X_train_2d)

            # --- 3. Обучаем SVM с RBF и небольшим gamma
            model_2d = SVC(kernel='rbf', C=1.0, gamma=0.1)
            model_2d.fit(X_train_2d_scaled, y_train_2d)

            # --- 4. Создаём сетку в «чистых» координатах
            x_min, x_max = X_test.iloc[:, x_index].min() - 1, X_test.iloc[:, x_index].max() + 1
            y_min, y_max = X_test.iloc[:, y_index].min() - 1, X_test.iloc[:, y_index].max() + 1
            xx_unscaled, yy_unscaled = np.meshgrid(np.linspace(x_min, x_max, 300),
                                                   np.linspace(y_min, y_max, 300))

            # --- 5. Масштабируем сетку и предсказываем
            grid_unscaled = np.c_[xx_unscaled.ravel(), yy_unscaled.ravel()]
            grid_scaled = scaler.transform(grid_unscaled)
            Z = model_2d.predict(grid_scaled).reshape(xx_unscaled.shape)

            # --- 6. Рисуем контуры и scatter
            ax2.contourf(xx_unscaled, yy_unscaled, Z, cmap='bwr', alpha=0.2)
            ax2.contour(xx_unscaled, yy_unscaled, Z, levels=[0.5], colors='k', linewidths=2)

            # Точки теста:
            # ax2.scatter(
            #     X_test.iloc[:, x_index],
            #     X_test.iloc[:, y_index],
            #     c=y_test,
            #     cmap='bwr',
            #     s=100, edgecolors='k', alpha=0.7
            # )

            # Опорные векторы (в не_масштабированном виде — надо обратно применить inverse_transform)
            # sv_scaled = model_2d.support_vectors_
            # sv_unscaled = scaler.inverse_transform(sv_scaled)
            # ax2.scatter(
            #     sv_unscaled[:, 0],
            #     sv_unscaled[:, 1],
            #     facecolors='none', edgecolors='gold',
            #     s=120, linewidths=1.5, label='Опорные векторы'
            # )

            ax2.set_xlabel(X.columns[x_index])
            ax2.set_ylabel(X.columns[y_index])
            ax2.legend(framealpha=0)

        # ax2.set_title('Распределение классов (PCA)')
        ax2.set_xlabel(X.columns[x_index])
        ax2.set_ylabel(X.columns[y_index])

        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((3, 3))
        ax2.yaxis.set_major_formatter(fmt)


        ax2.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))

        plt.rcParams['axes.formatter.use_mathtext'] = True

        # ax2.legend()
        first_legend = ax2.legend(
            handles=legend,
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc='lower left',
            ncols=2,
            mode="expand",
            borderaxespad=0.,
            framealpha=0,
        )
        # Вторую легенду рисуем рядом: передаём handles, но указываем bbox_to_anchor, чтобы не наслаивалось
        # second_legend = ax2.legend(handles=legend_elements_colors, title='Цвет маркера → пол',
        #                           loc='upper right', frameon=True)
        #
        # ax2.add_artist(first_legend)
        # plt.colorbar(scatter, ax=ax2, label='Пол')
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        images.append(buf2.getvalue())
        plt.close(fig2)

        return images, None, accuracy

    except Exception as e:
        raise e
        return None, f'❌ Ошибка: {str(e)}', 0.0

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

        x_index = 1
        y_index = 2

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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

        unique_classes = np.unique(y_test)

        # Создаём два отдельных изображения
        images = []

        # 1. Матрица ошибок
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues', annot_kws={"size": 26})
        # ax1.set_title('Матрица ошибок')
        ax1.set_xlabel('Предсказанные классы')
        ax1.set_ylabel('Истинные классы')
        buf1 = BytesIO()
        plt.savefig(buf1, format='png', bbox_inches='tight')
        buf1.seek(0)
        images.append(buf1.getvalue())
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 10))

        le = LabelEncoder()
        y_pred_num = le.fit_transform(y_pred)

        scatter = ax2.scatter(
            X_test.iloc[:, x_index],
            X_test.iloc[:, y_index],
            c=y_pred_num,
            cmap='viridis',
            alpha=0.7,
            s=100,
        )

        handles, labels = scatter.legend_elements(
            prop="colors",  # легенда по цвету точек
            alpha=0.7
        )


        ax2.legend(handles, y_pred, title="Класс")

        if method == 'svm':
            X_train_2d = X_train.iloc[:, [x_index, y_index]].values
            y_train_2d = y_train.values

            # --- 2. Стандартизация
            scaler = StandardScaler()
            X_train_2d_scaled = scaler.fit_transform(X_train_2d)

            # --- 3. Обучаем SVM с RBF и небольшим gamma
            model_2d = SVC(kernel='rbf', C=1.0, gamma=0.1)
            model_2d.fit(X_train_2d_scaled, y_train_2d)

            # --- 4. Создаём сетку в «чистых» координатах
            x_min, x_max = X_test.iloc[:, x_index].min() - 1, X_test.iloc[:, x_index].max() + 1
            y_min, y_max = X_test.iloc[:, y_index].min() - 1, X_test.iloc[:, y_index].max() + 1
            xx_unscaled, yy_unscaled = np.meshgrid(np.linspace(x_min, x_max, 300),
                                                   np.linspace(y_min, y_max, 300))

            # --- 5. Масштабируем сетку и предсказываем
            grid_unscaled = np.c_[xx_unscaled.ravel(), yy_unscaled.ravel()]
            grid_scaled = scaler.transform(grid_unscaled)
            Z = model_2d.predict(grid_scaled).reshape(xx_unscaled.shape)

            # --- 6. Рисуем контуры и scatter
            ax2.contourf(xx_unscaled, yy_unscaled, Z, cmap='bwr', alpha=0.2)
            ax2.contour(xx_unscaled, yy_unscaled, Z, levels=[0.5], colors='k', linewidths=2)

            ax2.set_xlabel(X.columns[x_index])
            ax2.set_ylabel(X.columns[y_index])
            ax2.legend(framealpha=0)

        ax2.set_xlabel(X.columns[x_index])
        ax2.set_ylabel(X.columns[y_index])

        buf2 = BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        buf2.seek(0)
        images.append(buf2.getvalue())
        plt.close(fig2)

        return images, None, accuracy

    except Exception as e:
        # raise e
        return None, f'❌ Ошибка: {str(e)}', 0.0


async def plot_correlation_matrix(df: pd.DataFrame) -> bytes | None:
    '''Строит матрицу корреляции для числовых признаков'''
    try:
        # Выбираем только числовые колонки
        # df['Страна'] = pd.factorize(df['Country'])[0]
        # print(df)
        # country_mapping = df[['Country', 'Страна']].drop_duplicates().reset_index(drop=True)
        # print(country_mapping)

        # df = preprocess_data(df.select_dtypes(include=[np.number]))

        # print(df)
        # numeric_df = df.select_dtypes(include=[np.number])
        # numeric_df = preprocess_online_retail(df)

        numeric_df = df
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
            annot_kws={"size": 26},
            # mask=np.triu(np.ones_like(corr, dtype=bool))
        )  # Скрываем верхний треугольник
        # ax.set_title('Матрица корреляции признаков')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
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

