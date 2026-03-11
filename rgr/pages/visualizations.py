def page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("📈 Визуализации зависимостей в наборе данных")

    @st.cache_data
    def load_data():
        return pd.read_csv("data/processed/planets.csv")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Файл `data/processed/planets.csv` не найден.")
        return

    target_col = None
    for col in ("hazardous", "is_hazardous"):
        if col in df.columns:
            target_col = col
            break

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    SAFE_COLOR = "#2ecc71"
    DANGER_COLOR = "#e74c3c"

    def show(fig, width=0.65):
        pad = (1 - width) / 2
        _, c, _ = st.columns([pad, width, pad])
        with c:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.header("1️⃣ Гистограмма")
    hist_col = st.selectbox("Признак", numeric_cols, key="h")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.hist(df[hist_col].dropna(), bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel(hist_col)
    ax.set_ylabel("Частота")
    ax.set_title(f"Распределение «{hist_col}»")
    fig.tight_layout()
    show(fig, 0.65)

    st.header("2️⃣ Корреляционная матрица")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Корреляция признаков")
    fig.tight_layout()
    show(fig, 0.65)

    st.header("3️⃣ Диаграмма рассеяния")
    c1, c2 = st.columns(2)
    with c1:
        x = st.selectbox("X", numeric_cols, index=0, key="sx")
    with c2:
        y = st.selectbox(
            "Y", numeric_cols, index=min(1, len(numeric_cols) - 1), key="sy"
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if target_col:
        colors = df[target_col].map(
            {0: SAFE_COLOR, 1: DANGER_COLOR, False: SAFE_COLOR, True: DANGER_COLOR}
        )
        ax.scatter(df[x], df[y], c=colors, alpha=0.4, s=8, edgecolors="none")
        from matplotlib.lines import Line2D

        legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=SAFE_COLOR,
                markersize=8,
                label="Безопасный",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=DANGER_COLOR,
                markersize=8,
                label="Опасный",
            ),
        ]
        ax.legend(handles=legend, loc="upper right")
    else:
        ax.scatter(df[x], df[y], alpha=0.4, s=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    fig.tight_layout()
    show(fig, 0.7)

    st.header("4️⃣ Box Plot по классам опасности")
    box_col = st.selectbox("Признак", numeric_cols, key="b")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    if target_col:
        safe = df.loc[df[target_col].isin([0, False]), box_col].dropna()
        danger = df.loc[df[target_col].isin([1, True]), box_col].dropna()
        groups = [v for v in [safe, danger] if len(v) > 0]
        labels = ["Безопасный", "Опасный"][: len(groups)]
        colors = [SAFE_COLOR, DANGER_COLOR][: len(groups)]

        bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.45)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
    else:
        ax.boxplot(df[box_col].dropna(), patch_artist=True)
    ax.set_ylabel(box_col)
    ax.set_title(f"«{box_col}» по классу опасности")
    fig.tight_layout()
    show(fig, 0.55)

    st.header("5️⃣ Плотность распределения (KDE)")
    kde_col = st.selectbox("Признак", numeric_cols, key="k")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    if target_col:
        safe = df.loc[df[target_col].isin([0, False]), kde_col].dropna()
        danger = df.loc[df[target_col].isin([1, True]), kde_col].dropna()
        if len(safe) > 1:
            sns.kdeplot(
                safe, ax=ax, color=SAFE_COLOR, fill=True, alpha=0.3, label="Безопасный"
            )
        if len(danger) > 1:
            sns.kdeplot(
                danger, ax=ax, color=DANGER_COLOR, fill=True, alpha=0.3, label="Опасный"
            )
        ax.legend()
    else:
        sns.kdeplot(df[kde_col].dropna(), ax=ax, fill=True, alpha=0.3)
    ax.set_title(f"Плотность «{kde_col}»")
    fig.tight_layout()
    show(fig, 0.65)

    if target_col:
        st.header("6️⃣ Баланс классов")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        counts = df[target_col].value_counts()
        safe_n = counts.get(0, counts.get(False, 0))
        danger_n = counts.get(1, counts.get(True, 0))
        bars = ax.bar(
            ["Безопасный", "Опасный"],
            [safe_n, danger_n],
            color=[SAFE_COLOR, DANGER_COLOR],
            edgecolor="black",
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 10,
                str(int(b.get_height())),
                ha="center",
                fontweight="bold",
            )
        ax.set_ylabel("Количество")
        ax.set_title("Баланс классов")
        fig.tight_layout()
        show(fig, 0.45)
