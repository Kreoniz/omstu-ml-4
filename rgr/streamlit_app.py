import streamlit as st
from app_pages import developer_info, dataset_info, visualizations, prediction

st.set_page_config(
    page_title="ML Dashboard — Near Earth Objects",
    page_icon="☄️",
    layout="wide",
)

PAGES = {
    "Разработчик": developer_info.page,
    "Набор данных": dataset_info.page,
    "Визуализации": visualizations.page,
    "Предсказание": prediction.page,
}

st.sidebar.title("☄️ NEO Dashboard")
st.sidebar.markdown("---")
selection = st.sidebar.radio("Навигация", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.info(
    "Дашборд для анализа околоземных объектов и предсказания их опасности")

PAGES[selection]()
