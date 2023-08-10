import os
import streamlit as st
from views import quick_search




def inject_custom_css():
    """
    Inject css for html navbar
    """
    with open(os.path.join(os.getcwd(), "app_files", "assets", "styles.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def get_current_route() -> str:
    """
    Get current page from url

    Returns
    -------
    str
        Query parameter for current page
    """
    try:
        return st.experimental_get_query_params()["nav"][0]
    except Exception:
        return None


def set_page_config(route: str):
    """
    Sets page config layout to wide with page title corresponding to the page selected

    Parameters
    ----------
    route : str
        Current page query parameter
    """


    st.set_page_config(layout="wide", page_title="Query Insights")
    st.set_option("deprecation.showPyplotGlobalUse", False)


def navigation(route: str):
    """
    Loads page view with respect to the selected page from query parameters

    Parameters
    ----------
    route : str
        Current query parameter of the page
    """

    quick_search.load_view()
