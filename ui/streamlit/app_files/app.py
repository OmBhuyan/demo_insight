from html_components import toggle_html_navbar
from views.footer import footer
from utils import get_current_route, inject_custom_css, navigation, set_page_config
import streamlit.components.v1 as components  # Import Streamlit
import streamlit as st
import os


# Get selected page from query parameters
route = get_current_route()
# Set page config wrt selected pages
set_page_config(route)


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Inject custom css into the page
inject_custom_css()

# Hide default streamlit header, footer and main menu
# hide_default_streamlit_style()
# Run HTML nav bar page layout
toggle_html_navbar()


# Toggle navigation wrt selected page
navigation(route)

footer()

# Adding js functions to all components
def read_html():
        with open(os.path.join(os.getcwd(), "app_files", "assets",'main.html')) as f:
            return f.read()
components.html(
        read_html(),width=0,height=0
    )




