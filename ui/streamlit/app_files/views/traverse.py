import pandas as pd
import streamlit as st
from st_config import STConfig

from .qi_utils import initialize_session

config = STConfig()


class Traverse:
    def __init__(self):
        try:
            self.knowledge_base = pd.read_excel(config.NEXT_PREV_BUTTON_PATH)
            self.knowledge_base = self.knowledge_base.fillna("")
            self.end = self.knowledge_base.shape[0] - 1

        except:
            raise Exception("Missing valid Knowledge Base.")

    def get_next(self):
        current_index = st.session_state["store"].get("index")
        initialize_session()
        st.session_state["store"]["is_button_click"] = True
        if current_index < self.end:
            current_index += 1
            st.session_state["store"].update({"index": current_index})
            (
                st.session_state["store"]["question"],
                st.session_state["store"]["additional_context"],
            ) = self.knowledge_base.loc[
                current_index, ["Question", "Additional business context"]
            ].values
        else:
            st.session_state["store"].update({"index": self.end})
            (
                st.session_state["store"]["question"],
                st.session_state["store"]["additional_context"],
            ) = self.knowledge_base.loc[
                self.end, ["Question", "Additional business context"]
            ].values

    def get_prev(self):
        current_index = st.session_state["store"].get("index")
        initialize_session()
        st.session_state["store"]["is_button_click"] = True
        if current_index > 0:
            current_index -= 1
            st.session_state["store"].update({"index": current_index})
            (
                st.session_state["store"]["question"],
                st.session_state["store"]["additional_context"],
            ) = self.knowledge_base.loc[
                current_index, ["Question", "Additional business context"]
            ].values
        else:
            st.session_state["store"].update({"index": 0})
            (
                st.session_state["store"]["question"],
                st.session_state["store"]["additional_context"],
            ) = self.knowledge_base.loc[0, ["Question", "Additional business context"]].values
            (
                st.session_state["store"]["question"],
                st.session_state["store"]["additional_context"],
            ) = self.knowledge_base.loc[0, ["Question", "Additional business context"]].values
