import streamlit as st

class Feedback:

    def __init__(
            self,
            key,
            user_feedback
        ):
        self.key           = key
        self.user_feedback  = user_feedback

    def feedback(self):
        st.session_state["store"]["feedback_rerun"] = True
        st.session_state.update(
            {
                self.key: self.user_feedback
            }
        )
