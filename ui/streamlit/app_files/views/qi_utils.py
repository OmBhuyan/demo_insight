import streamlit as st

from .feedback import Feedback


def show_dataframe(df):
    use_container_width = True
    df.reset_index(drop=True, inplace=True)
    if len(df.columns) <= 2:
        use_container_width = False
    st.header("Table")
    st.markdown("")
    st.dataframe(
        data=df.style.format(
            {col: "{:.0f}" if df[col].dtype == "int64" else "{}" for col in df.columns[:]}
        ),
        use_container_width=use_container_width,
    )


def show_feedback_buttons(key):
    col1, col2, col3, col4 = st.columns([5, 1, 1, 5], gap="small")
    with col2:
        st.button(
            "👍",
            on_click=Feedback(key=f"feedback_{key}", user_feedback="liked").feedback,
            key=f"like_{key}",
        )
    with col3:
        st.button(
            "👎",
            on_click=Feedback(key=f"feedback_{key}", user_feedback="disliked").feedback,
            key=f"dislike_{key}",
        )


def show_sql_query(query):
    st.header("Query")
    st.markdown("")
    st.code(body=query, language="sql", line_numbers=True)


def initialize_session():
    st.session_state["store"] = {
        "feedback_rerun": False,
        "question": None,
        "additional_context": None,
        "query": None,
        "plot": None,
        "summary": None,
        "table": None,
        "error_query": None,
        "error_plot": None,
        "error_summary": None,
        "feedback_plot": None,
        "feedback_query": None,
        "feedback_summary": None,
        "index": 0,
    }
