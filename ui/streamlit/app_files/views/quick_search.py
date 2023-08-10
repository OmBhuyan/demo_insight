import os
import textwrap

import streamlit as st
from st_config import STConfig

from query_insights.api import QueryInsights, config_init

from .qi_utils import (
    initialize_session,
    show_dataframe,
    show_feedback_buttons,
    show_sql_query,
)
from .traverse import Traverse

config = STConfig()
traverse = Traverse()


def get_question():
    st.session_state["store"].update({"question": st.session_state.question})


def get_context():
    st.session_state["store"].update({"additional_context": st.session_state.additional_context})


def load_view():
    # creating Query Insights Object
    user_config, data_config, model_config, debug_config = config_init(config.UI_USER_CONFIG_PATH, config.UI_DATA_CONFIG_PATH, config.UI_MODEL_CONFIG_PATH, config.UI_DEBUG_CONFIG_PATH)

    qi = QueryInsights(
        user_config=user_config,
        data_config=data_config,
        model_config=model_config,
        debug_config=debug_config,
        api_key=os.environ.get("OPENAI_API_KEY"),
        logging_level="INFO",
    )

    # Variables declaration
    skip_plot, skip_summary = False, False
    track1_output_query, track1_output_table, track1_output_table_dict = None, None, None
    question, additional_context = None, None

    # st.markdown("Start")
    # st.write(st.session_state)
    if not st.session_state:
        initialize_session()
        question = config.DEFAULT_QUESTION
        additional_context = config.DEFAULT_ADDITIONAL_QUESTION
    else:
        question = st.session_state["store"].get("question")
        additional_context = st.session_state["store"].get("additional_context")

    coll1, buff1, coll2 = st.columns([8, 2, 1])

    with coll1:
        st.markdown(
            f"""
                    <div class="indiCardBody">
                    <div class="indiCardtitle">
                    <h1 style="margin-bottom: 10px;
                    font-family:  sans-serif;
                    font-size: 38px;
                    font-weight:600;
                    font-style: normal;    line-height: 44px;color: #333">{config.TITLE}</h1>
                        </div>
                        <div class="indiCardtext">
                        <span>  {config.SUBTITLE}
                    </span>

                    </div>
                """,
            unsafe_allow_html=True,
        )

    with coll2:
        st.write(f"[Conversation App >]({config.GRADIO_URL})")

    st.markdown("")

    with st.container():
        col1, col2, col3 = st.columns([32, 1, 1.5], gap="small")
        with col2:
            st.button("<", on_click=traverse.get_prev, key="prev")
        with col3:
            st.button("\>", on_click=traverse.get_next, key="next")

        question = st.text_area(
            label="Enter your Question 👇", value=question, on_change=get_question, key="question"
        )
        additional_context = st.text_area(
            label="Additional Business Context 👇",
            value=additional_context,
            on_change=get_context,
            key="additional_context",
        )

        st.session_state["store"].update(
            {"question": question, "additional_context": additional_context}
        )

        # st.markdown("Middle")
        # st.write(st.session_state)

        with st.form(key="sub_form"):
            submit_button = st.form_submit_button(label="Search")

        if submit_button or st.session_state["store"].get("feedback_rerun", False):
            if question:
                with st.spinner("__Loading Model and Searching__"):
                    try:
                        if st.session_state["store"].get("feedback_rerun"):
                            st.session_state["store"]["feedback_rerun"] = False
                            # First plot logic
                            if st.session_state["store"]["error_plot"]:
                                # st.error(st.session_state["store"]["error_plot"])
                                if not st.session_state["store"]["table"].empty:
                                    show_dataframe(st.session_state["store"]["table"])
                            elif st.session_state["store"]["plot"]:
                                st.session_state["store"]["plot"].update_layout(
                                    width=1250, height=500
                                )
                                st.plotly_chart(st.session_state["store"]["plot"])
                                show_feedback_buttons(key="plot")

                            # summary logic
                            if st.session_state["store"]["error_summary"]:
                                st.header("Summary")
                                st.markdown("")
                                st.error(st.session_state["store"]["error_summary"])
                            elif st.session_state["store"]["summary"]:
                                st.header("Summary")
                                st.markdown("")
                                st.text(st.session_state["store"]["summary"])
                                show_feedback_buttons(key="summary")

                            # query logic
                            if st.session_state["store"]["error_query"]:
                                # st.header("Query")
                                # st.markdown("")
                                st.error(st.session_state["store"]["error_query"])
                            if st.session_state["store"]["query"]:
                                show_sql_query(query=st.session_state["store"]["query"])
                                show_feedback_buttons(key="query")

                        else:
                            all_outputs = []
                            index = st.session_state["store"].get("index")
                            initialize_session()
                            st.session_state["store"].update(
                                {
                                    "question": question,
                                    "additional_context": additional_context,
                                    "index": index,
                                }
                            )

                            # Getting Query logic
                            response = qi.text_to_query(
                                question=question, additional_context=additional_context
                            )

                            print(f"\n\ntext_to_query: {response}\n\n")
                            all_outputs.append(response.get("status"))
                            if response.get("status") == "failure":
                                skip_plot, skip_summary = True, True
                                error_msg, query = response.get("output")
                                st.session_state["store"].update(
                                    {"query": query, "error_query": error_msg}
                                )
                                # st.header("Query")
                                # st.markdown("")
                                st.error(error_msg)
                                show_sql_query(query=query)
                                show_feedback_buttons(key="query")

                            elif response.get("status") == "skip":
                                skip_plot = True
                                skip_reason = response.get("output")[0]
                                st.header("Query")
                                st.markdown("")
                                st.error(skip_reason)
                                st.session_state["store"].update({"error_query": skip_reason})
                            else:
                                print("query success")
                                (
                                    track1_output_query,
                                    track1_output_table,
                                    track1_output_table_dict,
                                ) = response.get("output")
                                print("query success....completed")

                            # Getting plot logic
                            if not skip_plot:
                                print("inside plot")
                                response = qi.query_to_chart(
                                    question=question,
                                    additional_context=additional_context,
                                    track1_output_table=track1_output_table,
                                    track1_output_table_dict=track1_output_table_dict,
                                )
                                print(f"\n\query_to_chart: {response}\n\n")
                                all_outputs.append(response.get("status"))
                                if response.get("status") == "skip":
                                    error_plot = response.get("output")
                                    if isinstance(error_plot, str):
                                        # st.header("Plot")
                                        # st.error(error_plot)
                                        # st.write(error_plot)
                                        st.session_state["store"].update(
                                            {"error_plot": error_plot}
                                        )
                                    else:
                                        print("Not here\n\n")
                                        if not error_plot[1].empty:
                                            show_dataframe(error_plot[1])
                                            print("here df")
                                            st.session_state["store"].update(
                                                {"table": error_plot[1]}
                                            )
                                        st.session_state["store"].update(
                                            {"error_plot": error_plot}
                                        )
                                elif response.get("status") == "failure":
                                    error_plot, table = response.get("output")
                                    # st.header("Plot")
                                    # st.error(error_plot)
                                    show_dataframe(table)
                                    st.session_state["store"].update(
                                        {"error_plot": error_plot, "table": table}
                                    )
                                else:
                                    plot, table = response.get("output")
                                    plot.update_layout(width=1250, height=500)
                                    st.plotly_chart(plot)
                                    show_feedback_buttons(key="plot")
                                    st.session_state["store"].update(
                                        {"plot": plot, "table": table}
                                    )
                            else:
                                all_outputs.append("skip")

                            # Getting summary logic
                            if not skip_summary:
                                response = qi.table_to_insights(
                                    question=question,
                                    track1_output_table=track1_output_table,
                                    track1_output_table_dict=track1_output_table_dict,
                                )
                                print(f"\n\nquery_to_insights\n\n: {response}")
                                all_outputs.append(response.get("status"))
                                if (
                                    response.get("status") == "skip"
                                    or response.get("status") == "failure"
                                ):
                                    st.header("Summary")
                                    st.markdown("")
                                    st.error(response.get("output"))
                                    st.session_state["store"].update(
                                        {"error_summary": response.get("output")}
                                    )
                                else:
                                    st.header("Summary")
                                    st.markdown("")
                                    st.text(response.get("output"))
                                    st.session_state["store"].update(
                                        {"summary": response.get("output")}
                                    )
                                    show_feedback_buttons(key="summary")
                            else:
                                all_outputs.append("skip")

                            # show Query
                            if track1_output_query:
                                show_sql_query(track1_output_query)
                                show_feedback_buttons(key="query")
                                st.session_state["store"].update({"query": track1_output_query})
                            alltracks_status = all_outputs
                            feedback = None
                            qi.update_knowledgebase(alltracks_status, feedback)
                            # st.write(st.session_state["store"])

                        # st.markdown("End")
                        # st.write(st.session_state)

                    except Exception as e:
                        print(e)
            else:
                st.warning("No Input Received", icon="⚠️")
