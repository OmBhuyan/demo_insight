import os
import random
import time

import gradio as gr
from feedback import Feedback
from query_insights.api import BotResponse, QueryInsights


class ChatUI:
    def __init__(self, user_config_path, data_config_path, model_config_path, debug_config_path):
        """
        initializes all the self variables

        Parameters
        ----------
        user_config_path : str
            path to the user config yaml file
        data_config_path : str
            path to the data config yaml file
        model_config_path : str
            path to the model config yaml file.
        """
        # TODO: update path to sessions folder
        # output_path = ""
        # if len(os.listdir(output_path)) > 0:
        #     ids = [int(i.split("_")[1]) for i in os.listdir(output_path)]
        #     self.id = max(ids) + 1
        # else:
        #     self.id = 1
        self.api_key = os.environ.get("OPENAI_API_KEY")
        # self.api_key = os.environ["OPENAI_API_KEY"]
        self.user_config_path = user_config_path
        self.data_config_path = data_config_path
        self.model_config_path = model_config_path
        self.debug_config_path = debug_config_path

        self.track1_query = None
        self.track1_table = None
        self.plot_ = None
        self.output_summary = None

    def get_query(self, qi, input_text, track1_status, track2_status, track3_status, history):
        """
        to call the api to generate the SQL query to the question

        Parameters
        ----------
        qi : class_instance
            class instance specific to a session
        input_text : str,
            question to be passed to model api
        track1_status : str
            keeps track of track1 status ("failure" / "skip" / "success")
        track2_status : str
            keeps track of track2 status ("skip" / "in_progress"). if "in_progress" it will run track2 else track2 will be skipped
        track3_status : str
            keeps track of track3 status ("skip" / "in_progress"). if "in_progress" it will run track3 else track3 will be skipped

        Returns
        -------
        Tuple
            updated values of bot_response, track1_query, track1_status, track2_status, track3_status, qi
        """

        track1_response = qi.text_to_query(question=input_text, additional_context=None)
        if track1_response["status"] == "success":
            track1_query, track1_table, track1_dict = track1_response["output"]

        elif track1_response["status"] == "skip":
            track1_query, track1_table = track1_response["output"]

        elif track1_response["status"] == "failure":
            error_message, track1_query = track1_response["output"]
            track1_query = (
                error_message + " \nFollowing is the generated query: \n\n" + track1_query
            )
            track1_query = track1_query.replace("': ", "'\n: ")

        skip_track2, skip_track3 = qi._skip_tracks(track1_response)

        processing_phrases = [
            "Hang tight, I'm processing your request!",
            "Just a moment while I fetch your information...",
            "I'm working on it! Sit tight!",
            "Please give us a moment to gather the necessary data.",
            "I'm crunching the numbers as we speak!",
            "I'll be with you shortly. Promise!",
            "I'm on the case! Please hold tight.",
            "One moment while I work my magic!",
        ]

        error_phrases = [
            "My apologies! Can you ask something else?",
            "Oh no! That didn't work. How about asking a different question?",
            "Sorry, I encountered an error. Maybe try a different query?",
            "Oops, I didn't catch that, can you please modify the question or change it",
            "Looks like I'm having trouble with that one. Can you ask me something else?",
            "Sorry about that, can you try a different question for me?",
            "Hm, I'm not quite sure about that. Can you ask me something else?",
        ]

        if (skip_track2) & (skip_track3):
            if qi.user_config.bot_response == "rule_based":
                # to generate the bot response using hardcoded custom responses
                bot_response = BotResponse(mode="rule_based").get_bot_error_message(track1_query)
            elif qi.user_config.bot_response == "model_based":
                # To use davinci 003 for bot response
                bot_response_ins = BotResponse(
                    user_config=qi.user_config,
                    model_config=qi.model_config,
                    conversation_history=history,
                    error_message=track1_query,
                    skip_model=False,
                    mode="model_based",
                )
                bot_response_ins.process_sql_error()
                bot_response = bot_response_ins.bot_response_output
            else:
                bot_response = random.choice(error_phrases)

            if bot_response is None:
                bot_response = random.choice(error_phrases)

            track1_status = "failure"
            track2_status = "skip"
            track3_status = "skip"
        elif (skip_track2) & (not skip_track3):
            bot_response = random.choice(processing_phrases)
            track1_status = "skip"
            track2_status = "skip"
            track3_status = "in_progress"
        elif (not skip_track2) & (not skip_track3):
            bot_response = random.choice(processing_phrases)
            track1_status = "success"
            track2_status = "in_progress"
            track3_status = "in_progress"
        else:
            bot_response = None
        return bot_response, track1_query, track1_status, track2_status, track3_status, qi

    def get_plot(self, state, qi, track2_status):
        """
        calls the model api to generate the plots to the given question and data

        Parameters
        ----------
        state : gr.State
            to keep track of user session
        qi : class_instance
            class instance specific to a session
        track2_status : str
            keeps track of track2 status ("skip" / "in_progress"). if "in_progress" it will run track2 else track2 will be skipped

        Returns
        -------
        Tuple
            if Track 2 runs succesfully, returns the plotly object in JSON format for displaying in UI
            gr.Plot (updated to be visible or not),
            if Track 2 fails or gets skipped, returns the table to be displayed in UI,
            gr.DataFrame (updated to be visible or not),
            track2_response["status"],
            qi,
            gr.State,
        """
        if track2_status == "skip":
            # if track2 gets skipped because of error in track1, dont show plot / table
            track2_response = {"status": "skip"}
            return (
                None,
                gr.Plot.update(visible=False),
                None,
                gr.DataFrame.update(visible=False),
                track2_response["status"],
                qi,
                state,
            )
        track2_response = qi.query_to_chart()
        plot_, track1_table = track2_response["output"]
        if (track2_response["status"] == "success") and (plot_ is not None):
            # when track 1 and 2 runs successfully show the plot in UI
            return (
                plot_.update_layout(width=1300, height=500),
                gr.Plot.update(visible=True),
                None,
                gr.DataFrame.update(visible=False),
                track2_response["status"],
                qi,
                state,
            )
        else:
            # when track 1 runs successfully but not track 2, show the table in UI
            return (
                None,
                gr.Plot.update(visible=False),
                track1_table.head(),
                gr.DataFrame.update(visible=True),
                track2_response["status"],
                qi,
                state,
            )

    def get_summary(self, track1_status, track2_status, track3_status, qi, state):
        """
        calls the model api to generate the insights to the given question and data

        Parameters
        ----------
        track1_status : str
            keeps track of track1 status ("failure" / "skip" / "success")
        track2_status : str
            keeps track of track2 status ("skip" / "in_progress"). if "in_progress" it will run track2 else track2 will be skipped
        track3_status : str
            keeps track of track3 status ("skip" / "in_progress"). if "in_progress" it will run track3 else track3 will be skipped
        qi : class_instance
            class instance specific to a session
        state : gr.State
            to keep track of user session

        Returns
        -------
        Tuple
            Returns the summary for displaying in UI. None if the table to insight generation gets failed,
            gr.Textbox (updated to be visible or not),
            qi,
            gr.State
        """
        if track3_status == "skip":
            # if track3 gets skipped because of error in track1, dont show the summary in UI
            self.save_to_knowledge_base(qi, track1_status, track2_status, track3_status)
            return None, gr.Textbox.update(visible=False), qi, state
        else:
            # runs track3 and show the summary output / error message if any
            track3_response = qi.table_to_insights()
            output_summary = track3_response["output"]
            track3_status = track3_response["status"]
            self.save_to_knowledge_base(qi, track1_status, track2_status, track3_status)
            return output_summary, gr.Textbox.update(visible=True), qi, state

    def save_to_knowledge_base(self, qi, track1_status, track2_status, track3_status):
        """
        updates the response to knowledge base.

        Parameters
        ----------
        qi : class_instance
            class instance specific to a session
        track1_status : str
            keeps track of track1 status ("failure" / "skip" / "success")
        track2_status : str
            keeps track of track2 status ("skip" / "in_progress"). if "in_progress" it will run track2 else track2 will be skipped
        track3_status : str
            keeps track of track3 status ("skip" / "in_progress"). if "in_progress" it will run track3 else track3 will be skipped
        """
        alltracks_status = [track1_status, track2_status, track3_status]
        feedback = None
        qi.update_knowledgebase(alltracks_status, feedback)

    def user(self, user_message, history, state):
        """
        appends followup question to the previous question seperated by " ; ". Will be cleared when the "click" button is triggered

        Parameters
        ----------
        user_message : str
            string that needs to be appended to existing question. If this is the first question, the `user_message` will be appended to an empty string `""`
        history : list
            contains the entire history of the chat conversation.
        state : gr.State
            to keep track of user session

        Returns
        -------
        history
            retuns updated chat bot history.
        """
        if history == []:
            state = str(int(time.time()))
        # to clear the user input text box after submitting
        empty_str = ""
        return empty_str, history + [[user_message, None]], state

    def bot(self, history, track1_status, track2_status, track3_status, qi_ins, state):
        """
        Triggers when the user question is submitted from UI.

        Parameters
        ----------
        history : list
            contains the current entire history of the chat conversation. example: [[user_msg1, bot_response1],[user_msg2, bot_response2],[user_msg3]]
        track1_status : str
            keeps track of track1 status ("failure" / "skip" / "success")
        track2_status : str
            keeps track of track2 status ("skip" / "in_progress"). if "in_progress" it will run track2 else track2 will be skipped
        track3_status : str
            keeps track of track3 status ("skip" / "in_progress"). if "in_progress" it will run track3 else track3 will be skipped
        qi_ins : class_instance
            class instance specific to a session
        state : gr.State
            to keep track of user session

        Returns
        -------
        Tuple[list, str, gr.Code, str, str, str, class, gr.State]
            It contains the following:\n
            chatbot history of conversation,
            track1_query,
            gr.Code (updates the code block's visibility),
            track1_status,
            track2_status,
            track3_status,
            qi_ins,
            state

        """
        # if len(history) > 1:
        # entire_conv = qi_ins.model_config.text_to_query.prompts.bot_conversations
        # for conv in history[:-1]:
        #     entire_conv += f"\n   user: {conv[0]}"
        #     entire_conv += f"\n   bot: {conv[1]}"
        # entire_conv += f"\n   user: {history[-1][0]}"

        # qi_ins.bot_history = history

        # else:
        # entire_conv = None

        # qi_ins.model_config.text_to_query.prompts.conv = entire_conv

        question_to_api = history
        # question_to_api = " ; ".join([q for [q, a] in history])
        (
            bot_message,
            track1_query,
            track1_status,
            track2_status,
            track3_status,
            qi_ins,
        ) = self.get_query(
            qi=qi_ins,
            input_text=question_to_api,
            track1_status=track1_status,
            track2_status=track2_status,
            track3_status=track3_status,
            history=history,
        )
        history[-1][1] = bot_message
        if bool(track1_query):
            # when track1 returns an SQL output or any skip/error message
            return (
                history,
                track1_query,
                gr.Code.update(visible=True),
                track1_status,
                track2_status,
                track3_status,
                qi_ins,
                state,
            )
        else:
            # when an error occurs and there is no output from track1 (ideally, shouldn't come to this else condition since, all errors are handled in track1)
            return (
                history,
                None,
                gr.Code.update(visible=False),
                track1_status,
                track2_status,
                track3_status,
                qi_ins,
                state,
            )

    def closing_message(self, history, track1_status, qi_ins):
        """
        prints appropriate bot response when the process ends

        Parameters
        ----------
        history : list
            contains the current entire history of the chat conversation. example: [[user_msg1, bot_response1],[user_msg2, bot_response2],[user_msg3]]

        Returns
        -------
        list
            contains the updated entire history of the chat conversation. example: [[user_msg1, bot_response1],[user_msg2, bot_response2],[user_msg3, bot_response3]]
        """
        bot_message = random.choice(
            [
                "Done! Here are the results.",
                "All set! Here's what I found.",
                "Got it! Here are the results for you.",
                "Finished! Here's what you were looking for.",
                "Here's what I came up with.",
                "Completed! Here's what I found out for you.",
                "Task accomplished! Here are the results you requested.",
                "Task fulfilled! Here's what I found.",
                "Done and done! Here are the results you asked for.",
            ]
        )
        if not (track1_status == "failure"):
            history[-1][1] = bot_message
        qi_ins.create_bot_training_data(history)
        return history

    def clear_all(self):
        """
        clears all the self variables so that the previous question's response wont affect the current question

        Returns
        -------
        None
        """
        # self.user_message = ""

        self.track1_query = None
        self.track1_table = None
        self.plot_ = None
        self.output_summary = None

        return None

    def _feedback_buttons_layout(self, feedback, block, visible=False):
        feedback[block] = {}
        with gr.Row():
            with gr.Column(scale=0.1, min_width=10):
                feedback[block]["like"] = gr.Button("👍", visible=False).style(size="sm")
            with gr.Column(scale=0.1, min_width=10):
                feedback[block]["dislike"] = gr.Button("👎", visible=False).style(size="sm")
        return feedback

    def run_chat_ui(self):
        """
        Generates the interface for taking user inputs, displaying results and the bot responses,
        then triggers the functions in a proper order to generate and display the responses
        """
        # TODO: one folder per orginal question. save the outputs of each follow up question into a sub folder.
        with gr.Blocks() as demo:
            with gr.Row().style(justify_content="flex-end"):
                with gr.Column(scale=1, min_width=1200):
                    gr.HTML("<br>")
                with gr.Column(scale=2, min_width=20):
                    gr.Markdown(f"[Static App >](https://nlpdemoprod.tigeranalytics.com/)")
            state = gr.State()
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="User input", placeholder="enter your query here...")
            gr.Examples(
                ["Get total shipped quantity by source location.", "sort it in decreasing order"],
                msg,
            )

            with gr.Row():
                # submit_bn = gr.Button("submit")
                clear = gr.Button("Clear")

            track_status = {}
            track_status["track1"] = gr.Textbox(value="", visible=False)
            track_status["track2"] = gr.Textbox(value="", visible=False)
            track_status["track3"] = gr.Textbox(value="", visible=False)
            # qi = QueryInsights(config_path=self.config_path, logging_level="DEBUG", api_key=self.api_key)
            qi_ins = gr.State([])

            feedback = {}
            plot_ = gr.Plot(visible=False)
            output_table = gr.DataFrame(
                label="Generated table (top 5 rows)",
                overflow_row_behaviour="show_ends",
                visible=False,
            )
            feedback = self._feedback_buttons_layout(feedback, block="plot_table")

            output_summary = gr.Textbox(label="Summary output", visible=False)
            feedback = self._feedback_buttons_layout(feedback, block="summary")

            display_query = gr.Code(label="SQL query used", visible=False)
            feedback = self._feedback_buttons_layout(feedback, block="query")

            gradio_output = (
                # take user input
                # submit_bn.click(
                msg.submit(
                    self.user,
                    [msg, chatbot, state],
                    [msg, chatbot, state],
                    queue=False,
                ).then(
                    lambda: QueryInsights(
                        user_config_path=self.user_config_path,
                        data_config_path=self.data_config_path,
                        model_config_path=self.model_config_path,
                        debug_config_path=self.debug_config_path,
                        logging_level="DEBUG",
                        api_key=self.api_key,
                    ),
                    None,
                    qi_ins,
                    queue=False,
                )
                # call the bot function to generate the SQL query and the bot response
                .then(
                    self.bot,
                    [
                        chatbot,
                        track_status["track1"],
                        track_status["track2"],
                        track_status["track3"],
                        qi_ins,
                        state,
                    ],
                    [
                        chatbot,
                        display_query,
                        display_query,
                        track_status["track1"],
                        track_status["track2"],
                        track_status["track3"],
                        qi_ins,
                        state,
                    ],
                    queue=False,
                )
                # generate the plotly chart/table and return figure/table
                .then(
                    self.get_plot,
                    [
                        state,
                        qi_ins,
                        track_status["track2"],
                    ],
                    [
                        plot_,
                        plot_,
                        output_table,
                        output_table,
                        track_status["track2"],
                        qi_ins,
                        state,
                    ],
                    queue=False,
                )
                # get the insights and save everything to knowledgebase
                .then(
                    self.get_summary,
                    [
                        track_status["track1"],
                        track_status["track2"],
                        track_status["track3"],
                        qi_ins,
                        state,
                    ],
                    [
                        output_summary,
                        output_summary,
                        qi_ins,
                        state,
                    ],
                    queue=False,
                )
                # for chat bot to say Done!!
                .then(
                    self.closing_message,
                    [
                        chatbot,
                        track_status["track1"],
                        qi_ins,
                    ],
                    chatbot,
                )
                # make the feedback buttons visible
                .then(
                    lambda: [
                        feedback[i][j].update(visible=True)
                        for i in list(feedback.keys())
                        for j in feedback[i]
                    ],
                    None,
                    [feedback[i][j] for i in list(feedback.keys()) for j in feedback[i]],
                    queue=False,
                )
            )

            user_feedback = Feedback()
            feedback["plot_table"]["like"].click(
                user_feedback.plot_or_table_liked, None, None, queue=False
            )
            feedback["plot_table"]["dislike"].click(
                user_feedback.plot_or_table_disliked, None, None, queue=False
            )
            feedback["summary"]["like"].click(user_feedback.summary_liked, None, None, queue=False)
            feedback["summary"]["dislike"].click(
                user_feedback.summary_disliked, None, None, queue=False
            )
            feedback["query"]["like"].click(user_feedback.query_liked, None, None, queue=False)
            feedback["query"]["dislike"].click(
                user_feedback.query_disliked, None, None, queue=False
            )

            clear.click(self.clear_all, None, chatbot, queue=False)
            clear.click(lambda: plot_.update(visible=False), None, plot_, queue=False)
            clear.click(
                lambda: output_table.update(visible=False), None, output_table, queue=False
            )
            clear.click(
                lambda: output_summary.update(visible=False), None, output_summary, queue=False
            )
            clear.click(
                lambda: display_query.update(visible=False), None, display_query, queue=False
            )
            clear.click(
                lambda: [
                    feedback[i][j].update(visible=False)
                    for i in list(feedback.keys())
                    for j in feedback[i]
                ],
                None,
                [feedback[i][j] for i in list(feedback.keys()) for j in feedback[i]],
                queue=False,
            )

        demo.launch()
        # demo.launch(server_name="0.0.0.0", server_port=3978)


if __name__ == "__main__":
    user_config_path = "ui/gradio/app_files/config/local/user_config_ui.yaml"
    data_config_path = "ui/gradio/app_files/config/local/data_config_ui.yaml"
    model_config_path = "ui/gradio/app_files/config/local/model_config_ui.yaml"
    debug_config_path = "configs/debug_code_config.yaml"

    chat_ins = ChatUI(
        user_config_path=user_config_path,
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        debug_config_path=debug_config_path,
    )

    chat_ins.run_chat_ui()
