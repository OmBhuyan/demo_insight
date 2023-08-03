import json
import logging
import os

import openai

from .utils import SensitiveContentError, TokenLimitError, timeout

MYLOGGERNAME = "QueryInsights"


class GPTModelCall:
    """
    A class that generates prompts and performs insights generation using OpenAI.

    Parameters
    ----------
    prompt_dict : dict
        A dictionary containing the prompt template and other static content.
    question : str
        A string representing the user's question.
    additional_context : str
        A string representing additional context to include in the prompt.
    connection_param_dict : dict
        A dictionary containing the parameters needed for the Azure's OpenAI API.
    dictionary : dict, optional
        A dictionary representing the data dictionary of the data specified in the config file. Defaults to None.
    suggestion : str, optional
        A string representing a suggestion for the user. Defaults to None.
    table : str, optional
        A string representing a table to include in the prompt. Defaults to None.
    sample_input: list [sample_question, sample_response], optional
        A list having a sample question and response for GPT's reference

    Attributes
    ----------
    api_key : str
        A string representing the OpenAI API key.
    dictionary : str
        A dictionary  representing the data dictionary of the data specified in the config file.
    prompt_dict : dict
        A dictionary containing the prompt template and other static content.
    question : str
        A string representing the user's question.
    additional_context : str
        A string representing additional context to include in the prompt.
    suggestion : str
        A string representing a suggestion for the user.
    table : str
        A string representing a table to include in the prompt.
    connection_param_dict : dict
        A dictionary containing the parameters needed for the Azure's OpenAI API.
    prompt : str
        A string representing the final prompt.

    Methods
    -------
    set_connection_params():
        Sets the connection parameters for the Azure's OpenAI API.

    generate_prompt():
        Generates the final prompt from the user inputs.

    model_response(model_param_dict):
        Performs insights generation using the OpenAI API and returns the output.

    extract_code(string_input, start, end):
        Extracts code from the ourput of OpenAI API response.
    """

    def __init__(
        self,
        prompt_dict,
        question,
        additional_context,
        connection_param_dict,
        dictionary=None,
        business_overview=None,
        suggestion=None,
        table=None,
        history=None,
        error_message=None,
        sample_input=None,
    ):
        self.logger = logging.getLogger(MYLOGGERNAME)

        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found in environment variables")
        openai.api_key = self.api_key

        self.dictionary = dictionary
        self.prompt_dict = prompt_dict
        self.question = question
        self.additional_context = additional_context
        self.business_overview = business_overview
        self.suggestion = suggestion
        self.table = table
        self.history = history
        self.error_message = error_message
        self.connection_param_dict = connection_param_dict

        self.sample_question = None
        self.sample_response = None
        if sample_input is not None:
            self.sample_question = sample_input[0]
            self.sample_response = sample_input[1]

        self.generate_prompt()
        self.set_connection_params()

    def set_connection_params(self):
        """
        Set the Azure's OpenAI API connection parameters.

        Parameters
        ----------
        self : GPTModelCall
            An instance of the GPTModelCall class.

        Raises
        ------
        KeyError
            If the 'platform' key is not found in the connection_param_dict dictionary.

        Returns
        -------
        None
        """
        try:
            if self.connection_param_dict["platform"] == "azure":
                openai.api_type = self.connection_param_dict["api_type"]
                openai.api_base = self.connection_param_dict["api_base"]
                openai.api_version = self.connection_param_dict["api_version"]
        except KeyError as e:
            raise KeyError(
                f"""An error occurred during the setting the connection parameters.
                                The error message is: {e}."""
            )

    def generate_prompt(self):
        """
        Generate the final prompt from the user inputs.

        Parameters
        ----------
        self : GPTModelCall
            An instance of the GPTModelCall class.

        Raises
        ------
        KeyError
            If the key is not found in the prompt_dict dictionary.

        Returns
        -------
        None
        """
        try:
            prompt = f"{self.prompt_dict['static_prompt']}"
            if self.sample_question is not None:
                prompt = prompt.replace("<sample_question>", f"{self.sample_question}")
            else:
                prompt = "\n".join(
                    [line for line in prompt.split("\n") if "<sample_question>" not in line]
                )
            if self.sample_response is not None:
                prompt = prompt.replace("<sample_response>", f"{self.sample_response}")
            else:
                prompt = "\n".join(
                    [line for line in prompt.split("\n") if "<sample_response>" not in line]
                )
            if self.question is not None:
                prompt = prompt.replace("<question>", f"{self.question}")
            else:
                prompt = "\n".join(
                    [line for line in prompt.split("\n") if "<question>" not in line]
                )
            if self.additional_context is not None:
                prompt = prompt + f"\n{self.prompt_dict['additional_context']}"
                prompt = prompt.replace("<additional_context>", f"{self.additional_context}")
            if self.suggestion is not None:
                prompt = prompt.replace("<suggestion>", f"{self.suggestion}")
            prompt = prompt + f"\n{self.prompt_dict['guidelines']}"

            data_dict_json_str = json.dumps(self.dictionary, indent=4)
            prompt = prompt.replace("<data_dictionary>", f"{data_dict_json_str}")

            if self.table is not None:
                prompt = prompt.replace("<table>", self.table)
            if self.history is not None:
                prompt = prompt.replace("<history>", self.history)
            if self.error_message is not None:
                prompt = prompt.replace("<error_message>", self.error_message)
            # if bool(self.prompt_dict['additional_context']):
            #    prompt = prompt + f"\n{self.prompt_dict['additional_context']}"
            if self.business_overview is not None:
                prompt = prompt + f"\n{self.prompt_dict['business_overview']}"
                prompt = prompt.replace("<business_overview>", f"{self.business_overview}")
            # if bool(self.prompt_dict["business_overview"]):
            #     if os.path.exists(self.prompt_dict["business_overview"]):
            #         with open(self.prompt_dict["business_overview"], "r") as file_:
            #             business_overview = file_.read()
            #         prompt = (
            #             prompt + "\n\n\nFoloowing is the business overview: \n" + business_overview
            #         )
            #     else:
            #         self.logger.warning("The business overview file does not exist.")

            self.prompt = prompt
            # self.logger.info(f"PROMPT:\n\n{prompt}\n")
        except KeyError as e:
            raise KeyError(
                f"An error occurred during the creating the prompt. The error message is: {e}."
            )

    @timeout()
    def model_response(
        self, model_param_dict: dict, debug_prompt: str = None, history: str = None
    ):
        """
        Generates a response from the provided model parameters and prompt.

        Parameters:
        -----------
        model_param_dict : dict
            A dictionary containing the parameters for the OpenAI Completion and Chat Completion API.

        debug_prompt : str
            Debug prompt

        history : str
            Previous response by GPT.

        Returns:
        --------
        tuple
            A tuple of output, finish reason and tokens from the OpenAI response.

        Raises:
        -------
        None
        """
        if model_param_dict["function"].lower() == "chatcompletion":
            current_message = [
                {
                    "role": "system",
                    "content": self.prompt_dict["system_role"],
                },
            ]
            bot_history = model_param_dict["history"]

            data = {"role": "user", "content": self.prompt}
            current_message.append(data)
            if bool(bot_history) and len(bot_history) > 1:
                if bot_history[0][1] is not None:
                    current_message.append({"role": "assistant", "content": bot_history[0][1]})
                for conv in bot_history[1:]:
                    current_message.append({"role": "user", "content": conv[0]})
                    if (conv[1] is not None) or (conv[1].strip() != ""):
                        current_message.append({"role": "assistant", "content": conv[1]})

            if not (debug_prompt is None and history is None):
                current_message = current_message + [
                    {"role": "assistant", "content": history},
                    {
                        "role": "user",
                        "content": debug_prompt,
                    },
                ]
                self.logger.debug(f"debug prompt:-\n\n{current_message}")
            self.current_message = current_message

            response = openai.ChatCompletion.create(
                engine=model_param_dict["engine"],
                messages=current_message,
                temperature=model_param_dict["temperature"],
                max_tokens=model_param_dict["max_tokens"],
                n=model_param_dict["n"],
                stop=model_param_dict["stop"],
            )

            output = response["choices"][0]["message"]["content"]

            finish_reason = response["choices"][0]["finish_reason"]
            tokens = response["usage"]

            if finish_reason == "length":
                raise TokenLimitError(f"Token limit exceeded. {tokens}")
            elif finish_reason == "content":
                raise SensitiveContentError(
                    "Question is flagged as sensitive content by the OpenAI's model. Please change the language or the data."
                )

            return output, finish_reason, tokens

        elif model_param_dict["function"].lower() == "completion":
            if debug_prompt is not None and history is not None:
                self.prompt = self.prompt + "\n\n"
                self.prompt += f"GPT response:\n\n{history}\n\n"
                self.prompt += f"New Question: {debug_prompt}"

            response = openai.Completion.create(
                engine=model_param_dict["engine"],
                prompt=self.prompt,
                temperature=model_param_dict["temperature"],
                max_tokens=model_param_dict["max_tokens"],
                n=model_param_dict["n"],
                stop=model_param_dict["stop"],
            )
            output = response.choices[0].text
            finish_reason = response["choices"][0]["finish_reason"]
            tokens = response["usage"]
            return output, finish_reason, tokens
        else:
            raise ValueError(
                f"Invalid function {model_param_dict['function']} is passed in the config. Acceptable values are 'Completion' and 'ChatCompletion'"
            )
