import ast
import datetime
import logging
import os
import posixpath as pp
import re
import subprocess
import sys

import fsspec
import numpy as np
import pandas as pd

from .utils import generate_env_dict

# Default logger
MYLOGGERNAME = "QueryInsights"
logger = logging.getLogger(MYLOGGERNAME)


def append_user_query_track3(other_questions: str, user_query: str) -> str:
    """Appends User questions to GPT generated questions

    Parameters
    ----------
    other_questions : str
        GPT generated questions
    user_query : str
        Business user question

    Returns
    -------
    str
        Concatenated questions.
    """
    # Here, we assume we asked for two other questions only.
    all_questions = other_questions + "\n\nQuestion 3: " + user_query
    return all_questions


def _preprocess_track3_code(
    full_code_str: str,
    input_file_path: str,
    tmp_dir_path: str,
) -> str:
    """Prepend pandas import and read. Even if it's already present, there is no problem if we have same import twice.
    This function also handles replaces show with savefig method.

    Parameters
    ----------
    full_code_str : str
        GPT generated code
    input_file_path : str
        Track 1 output table that will be used in the code.
    tmp_dir_path : str
        Relative file path where we are saving the code temporarily.
    Returns
    -------
    str
        Modified code string which contains the pandas import and read.
    """

    full_code_block = "import pandas as pd\n"
    full_code_block += "import fsspec\n"
    full_code_block += "import os\n"
    full_code_block += "import posixpath as pp\n"
    full_code_block += "\n"

    full_code_block += """
prefix_url = os.getenv("prefix_url")
DefaultEndpointsProtocol = os.getenv("DefaultEndpointsProtocol")
AccountName = os.getenv("AccountName")
AccountKey = os.getenv("AccountKey")
EndpointSuffix = os.getenv("EndpointSuffix")

if any(value is None for value in (prefix_url, DefaultEndpointsProtocol, AccountName, AccountKey, EndpointSuffix)):
    prefix_url = ""
    storage_options = None
    fs = fsspec.filesystem("file")
else:
    fs_connection_string = f"DefaultEndpointsProtocol={DefaultEndpointsProtocol};AccountName={AccountName};AccountKey={AccountKey};EndpointSuffix={EndpointSuffix}"
    storage_options = {
        "connection_string": fs_connection_string,
        "account_key": AccountKey,
    }
    fs, _, _ = fsspec.core.get_fs_token_paths(prefix_url, storage_options=storage_options)"""

    full_code_block += "\n"
    # Since prefix_url is an empty string and storage_options will be None in local, the below block of code should work in local
    full_code_block += (
        f"""df = pd.read_csv(prefix_url+r"{input_file_path}", storage_options=storage_options)"""
    )
    full_code_block += "\n"
    full_code_block += full_code_str
    full_code_block += "\n"

    fig_show_str_count = full_code_block.count(".show(")
    if fig_show_str_count > 0:
        repl_strings = [
            pp.join(tmp_dir_path, f""".savefig("additional_charts_{i+1}.png")""")
            for i in range(fig_show_str_count)
        ]
        full_code_block = re.sub(
            pattern="\.show\(.+", repl=lambda repl: repl_strings.pop(0), string=full_code_block
        )

    return full_code_block


def _run_full_code(
    code_path: str,
    fs=None,
    fs_connection_dict=None,
    fs_key=None,
) -> bool:
    """Runs GPT generated code that contains 3 blocks of code as a whole.

    Parameters
    ----------
    code_path: str
        Python code path that will be run for track 3
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``
    fs_connection_dict : dict, optional
        Dictionary containing configuration settings to connect to the cloud, by default None
    fs_key : str, optional
        Account key to make connection to the specified platform (in fs_connection_dict). If platform is not None, it will look for the path in the data_config and read the key from there. Can be left as None for using local File storage (Windows, Linux) (when platform in None), by default None

    Returns
    -------
    bool
        Returns True if code execution is successful else, it returns false.
    """
    fs = fs or fsspec.filesystem("file")
    # subprocess_args = ["python", code_path]

    # sys.executable is used as sometimes when setting env variables
    # python path is not being recognized
    subprocess_args = [sys.executable, "-c"]
    try:
        with fs.open(code_path, "r") as f:
            code = f.read()
        subprocess_args.append(code)

        # Generate env dictionary from cloud storage parameters to be passed to subprocess as env variables
        # env will be None in case no cloud storage parameters are specified
        env = generate_env_dict(
            cloud_storage_dict=fs_connection_dict,
            account_key=os.getenv("BLOB_ACCOUNT_KEY"),
        )

        # subprocess output is of type - https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess
        subprocess_result = subprocess.run(
            args=subprocess_args,
            text=True,
            capture_output=True,
            env=env,  # Sets environment variables to run the code
        )
        logger.debug(f"Insights full code run return status = {subprocess_result.returncode}")
        subprocess_result.check_returncode()
        print(subprocess_result.stdout)
        # exec(full_code_str)
        return True
    except subprocess.CalledProcessError as e:
        print(
            "Error while running the entire code:-\n",
            file=sys.stderr,
        )
        print("Error description:\n", e.stderr, file=sys.stderr)
        return False
    except Exception as e:
        print(
            f"Error while running the full code {e}",
            file=sys.stderr,
        )
        print("Error description:\n", e, file=sys.stderr)
        return False


def _get_import_statements(full_code_str: str) -> str:
    """From given GPT generated code string, extract all the lines which does import and pandas read_csv

    Parameters
    ----------
    full_code_str : str
        GPT generated code string

    Returns
    -------
    str
        import statements with read_csv
    """
    pat = r"(^import \w+)|(^from\s\w+(?:\.\w+)*\simport\s\w+)"
    import_block = ""
    # Capture all imports
    for line in full_code_str.splitlines():
        if re.match(pattern=pat, string=line, flags=re.IGNORECASE):
            import_block += line
            import_block += "\n"
        # Capture read csv
        elif "pd.read_csv" in line:
            import_block += line
            import_block += "\n"

    return import_block


def run_insights_code(
    full_code_str: str,
    input_file_path: str,
    tmp_dir_path: str,
    track3_path: str,
    fs=None,
    fs_connection_dict: dict = None,
    fs_key: str = None,
) -> None:
    """For a given GPT generated Track 3 code, apply below processing steps to run it to get output of the code.

    Processing steps:
    1. Normalize the path as per the OS.
    2. Prepend pandas import and read.
    3. Convert .show to .savefig
    4. Extract import block statements.
    5. Run entire code to see if its runs successfully.
    6. IF entire code didn't run due to an issue in one of the code, run in block as we asked GPT to construct code for different questions.
    7. Save code for running via subprocess.

    Parameters
    ----------
    full_code_str : str
        GPT generated code
    input_file_path : str
        Relative file path of Track 1 output
    tmp_dir_path : str
        Relative file path where we are saving the code temporarily.
    track3_path : str
        Relative file path where we are saving track 3 results.
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``
    fs_connection_dict : dict, optional
        Dictionary containing configuration settings to connect to the cloud, by default None
    fs_key : str, optional
        Account key to make connection to the specified platform (in fs_connection_dict). If platform is not None, it will look for the path in the data_config and read the key from there. Can be left as None for using local File storage (Windows, Linux) (when platform in None), by default None

    """
    fs = fs or fsspec.filesystem("file")
    code_path = pp.join(track3_path, "track3_code.py")
    # First prepare the code
    full_code_str = _preprocess_track3_code(
        full_code_str=full_code_str,
        input_file_path=input_file_path,
        tmp_dir_path=tmp_dir_path,
    )

    # Write the code to python file as per folder structure.
    logger.info("Saving Track 3 code.")

    with fs.open(code_path, "w") as f_code:
        f_code.writelines(full_code_str)

    # First run the entire code
    entire_code_status = _run_full_code(
        code_path=code_path, fs=fs, fs_connection_dict=fs_connection_dict, fs_key=fs_key
    )

    if entire_code_status:
        logger.info("Entire code for Track 3 ran successfully.")
        return
    else:
        import_block = _get_import_statements(full_code_str=full_code_str)
        logger.debug("Import statements for Track 3 code extracted successfully.")
        # We have asked GPT to comment '#----' at the end of every section.
        # We also noticed that we get '# ----' at the end of every section as another type
        # If neither of these patterns are present, r"^# ?Question" or r"^# ?\d{1}"
        # If still not found, we cannot split it and as we run the entire code already return.

        patterns = [r"#----", r"# ----", r"^# ?Question", r"^# ?\d{1}"]
        split_point_linenum = []
        code_at_line_level = []

        # Identify linenum to break the code into individual blocks of code as per the Qn.
        for linenum, line in enumerate(full_code_str.splitlines()):
            code_at_line_level.append(line)
            for pat in patterns:
                if re.match(pattern=pat, string=line, flags=re.IGNORECASE):
                    split_point_linenum.append(linenum)
                    break

        # Break the code into blocks.
        starting_linenum = 0
        code_blocks = []
        for linenum in split_point_linenum:
            block = code_at_line_level[starting_linenum:linenum]
            block = "\n".join(block)  # Convert list to string.

            if import_block not in block:
                block = import_block + "\n\n" + block

            code_blocks.append(block)
            starting_linenum = linenum + 1

        # Execute the code blocks.
        for i, blk in enumerate(code_blocks):
            logger.debug(f"running code block {i}")
            current_ts = (
                str(datetime.datetime.now(datetime.timezone.utc))
                .replace("+00:00", "Z")
                .replace(":", "_")
                .replace(" ", "_")
                .replace(".", "_")
            )
            code_block_fname = pp.join(tmp_dir_path, f"code_blk{i}_{current_ts}.py")

            with fs.open(code_block_fname, "w") as f_code:
                f_code.writelines(blk)

            # sys.executable is used as sometimes when setting env variables
            # python path is not being recognized
            subprocess_args = [sys.executable, "-c", blk]

            try:
                # subprocess output is of type - https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess

                # Generate env dictionary from cloud storage parameters to be passed to subprocess as env variables
                # env will be None in case no cloud storage parameters are specified
                env = generate_env_dict(
                    cloud_storage_dict=fs_connection_dict,
                    account_key=fs_key,
                )

                subprocess_result = subprocess.run(
                    args=subprocess_args,
                    text=True,
                    capture_output=True,
                    env=env,  # Sets environment variables to run the code
                )
                logger.debug(
                    f"Insights code block run return status = {subprocess_result.returncode}"
                )
                subprocess_result.check_returncode()
                print(subprocess_result.stdout)
            except subprocess.CalledProcessError as e:
                print(
                    f"Error while running the code block {i}, going to the next block",
                    file=sys.stderr,
                )
                print("Error description:\n", e.stderr, file=sys.stderr)
            except Exception as e:
                print(
                    f"Error while running the code block {i}, going to the next block",
                    file=sys.stderr,
                )
                print("Error description:\n", e, file=sys.stderr)

            if fs.exists(code_block_fname):
                # Remove the tmp code files created.
                fs.rm(code_block_fname)  # Book keeping.

    return


def _update_percent_sign(query, update):
    """
    Adds or Removes the % sign before and after the word present after the word LIKE in the query.

    Parameters
    ----------
    query : str
        The SQL query.
    update : str
        Can have values 'add' or 'remove'

    Returns
    -------
    str
        The SQL query with the % sign added before and after the word present after the word LIKE in the query.
    """
    if "LIKE" not in query:
        return query
    else:
        # Define a regular expression pattern for the search term after LIKE
        pattern = re.compile(r"LIKE\s+'%?(.*?)%?'")

        if update == "add":
            query = pattern.sub(lambda x: "LIKE '%" + x.group(1) + "%'", query)
        elif update == "remove":
            query = pattern.sub(lambda x: "LIKE '" + x.group(1) + "'", query)
        else:
            return query

        return query


def extract_code(string_input: str, start: list, end: str, extract: str):
    """
    Extracts the code from the OpenAI API response using the start and end delimiters.
    If there are more than one start substrings and one element is a substring of other, please order it in a way the first element is a subset of other element.
    For example -
        If the start elements are <start> and <begin>, they can be given in any order.
        If start elements are "```python" and "```", one is a substring of the other. "```" should be specified before "```python".

    Parameters
    ----------
    string_input : str
        The input string from which to extract the code.
    start : list
        The starting delimiter of the code.
    end : str
        The ending delimiter of the code.
    extract: str
        possible values - first / all
        Extract either the first instance or all instances from the string_input.

    Returns
    -------
    str/list
        First extracted code from the OpenAI API response if extract is passed as 'first'.
        List of all extracted codes from the OpenAI API response if extract is passed as 'all'.
    """
    codes = []
    if len(start) > 1:
        # If there are more than one starting patterns, replace all other patterns with the first pattern.
        for i in range(1, len(start)):
            string_input = string_input.replace(start[i], start[0])

    pattern = f"{re.escape(start[0])}\n?([\s\S]*?){re.escape(end)}"
    matches = re.finditer(pattern, string_input)

    for match in matches:
        code = match.group(1).strip()
        if code:
            codes.append(code)

    if len(codes) > 0:
        if extract == "first":
            return codes[0]
        elif extract == "all":
            return codes
        else:
            return None
    else:
        return None


def round_float_numbers_in_dataframe(df):
    """
    Rounds the float numbers in the dataframe to 2 decimal places.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose float numbers are to be rounded.

    Returns
    -------
    pd.DataFrame
        The dataframe with the float numbers rounded to 2 decimal places.

    """
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].round(2)

    return df


def clean_chart_code(
    full_code_str: str,
    input_file_path: str,
    chart_save_path: str,
    import_statements: str,
) -> str:
    """For the GPT generated chart code, we do below post processing steps before we run it using subprocess.

    Steps::

        1. Prepend pandas import and read_csv for reading the track 1 output table.
        2. remove fig.show so that we don't print the chart to stdout.
        3. save fig temporarily so that we can read it back.

    Parameters
    ----------
    full_code_str : str
        GPT generated chart code.
    input_file_path : str
        Track 1 output table path.
    tmp_chart_path : str
        Place where we add the save chart.
    import_statements : str
        Import statements to be added to the code.

    Returns
    -------
    str
        post processed and cleaned chart code.
    """
    # Correcting slashes in paths to a standard format. Regex considers backward slash (windows) as end character and throws error
    input_file_path = input_file_path.replace(os.path.sep, pp.sep)
    chart_save_path = chart_save_path.replace(os.path.sep, pp.sep)
    # Prepend pandas import and read. Even if it's already present, there is no problem if we have same import twice.
    # Patterns to be removed in import statements and GPT response
    pattern = r".*pd\.read_csv\(.*"

    full_code_block = "import pandas as pd\n"
    full_code_block += "import fsspec\n"
    full_code_block += "import yaml\n"
    full_code_block += "import plotly\n"
    full_code_block += "import os\n"
    full_code_block += "import posixpath as pp\n"
    full_code_block += "\n"
    full_code_block += "\n"

    full_code_block += """
prefix_url = os.getenv("prefix_url")
DefaultEndpointsProtocol = os.getenv("DefaultEndpointsProtocol")
AccountName = os.getenv("AccountName")
AccountKey = os.getenv("AccountKey")
EndpointSuffix = os.getenv("EndpointSuffix")

if any(value is None for value in (prefix_url, DefaultEndpointsProtocol, AccountName, AccountKey, EndpointSuffix)):
    prefix_url = ""
    storage_options = None
    fs = fsspec.filesystem("file")
else:
    fs_connection_string = f"DefaultEndpointsProtocol={DefaultEndpointsProtocol};AccountName={AccountName};AccountKey={AccountKey};EndpointSuffix={EndpointSuffix}"
    storage_options = {
        "connection_string": fs_connection_string,
        "account_key": AccountKey,
    }
    fs, _, _ = fsspec.core.get_fs_token_paths(prefix_url, storage_options=storage_options)"""

    full_code_block += "\n"
    # Add import statements to the code.
    if import_statements is not None:
        # Removing any step which contains pd.read_csv(.
        import_statements = re.sub(pattern, "", import_statements, flags=re.MULTILINE)
        full_code_block += import_statements
        full_code_block += "\n"

    # Step to read the input CSV file (Result from track 1 initial SQL query)
    full_code_block += (
        f"""df = pd.read_csv(prefix_url+r"{input_file_path}", storage_options=storage_options)"""
    )
    full_code_block += "\n"
    # Removing any step which contains pd.read_csv( from GPT response.
    full_code_str = re.sub(pattern, "", full_code_str, flags=re.MULTILINE)
    full_code_block += full_code_str

    # Regular expression to search for variable declaration with go.Figure()
    figure_declaration_match = re.search(r"(\w+)\s*=\s*go\.Figure\(.*\)", full_code_block)

    # Regular expression to search for fig.update_layout() and fetch the variable name
    update_layout_variable_match = re.search(r"(\w+)\s*\.update_layout\(", full_code_block)

    # Initialize plotly_figure as None
    plotly_figure = None

    # Check if go.Figure() is declared in the code and fetch the variable name
    if figure_declaration_match:
        plotly_figure = figure_declaration_match.group(1)

    # If go.Figure() is not declared, check for fig.update_layout() and fetch the variable name
    elif update_layout_variable_match:
        plotly_figure = update_layout_variable_match.group(1)

    logger.debug(f"fig_variable : {plotly_figure}")

    # Remove show()
    if "import plotly.io" in full_code_block:
        if re.search(pattern=r"\w+\.show\((\w+)\)", string=full_code_block, flags=re.MULTILINE):
            plotly_figure = re.search(pattern="\w+\.show\((\w+)\)", string=full_code_block).group(
                1
            )
            full_code_block = re.sub(
                pattern=r"\w+\.show\((\w+)\).*", repl="\n", string=full_code_block
            )
            save_fig_line = rf"""
with fs.open("{chart_save_path}", mode="w") as fp:"""
            save_fig_line += (
                rf"""
    plotly.io.write_json({plotly_figure}, file=fp, pretty=True)
            """
                + "\n"
            )
            full_code_block += save_fig_line
            full_code_block += "\n"
        else:
            # In case a plotly graph is created but show is not present in the generated code.
            save_fig_line = rf"""
with fs.open("{chart_save_path}", mode="w") as fp:"""
            save_fig_line += (
                rf"""
    plotly.io.write_json({plotly_figure}, file=fp, pretty=True)
            """
                + "\n"
            )
            full_code_block += save_fig_line
            full_code_block += "\n"
    else:
        search_pattr = re.search(
            pattern=r"^(\s*)(.*?)\.show\(\)", string=full_code_block, flags=re.MULTILINE
        )
        if search_pattr:
            leading_whitespace = search_pattr[1]
            plotly_figure = search_pattr[2]
            save_fig_line = rf"""
with fs.open("{chart_save_path}", mode="w") as fp:"""
            save_fig_line += (
                rf"""
    plotly.io.write_json({plotly_figure}, file=fp, pretty=True)
            """
                + "\n"
            )
            indented_replacement = re.sub(r"(?m)^", leading_whitespace, save_fig_line)
            # Convert to raw string so that it doesnt escape anything inside it
            if os.name == "nt":
                indented_replacement = indented_replacement.encode("unicode-escape").decode()

            full_code_block = "import plotly.io\n" + full_code_block

            full_code_block = re.sub(
                pattern=r"^.*?\.show\(\)",
                repl=indented_replacement,
                string=full_code_block,
                flags=re.MULTILINE,
            )
            full_code_block += "\n"
        else:
            # In case a plotly graph is not created and show is not present in the generated code.
            full_code_block = "import plotly.io\n" + full_code_block
            # In case show is not present in the generated code
            save_fig_line = rf"""
with fs.open("{chart_save_path}", mode="w") as fp:"""
            save_fig_line += (
                rf"""
    plotly.io.write_json({plotly_figure}, file=fp, pretty=True)
            """
                + "\n"
            )
            full_code_block += save_fig_line
            full_code_block += "\n"

    return full_code_block


def add_exception_to_code(
    full_code_str: str, include_pattern: tuple, exclude_pattern: tuple
) -> str:
    """
    For the GPT generated chart code, we do below post processing steps before we run it using subprocess.

    Steps::
        1. Parse the GPT generated code using AST to extract the lines to a list.
        2. Save the comments with the line numbers and use them when looping
        3. Loop through each line and do the below steps -
            - If the lines start with include_patterns provided, insert a try-except condition to the code.
            But this can be skipped if it starts with exclude patterns.
            - Also remove any line which has pd.DataFrame(.
            Since we are adding code to read our own data, we can remove the sample data creation steps from GPT response.

    Parameters
    ----------
    full_code_str : str
        GPT generated chart code.
    include_pattern : tuple
        Tuple containing the starting patterns for which try-except should be included.
    exclude_pattern : str
        Tuple containing the starting patterns for which try-except need not be added.

    Returns
    -------
    str
        code with try-except conditions added wherever necessary.
    """
    # Adding try-except to lines wherever "fig." is encountered.
    # Parse the code and get each line to a separate element in a list using AST.

    # Split code into lines(This split is just to identify the comments line by line and not to be used for parsing the code as this will not take care of code statements that are broken into more than one lines)
    lines = full_code_str.splitlines()

    # As we are using abstract syntax tree(AST), it doesn't look for the comments.
    # We are storing the comments returned by GPT and their respective line numbers in a dictionary. They'll be added back to the processed code in the following code blocks
    comment_lines = {}
    for i, line in enumerate(lines, 1):
        # Check if the line starts with '#'
        if line.lstrip().startswith("#"):
            # Extract the comment text by removing leading '#' and stripping whitespace
            comment = line.lstrip("#").strip()
            # Store the comment with its line number in the dictionary
            comment_lines[i] = comment

    tree = ast.parse(full_code_str)
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            line_start = node.lineno
            line_end = node.end_lineno
            # Retrieve comments for a specific line number. If no comments were found, then it will return the default value i.e here an empty string.
            lines.append(
                [
                    comment_lines.get(line_start - 1, ""),
                    (full_code_str.splitlines()[line_start - 1 : line_end]),
                ]
            )

    if exclude_pattern is None:
        exclude_pattern = ()
    # Loop through each line and check if the line starts with '<include pattern>' and not 'exclude pattern'. If so, try-except condition is added to the line.
    # Also if the line has 'pd.DataFrame(, that line is removed.
    modified_lines = []

    # Here, we are checking whether a comment is present in the dictionary for each line number.
    # if present, then we fetch the comment. if not, then the default value.
    for comment, line in lines:
        if not line[0].startswith("    ") and "pd.DataFrame(" not in line[0]:
            if line[0].startswith((include_pattern)) and not line[0].startswith((exclude_pattern)):
                modified_lines.append("try:")
                if comment:
                    # Split the comment into individual lines and add them separately
                    for c in comment.splitlines():
                        modified_lines.append("    # " + c.strip())
                for li in line:
                    modified_lines.append("    " + li)
                modified_lines.append("except Exception as e:")
                error_message = "Error in the line starting with - " + line[0].replace('"', "'")
                modified_lines.append("    " + f"""print("{error_message}", e)""")
            else:
                if comment:
                    # Split the comment into individual lines and add them separately
                    for c in comment.splitlines():
                        modified_lines.append("# " + c.strip())
                modified_lines.append("\n".join(line))

    full_code_block = "\n".join(modified_lines)

    return full_code_block


# write a python function to detect alter table statements and drop table statements in the sql query
def _detect_alter_drop_table(query):
    """
    Detects the ALTER TABLE and DROP TABLE statements in the SQL query.

    Parameters
    ----------
    query : str
        The SQL query.

    Returns
    -------
    bool
        True if the ALTER TABLE or DROP TABLE or CREATE TABLEstatements are present in the SQL query, else False.

    """
    # Define a regular expression pattern for the ALTER TABLE and DROP TABLE statements
    pattern = re.compile(r"(ALTER TABLE|DROP TABLE|CREATE TABLE|INSERT INTO)", re.IGNORECASE)

    # Search for the ALTER TABLE and DROP TABLE statements in the SQL query
    if pattern.search(query):
        return True
    else:
        return False


def _extract_queries(query):
    """
    Detects if there are multiple SQL queries present and
    checks if they are starting with Select or With.

    Parameters
    ----------
    query : str
        SQL Query(s)

    Returns
    -------
    list of tuple
        First parameter - Each SQL Query
        Second parameter - Boolean to denote whether it's a SELECT/CTE statement or not
    """
    # Split the string into multiple SQL queries
    queries = query.split(";")

    # Check if the queries are SELECT or WITH (CTE) statements
    query_list = []
    for q in queries:
        # Remove the comments if any. Comments usually start with -- or # or between this /* */.
        sql_query = re.sub(r"--.*$", "", q, flags=re.MULTILINE).strip()
        sql_query = re.sub(r"#.*$", "", sql_query, flags=re.MULTILINE).strip()
        sql_query = re.sub(r"/\*.*?\*/", "", sql_query, flags=re.DOTALL).strip()
        if sql_query:
            if sql_query.strip().lower().startswith(
                "select"
            ) or sql_query.strip().lower().startswith("with"):
                select_flag = True
            else:
                select_flag = False

            query_list = query_list + [(sql_query + ";", select_flag)]

    return query_list


def _uniquecategory_check(input_df):
    """
    Checks if the input df has only one categorical column with all unique values.

    Parameters
    ----------
    input_df : pandas dataframe
        Dataframe which is to be checked

    Returns
    -------
    unique_flag : bool
        Denotes whether the input df has all unique categorical values
    """
    # Initializing the flag to False first
    unique_flag = False
    # Check if the dataframe has only one column
    if input_df.shape[1] == 1:
        col_name = input_df.columns[0]
        try:
            # Check if the column can be converted to numeric.
            # To numeric is used, as numeric values can be considered as object
            pd.to_numeric(input_df[col_name], errors="raise")
        except ValueError:
            # Check if all values are unique and change the unique Flag to True.
            if input_df.shape[0] == input_df[col_name].nunique():
                unique_flag = True

    return unique_flag


def _string_to_dict(input_string: str) -> list:
    """
    Convert a string with dictionaries to a list of dictionaries

    Parameters
    ----------
    input_string : str
        Input string with dictionary

    Returns
    -------
    dict_list
        Converted list of dictionaries
    """
    try:
        dict_list = ast.literal_eval(input_string)
        # Removed below custom logic as its not foolproof to several variations of the string returned by GPT such as " instead of '
        # dict_strings = re.findall(r"{.*?}", input_string)

        # # Convert each dictionary string to a dictionary and append to a list
        # dict_list = []
        # for dict_string in dict_strings:
        #     # Regex is not capturing None, thus we need to check for it and replace it with blank.
        #     if dict_string.lower() in ["none", "nan"]:
        #         dict_string = re.sub(
        #             pattern=r":(\s?none|\s?nan)", repl=": ' '", flags=re.IGNORECASE, string=dict_string
        #         )

        #     dict_items = re.findall(r"'([^']*)'\s*:\s*'([^']*)'", dict_string)
        #     dict_list.append({k: v for k, v in dict_items})
        return dict_list
    except Exception as e:
        logger.warning(f"Data dictionary is not generated. Error msg: {e}")
        return []


def _complete_data_dict(output_table, raw_dict: dict, result_dict: list):
    """
    complete data dict using the raw dat dictionary, when data dictionary from track1 gpt response does not include all the column present in the generated table

    Parameters
    ----------
    output_table : pd.DataFrame
        track1 table output
    raw_dict : dict
        input data dictiionary by the user. format:{table1: {table_name:"name",
                                                            columns:[{"name":"col1", description:"..."},{"name":"col2", description:"..."}]}
                                                    table2: {table_name:"name",
                                                            columns:[{"name":"col1", description:"..."},{"name":"col2", description:"..."}]}
                                                    ...}
    result_dict : list
        track1 data dictionay output. format: [{'column_name':"col1", 'description':"...", 'unit_of_measurement':"...", 'id':<Yes/No>},
                                               {'column_name':"col2", 'description':"...", 'unit_of_measurement':"...", 'id':<Yes/No>},
                                               ...]

    Returns
    -------
    _type_
        _description_
    """
    final_data_dict = []
    raw_dict_list = []
    # to list the column dict {name, description} of all columns from all the tables present in the DB
    for _, dict_ in raw_dict.items():
        raw_dict_list += dict_["columns"]

    # get the list of column names in the result dict and raw dict
    new_cols = np.array([i["column_name"] for i in result_dict])
    raw_cols = np.array([i["name"] for i in raw_dict_list])
    result_dict = np.array(result_dict)
    raw_dict_list = np.array(raw_dict_list)

    # loop over all the columns present in output table
    for col in output_table.columns:
        # check if column name is present in result dict use that column dictionary in the final dict
        if col in new_cols:
            col_dict = result_dict[np.argwhere(col == new_cols).ravel()][0]
            final_data_dict.append(col_dict)
        # else if column name is present in raw dict use that column dictionary in the final dict
        elif col in raw_cols:
            col_dict = raw_dict_list[np.argwhere(col == raw_cols).ravel()][0]
            # renaming the dictionary key from "name" to "column_name"
            col_dict["column_name"] = col_dict.pop("name")
            final_data_dict.append(col_dict)
        # if it is not present in either dictionaries, then just add the column name and raise a warning
        else:
            final_data_dict.append({"column_name": col})
            logger.warning(f"Description not found for the column: {col}")
    return final_data_dict
