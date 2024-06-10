import os
import boto3
import re
from rich.console import Console
from rich.panel import Panel
from datetime import datetime
import json
from tavily import TavilyClient
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

orchestrator_modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
#orchestrator_modelId = "anthropic.claude-3-opus-20240229-v1:0"
agent_modelId = "anthropic.claude-3-haiku-20240307-v1:0"

# get the Tavily API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#create a boto3 bedrock client
client = boto3.client(service_name = 'bedrock-runtime', region_name = region,)



def calculate_subagent_cost(model, input_tokens, output_tokens):
    """
    Calculate the cost of using a subagent model based on the pricing information.

    Args:
        model (str): The name of the subagent model.
        input_tokens (int): The number of input tokens.
        output_tokens (int): The number of output tokens.

    Returns:
        float: The total cost of using the subagent model.

    Raises:
        KeyError: If the specified model is not found in the pricing information.

    """
    # Pricing information per model
    pricing = {
        "anthropic.claude-3-opus-20240229-v1:0": {"input_cost_per_mtok": 15.00, "output_cost_per_mtok": 75.00},
        "anthropic.claude-3-haiku-20240307-v1:0": {"input_cost_per_mtok": 0.25, "output_cost_per_mtok": 1.25},
    }

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input_cost_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output_cost_per_mtok"]
    total_cost = input_cost + output_cost

    return total_cost

# Initialize the Rich Console
console = Console()


def break_down_objective(objective, file_content=None, previous_results=None, use_search=False):
    
    """
    Orchestrates the execution of a sub-task based on the given objective, file content, and previous results.

    Args:
        objective (str): The objective for the sub-task.
        file_content (str, optional): The content of the file related to the objective. Defaults to None.
        previous_results (list, optional): The previous sub-task results. Defaults to None.
        use_search (bool, optional): Flag indicating whether to generate a search query. Defaults to False.

    Returns:
        tuple: A tuple containing the response text, file content, and search query (if generated).
    """

    console.print(f"\n[bold]Calling Orchestrator for your objective[/bold]")
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    if file_content:
        console.print(Panel(f"File content:\n{file_content}", title="[bold blue]File Content[/bold blue]", title_align="left", border_style="blue"))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"text": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}" + ('\\nFile content:\\n' + file_content if file_content else '') + f"\n\nPrevious sub-task results:\n{previous_results_text}"}
            ]
        }
    ]
    if use_search:
        messages[0]["content"].append({"text": "Please also generate a JSON object containing a single 'search_query' key, which represents a question that, when asked online, would yield important information for solving the subtask. The question should be specific and targeted to elicit the most relevant and helpful resources. Format your JSON like this, with no additional text before or after:\n{\"search_query\": \"<question>\"}\n"})

    try:
        orchestrator_params = {
        "modelId": orchestrator_modelId,
        "inferenceConfig": {"maxTokens": 4096 },
        "messages": messages,
        }

        orchestrator_response = client.converse(**orchestrator_params)

        response_text = orchestrator_response['output']['message']['content'][0]['text']
        console.print(f"Input Tokens: {orchestrator_response['usage']['inputTokens']}, Output Tokens: {orchestrator_response['usage']['outputTokens']}")
        total_cost = calculate_subagent_cost(orchestrator_modelId, orchestrator_response['usage']['inputTokens'], orchestrator_response['usage']['outputTokens'])
        console.print(f"Opus Orchestrator Cost: ${total_cost:.2f}")

        search_query = None
        if use_search:
            # Extract the JSON from the response
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                json_string = json_match.group()
                try:
                    search_query = json.loads(json_string)["search_query"]
                    console.print(Panel(f"Search Query: {search_query}", title="[bold blue]Search Query[/bold blue]", title_align="left", border_style="blue"))
                    response_text = response_text.replace(json_string, "").strip()
                except KeyError as key_error:
                    console.print(Panel(f"Error parsing JSON: {key_error}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
                    console.print(Panel(f"Skipping search query extraction.", title="[bold yellow]Search Query Extraction Skipped[/bold yellow]", title_align="left", border_style="yellow"))
                except json.JSONDecodeError as json_parsing_error:
                    console.print(Panel(f"Error parsing JSON: {json_parsing_error}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
                    console.print(Panel(f"Skipping search query extraction.", title="[bold yellow]Search Query Extraction Skipped[/bold yellow]", title_align="left", border_style="yellow"))
            else:
                search_query = None

    except Exception as e:
        console.print(Panel(f"Error in Opus Orchestrator: {e}", title="[bold red]Opus Orchestrator Error[/bold red]", title_align="left", border_style="red"))
        return None, None, None

    console.print(Panel(response_text, title=f"[bold green]Opus Orchestrator[/bold green]", title_align="left", border_style="green", subtitle="Sending task to Haiku 👇"))
    return response_text, file_content, search_query


def execute_subtask(prompt, search_query=None, previous_agent_tasks=None, use_search=False, continuation=False):
    """
    Generates a agent response based on the given prompt.

    Args:
        prompt (str): The prompt for generating the agent response.
        search_query (str, optional): The search query for performing a QnA search. Defaults to None.
        previous_agent_tasks (list, optional): A list of previous agent tasks. Defaults to None.
        use_search (bool, optional): Flag indicating whether to perform a QnA search. Defaults to False.
        continuation (bool, optional): Flag indicating whether the response is a continuation from a previous answer. Defaults to False.

    Returns:
        str: The generated agent response.
    """
    if previous_agent_tasks is None:
        previous_agent_tasks = []

    continuation_prompt = "Continuing from the previous answer, please complete the response."
    system_message = "Previous Haiku tasks:\n" + "\n".join(f"Task: {task['task']}\nResult: {task['result']}" for task in previous_agent_tasks)
    if continuation:
        prompt = continuation_prompt

    qna_response = None
    if search_query and use_search:
        # Initialize the Tavily client
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        # Perform a QnA search based on the search query
        qna_response = tavily.qna_search(query=search_query)
        console.print(f"QnA response: {qna_response}", style="yellow")


    messages = [
        {"role": "user",
         "content": [
             {"text": prompt},
             {"text": f"\nSearch Results:\n{qna_response}" if qna_response else ""}
             ]
        }
    ]

    agent_params = {
        "modelId": agent_modelId,
        "system": [{"text": system_message}],
        "inferenceConfig": {"maxTokens": 4096 },
        "messages": messages,
    }

    agent_params = client.converse(**agent_params)


    response_text = agent_params['output']['message']['content'][0]['text']
    console.print(f"Input Tokens: {agent_params['usage']['inputTokens']}, Output Tokens: {agent_params['usage']['outputTokens']}")
    total_cost = calculate_subagent_cost(agent_modelId, agent_params['usage']['inputTokens'], agent_params['usage']['outputTokens'])
    console.print(f"Haiku Sub-agent Cost: ${total_cost:.2f}")

    if agent_params['usage']['outputTokens'] >= 4000:  # Threshold set to 4000 as a precaution
        console.print("[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response.")
        continuation_response_text = execute_subtask(prompt, search_query, previous_agent_tasks, use_search, continuation=True)
        response_text += continuation_response_text

    console.print(Panel(response_text, title="[bold blue]Haiku Sub-agent Result[/bold blue]", title_align="left", border_style="blue", subtitle="Task completed, sending result to Opus 👇"))
    return response_text


def generate_refined_output(objective, sub_task_results, filename, projectname, continuation=False):
    """
    Calls Opus to provide the refined final output for the given objective.

    Args:
        objective (str): The objective for the refined final output.
        sub_task_results (list): A list of sub-task results.
        filename (str): The filename of the code file.
        projectname (str): The name of the project.
        continuation (bool, optional): Indicates whether the function is being called as part of a continuation. Defaults to False.

    Returns:
        str: The refined final output.

    Raises:
        <Exceptions>: Any exceptions that may occur during the execution of the function.

    """
    print("\nCalling Opus to provide the refined final output for your objective:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Objective: " + objective + "\n\nSub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\n​python\n<code>\n​"}
            ]
        }
    ]

    try:
        orchestrator_response = client.messages.create(
            modelId= orchestrator_modelId,
            max_tokens=4096,
            messages=messages
        )

        response_text = orchestrator_response['output']['message']['content'][0]['text'].strip()
        console.print(f"Input Tokens: {orchestrator_response['usage']['inputTokens']}, Output Tokens: {orchestrator_response['usage']['outputTokens']}")
        total_cost = calculate_subagent_cost(orchestrator_modelId, orchestrator_response['usage']['inputTokens'], orchestrator_response['usage']['outputTokens'])
        console.print(f"Opus Refine Cost: ${total_cost:.2f}")

        if orchestrator_response['usage']['outputTokens'] >= 4000 and not continuation:  # Threshold set to 4000 as a precaution
            console.print("[bold yellow]Warning:[/bold yellow] Output may be truncated. Attempting to continue the response.")
            continuation_response_text = generate_refined_output(objective, sub_task_results + [response_text], filename, projectname, continuation=True)
            response_text += "\n" + continuation_response_text

    except Exception as e:
        console.print(Panel(f"Error in Opus Refine: {e}", title="[bold red]Opus Refine Error[/bold red]", title_align="left", border_style="red"))
        return None

    console.print(Panel(response_text, title="[bold green]Final Output[/bold green]", title_align="left", border_style="green"))
    return response_text


def create_project_structure(project_name, folder_structure, code_blocks):
    """
    Create the folder structure for a project and populate it with files.

    Args:
        project_name (str): The name of the project folder to be created.
        folder_structure (dict): A dictionary representing the desired folder structure.
        code_blocks (list): A list of code blocks to be written to files.

    Returns:
        None
    """
    # Create the project folder
    project_path = Path(project_name)
    try:
        project_path.mkdir(parents=True, exist_ok=True)
        console.print(Panel(f"Created project folder: [bold]{project_path}[/bold]", title="[bold green]Project Folder[/bold green]", title_align="left", border_style="green"))
    except FileNotFoundError as file_not_found_error:
        console.print(Panel(f"Error creating project folder: [bold]{project_path}[/bold]\nError: {file_not_found_error}", title="[bold red]Project Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        return
    except PermissionError as permission_error:
        console.print(Panel(f"Permission denied while creating project folder: [bold]{project_path}[/bold]\nError: {permission_error}", title="[bold red]Project Folder Creation Permission Error[/bold red]", title_align="left", border_style="red"))
        return
    except OSError as os_error:
        console.print(Panel(f"OS error occurred while creating project folder: [bold]{project_path}[/bold]\nError: {os_error}", title="[bold red]Project Folder Creation OS Error[/bold red]", title_align="left", border_style="red"))
        return
    except Exception as general_error:
        console.print(Panel(f"An unexpected error occurred while creating project folder: [bold]{project_path}[/bold]\nError: {general_error}", title="[bold red]Project Folder Creation Unexpected Error[/bold red]", title_align="left", border_style="red"))
        return

    # Recursively create the folder structure and files
    create_folders_and_files_recursively(project_path, folder_structure, code_blocks)


def create_folders_and_files_recursively(current_path, structure, code_blocks):
    """
    Recursively creates folders and files based on the given structure.

    Args:
        current_path (Path): The current path where the folders and files will be created.
        structure (dict): The structure of the folders and files to be created.
        code_blocks (list): A list of tuples containing the file name and its corresponding code content.

    Returns:
        None
    """
    for key, value in structure.items():
        path = current_path / key
        if isinstance(value, dict):
            try:
                path.mkdir(parents=True, exist_ok=True)
                console.print(Panel(f"Created folder: [bold]{path}[/bold]", title="[bold blue]Folder Creation[/bold blue]", title_align="left", border_style="blue"))
                create_folders_and_files_recursively(path, value, code_blocks)
            except FileNotFoundError as e:
                console.print(Panel(f"Error creating folder: [bold]{path}[/bold]\nError: {e}", title="[bold red]Folder Creation Error[/bold red]", title_align="left", border_style="red"))
            except PermissionError as e:
                console.print(Panel(f"Permission denied while creating folder: [bold]{path}[/bold]\nError: {e}", title="[bold red]Folder Creation Permission Error[/bold red]", title_align="left", border_style="red"))
        else:
            code_content = next((code for file, code in code_blocks if file == key), None)
            if code_content:
                try:
                    with path.open('w') as file:
                        file.write(code_content)
                    console.print(Panel(f"Created file: [bold]{path}[/bold]", title="[bold green]File Creation[/bold green]", title_align="left", border_style="green"))
                except FileNotFoundError as e:
                    console.print(Panel(f"Error creating file: [bold]{path}[/bold]\nError: {e}", title="[bold red]File Creation Error[/bold red]", title_align="left", border_style="red"))
                except PermissionError as e:
                    console.print(Panel(f"Permission denied while creating file: [bold]{path}[/bold]\nError: {e}", title="[bold red]File Creation Permission Error[/bold red]", title_align="left", border_style="red"))
            else:
                console.print(Panel(f"Code content not found for file: [bold]{key}[/bold]", title="[bold yellow]Missing Code Content[/bold yellow]", title_align="left", border_style="yellow"))


def read_file_content(file_path):
    """
    Read the content of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file.
            Returns None if there was an error reading the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If permission is denied while reading the file.
        IOError: If an IO error occurs while reading the file.
        Exception: If an unexpected error occurs while reading the file.
    """
    path = Path(file_path)
    try:
        with path.open('r') as file:
            content = file.read()
        return content
    except FileNotFoundError as file_not_found_error:
        console.print(Panel(f"Error reading file: [bold]{file_path}[/bold]\nError: {file_not_found_error}", title="[bold red]File Reading Error[/bold red]", title_align="left", border_style="red"))
        return None
    except PermissionError as permission_error:
        console.print(Panel(f"Permission denied while reading file: [bold]{file_path}[/bold]\nError: {permission_error}", title="[bold red]File Reading Permission Error[/bold red]", title_align="left", border_style="red"))
        return None
    except IOError as io_error:
        console.print(Panel(f"IO error occurred while reading file: [bold]{file_path}[/bold]\nError: {io_error}", title="[bold red]File Reading IO Error[/bold red]", title_align="left", border_style="red"))
        return None
    except Exception as general_error:
        console.print(Panel(f"An unexpected error occurred while reading file: [bold]{file_path}[/bold]\nError: {general_error}", title="[bold red]File Reading Unexpected Error[/bold red]", title_align="left", border_style="red"))
        return None

# Get the objective from user input
objective = input("Please enter your objective with or without a text file path: ")

# Check if the input contains a file path
if "./" in objective or "/" in objective:
    # Extract the file path from the objective
    file_path = re.findall(r'[./\w]+\.[\w]+', objective)[0]
    file_path = Path(file_path)
    # Read the file content
    file_content = read_file_content(file_path)
    # Update the objective string to remove the file path
    objective = objective.split(str(file_path))[0].strip()
else:
    file_content = None

# Ask the user if they want to use search
use_search = input("Do you want to use search? (y/n): ").lower() == 'y'

task_exchanges = []
agent_tasks = []

while True:
    # Call Orchestrator to break down the objective into the next sub-task or provide the final output
    previous_results = [result for _, result in task_exchanges]
    if not task_exchanges:
        # Pass the file content only in the first iteration if available
        orchestrator_result, file_content_for_agent, search_query = break_down_objective(objective, file_content, previous_results, use_search)
    else:
        orchestrator_result, _, search_query = break_down_objective(objective, previous_results=previous_results, use_search=use_search)

    if "The task is complete:" in orchestrator_result:
        # If Opus indicates the task is complete, exit the loop
        final_output = orchestrator_result.replace("The task is complete:", "").strip()
        break
    else:
        sub_task_prompt = orchestrator_result
        # Append file content to the prompt for the initial call to execute_subtask, if applicable
        if file_content_for_agent and not agent_tasks:
            sub_task_prompt = f"{sub_task_prompt}\n\nFile content:\n{file_content_for_agent}"
        # Call execute_subtask with the prepared prompt, search query, and record the result
        sub_task_result = execute_subtask(sub_task_prompt, search_query, agent_tasks, use_search)
        # Log the task and its result for future reference
        agent_tasks.append({"task": sub_task_prompt, "result": sub_task_result})
        # Record the exchange for processing and output generation
        task_exchanges.append((sub_task_prompt, sub_task_result))
        # Prevent file content from being included in future execute_subtask calls
        file_content_for_agent = None


# Create the .md filename
sanitized_objective = re.sub(r'\W+', '_', objective)
timestamp = datetime.now().strftime("%H-%M-%S")


# Call Opus to review and refine the sub-task results
refined_output = generate_refined_output(objective, [result for _, result in task_exchanges], timestamp, sanitized_objective)


# Extract the project name from the refined output
project_name_match = re.search(r'Project Name: (.*)', refined_output)
project_name = project_name_match.group(1).strip() if project_name_match else sanitized_objective


# Extract the folder structure from the refined output
folder_structure_match = re.search(r'<folder_structure>(.*?)</folder_structure>', refined_output, re.DOTALL)
folder_structure = {}
if folder_structure_match:
    json_string = folder_structure_match.group(1).strip()
    try:
        folder_structure = json.loads(json_string)
    except json.JSONDecodeError as json_parsing_error:
        console.print(Panel(f"Error parsing JSON: {json_parsing_error}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
        console.print(Panel(f"Invalid JSON string: [bold]{json_string}[/bold]", title="[bold red]Invalid JSON String[/bold red]", title_align="left", border_style="red"))


# Extract code files from the refined output
code_blocks = re.findall(r'Filename: (\S+)\s*```[\w]*\n(.*?)\n```', refined_output, re.DOTALL)


# Create the folder structure and code files
create_project_structure(project_name, folder_structure, code_blocks)


# Truncate the sanitized_objective to a maximum of 50 characters
max_length = 25
truncated_objective = sanitized_objective[:max_length] if len(sanitized_objective) > max_length else sanitized_objective


# Update the filename to include the project name
filename = f"{timestamp}_{truncated_objective}.md"
file_path = Path(project_name) / filename


# Prepare the full exchange log
exchange_log = f"Objective: {objective}\n\n"
exchange_log += "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
for i, (prompt, result) in enumerate(task_exchanges, start=1):
    exchange_log += f"Task {i}:\n"
    exchange_log += f"Prompt: {prompt}\n"
    exchange_log += f"Result: {result}\n\n"

exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n"
exchange_log += refined_output

console.print(f"\n[bold]Refined Final output:[/bold]\n{refined_output}")

file_path.parent.mkdir(parents=True, exist_ok=True)
with file_path.open('w') as file:
    file.write(exchange_log)
print(f"\nFull exchange log saved to {file_path}")

