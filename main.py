from langchain_community.llms import Ollama

from langchain import hub
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, BaseTool
from typing import Dict, Union, Tuple
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults, Tool
import platform
import json
import docker
from docker.errors import ContainerError
import asyncio
import pprint
from rich import print
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.tree import Tree
from rich.console import Console, Group
from rich.syntax import Syntax
from rich.panel import Panel
import logging
from langchain.memory import ChatMessageHistory


console = Console()

# Get the prompt to use - you can modify this!
#prompt = hub.pull("hwchase17/react")

# template based on hwchase17/react-chat, trans from PromptTemplate to ChatPromptTemplate
template = """
The assistant, built on the 'Qwen' large language model, acts as an adaptive system manager, capable of intelligently handling tasks by tailoring its actions to the specific needs of the host operating system and the current user's permissions.

Its primary role is to directly handle tasks on the host operating system based on user requests, without merely instructing the user on how to complete them.

When a task is requested, the assistant:
    - Detection and adaptation to the specific Linux distribution or any other operating system version in use. This allows the assistant to select and use commands and utilities that are compatible and optimal for the particular system environment.
    - Recognition and adjustment according to the user's permission level. The assistant generates commands that are executable within the user's current access rights, avoiding commands that require higher privileges unless those rights are available.
    - Automatically generates and executes the necessary Bash commands or scripts to accomplish the task directly. It avoids using any interactive or manual commands, ensuring all operations are fully autonomous.
    - Robust error handling and security measures in place to prevent execution failures and protect against vulnerabilities. Feedback is promptly provided to the user about the execution status and results through straightforward outputs or detailed logs.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Before output the "Final Answer", you MUST make sure it is solely a batch of commands.
Let's think step by step.

Here are some examples:
### Example 1
Question: List all files in the current directory.
Thought: I need to find out what my host OS (OS distro specially) is so that I can know which kind of commands to output
Action: get_platform_info
Action Input:
Observation: I now know my host OS is Linux, so I should generate Linux commands
Thought: command `ls ./` can list files in the current directory under Linux
Action: run_commands
Action Input: ls ./
Observation: I have get the output
Thought: Do I need to use a tool? No
Final Answer: the output

### Example 2
Question: What the distro is?
Thought: I need to find out what my host OS (OS distro specially) is so that I can know which kind of commands to output
Action: get_platform_info
Action Input:
Observation: I now know what my host OS is
Thought: `neofetch` will show the distro info, but I need to check if this command is available
Action: run_commands
Action Input: which neofetch
Observation: The output is "neofetch not found", which means `neofetch` is not available. I need to find another way
Thought: Try `lsb_release -a`, but I need to check if this command is available
Action: run_commands
Action Input: which lsb_release
Observation: The output is "lsb_release not found", which means `lsb_release` is not available. I need to find another way
Thought: File /etc/os-release may also contain distro info
Action: run_commands
Action Input: cat /etc/os-release
Observation: It turns out that the file exists and its content shows the distro is "Ubuntu 22.04.4 LTS"
Thought: Do I need to use a tool? No
Final Answer: Ubuntu 22.04.4 LTS

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

#llm = Ollama(model="llama3:8b-instruct-fp16")
llm = Ollama(model="qwen:14b")
llm.temperature = 0.5
llm.top_k = 10
llm.top_p = 0.5
# Construct the ReAct agent
search = DuckDuckGoSearchResults()
search.handle_tool_error = True # agent will continue executing and the error will be printed

class GetPlatformInfoTool(BaseTool):
    name = "get_platform_info"
    description = "Returns the basic information (system, node, kernel release, OS distro version, machine, processor) of host platform."

    # override to make this tool take no input, check [issue](https://github.com/langchain-ai/langchain/issues/7685)
    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        return (), {}

    def _run(self):
        return get_platform_info()


def get_platform_info() -> dict:
    system_info = platform.uname()

    # 将uname_result转化为字典
    system_info_dict = {
            "system": system_info.system,
            "node": system_info.node,
            "release": system_info.release,
            "version": system_info.version,
            "machine": system_info.machine,
            "processor": system_info.processor,
    }

    # 将字典转化为JSON字符串
    return json.dumps(system_info_dict)

@tool
def run_commands(commands: str) -> str:
    """ Tool to run shell commands. Input string consists solely of bash commands or bash script with explanation in comments. Returns the output of commands. """
    # is_run = Confirm.ask("Confirm to run?", console=console, default=True)
    # if not is_run:
    #     return "The user refuse to run the commands, maybe the commands are wrong or cannot satisfy the requirements."
    client = docker.from_env()
    container = None
    try:
        container = client.containers.create(image="ubuntu:22.04", command=commands)
        container.start()
        # Wait for the container to finish and capture the exit status
        result = container.wait()
        output = container.logs(stdout=True, stderr=True)
    except Exception as e:
        # Handle other exceptions that might occur
        output = str(e).encode()  # Ensuring the output is bytes-like for consistency
    finally:
        if container is not None:
            # Remove the container in all cases
            container.remove()
    return output.decode('utf-8')

tools = [DuckDuckGoSearchResults(), GetPlatformInfoTool(), run_commands]

agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.handle_parsing_errors = True

class Prompt(Prompt):
    prompt_suffix = " "

if __name__ == "__main__":
    round = 0
    history = ChatMessageHistory()
    while True:
        console.rule("ROUND " + str(round))
        round += 1
        input = Prompt.ask("[bold green]:man:")
        history.add_user_message(input)
        if input == "exit" or input == "quit" or input == "q":
            console.print("[bold green]Good Bye!:wave:")
            exit(0)
        if input == "clean" or input == "clear":
            history.clear()
            console.print("[bold green]Chat history cleared!:innocent:")
            continue
        console.print()
        tree = Tree("[bold green]:robot:", highlight=True, guide_style="bold green")
        with Live(tree, auto_refresh=False) as live:
            with console.status("[green]Thinking:light_bulb:...", spinner="arc"):
                for output in agent_executor.stream({"input": input, "chat_history": history.messages}):
                    if "actions" in output: # AgentAction
                        action_branch = tree.add("[bold yellow]:runner: Action", guide_style="bold yellow")
                        for agent_action in output["actions"]:
                            log_panel = Panel.fit(agent_action.log,  border_style="orange3")
                            action_branch.add(Group("[bold orange3]:thought_balloon: Thought", log_panel))
                            if agent_action.tool_input:
                                syntax = Syntax(agent_action.tool_input, "bash", theme="gruvbox-dark", line_numbers=True)
                            else:
                                syntax = ""
                            action_branch.add(Group("[bold orange3]:hammer: " + agent_action.tool, syntax))
                    elif "steps" in output: # AgentStep
                        for agent_step in output["steps"]:
                            tree.add(Group("[bold yellow]:eyes: Observation", agent_step.observation))
                    elif "output" in output: # Output
                        history.add_ai_message(output["output"])
                        output_panel = Panel.fit(output["output"], border_style="bold yellow")
                        tree.add(Group("[bold yellow]:star2: Output", output_panel))
                    live.refresh()
                    #live.console.log("-----------------------------------")
                    #live.console.log(output)

