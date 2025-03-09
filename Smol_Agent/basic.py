from smolagents import CodeAgent, HfApiModel
import os
from dotenv import load_dotenv
load_dotenv()
#model_id = "google/gemma-7b" 

#model = HfApiModel(model_id=model_id) # You can choose to not pass any model_id to HfApiModel to use a default free model
model= HfApiModel()
# you can also specify a particular provider e.g. provider="together" or provider="sambanova"
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)