from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

model = HfApiModel()
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

#agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
result=agent.run("What is the latest version of Microsot Autogen Agent Framework?")
print(result)