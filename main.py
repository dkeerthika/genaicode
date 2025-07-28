import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI

from tools.flight_finder import flight_finder
from tools.hotel_finder import hotel_finder
from tools.weather_tool import weather_tool

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4")

tools = [
    Tool(name="FlightFinder", func=flight_finder, description="Finds flights to a city"),
    Tool(name="HotelFinder", func=hotel_finder, description="Finds hotels in a city"),
    Tool(name="WeatherTool", func=weather_tool, description="Gives weather updates for a city"),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Example use
response = agent.run("I'm planning a trip to Goa. Can you find flights, hotels, and tell me the weather?")
print(response)
