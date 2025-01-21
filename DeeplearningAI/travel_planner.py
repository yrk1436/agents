from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import os
from langchain_openai import ChatOpenAI

# from signals.helpers import Utility
from utils import get_openai_api_key, get_serper_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = get_serper_api_key()

# openai_api_key = Utility.get_cached_secret('openai_api_key')
# os.environ["OPENAI_API_KEY"] = openai_api_key
# os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
# os.environ["SERPER_API_KEY"] = "XXXXX"



search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

location_agent = Agent(
    role="Travel Planning Specialist",
    goal="Identify the best locations and attractions around {city} based on the traveler's preferences.",
    backstory=(
        "As a member of the travel planning team, your primary goal is to identify key attractions and destinations "
        "near or around {city} for a {number_of_days}-day trip. Consider the traveler is visiting with a family of four, "
        "including two children aged 5 and 8. Assume the traveler is flying into {city} and will rent a car for local transportation. "
        "Focus on destinations that are suitable for children, ensuring the attractions are family-friendly and manageable within the given number of days. "
        "In your selections, prioritize top-rated attractions and places that are easy to reach by car. "
        "Pay attention to the specific month of travel, {month}, to suggest destinations that are ideal for the season. "
        "Ensure that the locations you shortlist are kid-friendly and consider factors like weather and travel comfort during this time of year."
    ),
    allow_delegation=False,
    verbose=True
)


itenary_agent = Agent(
    role="Itinerary Planning Specialist",
    goal="Create a detailed, family-friendly itinerary based on the shortlisted attractions from {city}.",
    backstory=(
        "As an expert in crafting personalized travel itineraries, your task is to create a detailed itinerary "
        "for a family visiting the shortlisted attractions around {city}. Your focus should be on creating a well-paced, "
        "relaxed itinerary that suits children aged 5 and 8 while also keeping the parents engaged. "
        "Consider travel times between locations, appropriate stops to rest, and meal options. "
        "For attractions outside of {city}, suggest family-friendly accommodation options, preferably with breakfast included, "
        "and accommodations that cater to children. Make sure to account for driving distances, breaks during the journey, and "
        "give suggestions for clothing and items to pack based on the {month} of travel. "
        "Your role is crucial in making the family's vacation smooth and enjoyable, from planning their days to ensuring they have everything they need for a memorable experience."
    ),
    allow_delegation=False,
    verbose=True
)


shortlist_locations_task = Task(
    description=(
        "Using the input {city}, {month}, and {number_of_days} provided by the traveler, "
        "the goal of this task is to identify the most suitable attractions and locations to visit near {city}. "
        "The focus should be on family-friendly destinations, considering a family of four with children aged 5 and 8. "
        "The traveler is planning to rent a car, so locations within a reasonable driving distance from {city} should be prioritized. "
        "Make sure to account for the season, {month}, and ensure the attractions are enjoyable and comfortable for kids. "
        "When shortlisting, include a mix of city-based and natural attractions (if applicable), "
        "and ensure the trip is suitable for a {number_of_days}-day visit."
        "ask user for confirming if he/she is good to add the location into shortlist for each suggestion before finializing it"
        "if user says not interested or NA, ask if they are interested to add any other particular attraction into the list. And try to "
        "consider it at all cost provided it can be covered in their stay"
    ),
    expected_output=(
        "A list of carefully chosen family-friendly attractions and locations near {city}, "
        "along with their suitability based on the traveler's {month} trip. "
        "Each suggestion should be a destination that can be visited comfortably within {number_of_days}, "
        "with consideration for drive times and kid-friendly activities."
    ),
    human_input=True,
    tools=[location_agent],
    agent=location_agent
)


create_itenary_task = Task(
    description=(
        "Now that the shortlist of family-friendly locations and attractions near {city} is ready, "
        "this task involves creating a detailed travel itinerary. "
        "The focus should be on ensuring a smooth flow of activities across {number_of_days}, considering the needs of both the parents and children. "
        "The itinerary should prioritize attractions that are suitable for kids aged 5 and 8, including rest breaks, meal stops, and suitable accommodation. "
        "Accommodation should be recommended based on comfort and convenience, preferably with breakfast included, "
        "and should take into account the {month} of travel. "
        "The itinerary should also suggest the best clothing and travel tools to carry for the trip, ensuring everything is family-friendly."
    ),
    expected_output=(
        "A comprehensive itinerary detailing daily activities for the family, with rest breaks, meal suggestions, "
        "kid-friendly accommodations, and travel tips. The itinerary should be well-paced, ensuring the family enjoys each day "
        "without feeling rushed, and considers the {month} of travel when suggesting appropriate clothing and tools."
    ),
    tools=[location_agent],
    agent=itenary_agent
)

# Define the crew with agents and tasks
travel_planning_crew = Crew(
    agents=[location_agent,             
            itenary_agent],
    
    tasks=[shortlist_locations_task, 
           create_itenary_task],
    
    manager_llm=ChatOpenAI(model="gpt-4o-2024-08-06", 
                           temperature=0.7),
    process=Process.hierarchical,
    verbose=True
)

# Example data for kicking off the process
inputs = {
    'city': 'Vegas',
    'number_of_days': '7',
    'month': 'Febraury'
}

### this execution will take some time to run
result = travel_planning_crew.kickoff(inputs=inputs)

print(result)

# from IPython.display import Markdown
# Markdown(result)