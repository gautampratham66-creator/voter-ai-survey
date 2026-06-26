"""
agents/crew_agents.py
Multi-Agent System using CrewAI + LangChain
Each agent has a specific role in the Voter ID Survey pipeline
"""

# ─── Install requirements ───────────────────────────────────────────────────
# pip install crewai langchain langchain-anthropic chromadb

from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
import json

# ─── LLM Brain (Claude) ─────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key="YOUR_ANTHROPIC_API_KEY"  # Replace with actual key
)

# ─── Custom Tools for Agents ────────────────────────────────────────────────

def extract_document_info(document_text: str) -> dict:
    """
    Document Agent Tool:
    Extracts Name, DOB, Gender from Aadhaar / ID document text.
    In production: uses Claude Vision API on uploaded image.
    """
    # In real implementation:
    # response = claude_client.messages.create(
    #     model="claude-sonnet-4-6",
    #     messages=[{"role":"user","content":[
    #         {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data": img_b64}},
    #         {"type":"text","text":"Extract name, date of birth, gender from this Aadhaar card. Return JSON."}
    #     ]}]
    # )
    return {
        "name": "Ramesh Kumar",
        "dob": "12/04/1985",
        "age": 39,
        "gender": "Male",
        "address": "Village Kairana, Shamli, UP",
        "eligible": True  # age >= 18
    }

def store_family_survey(family_data: dict) -> str:
    """
    Survey Agent Tool:
    Validates and stores family survey in database.
    In production: writes to PostgreSQL.
    """
    # In real: cursor.execute("INSERT INTO families ...")
    print(f"[Survey Agent] Storing family: {family_data.get('head')}")
    return f"✅ Family '{family_data.get('head')}' stored successfully with ID F{__import__('random').randint(100,999)}"

def query_survey_database(question: str) -> str:
    """
    RAG Agent Tool:
    Retrieves relevant records from ChromaDB vector store.
    In production: uses ChromaDB + embeddings.
    """
    # In real:
    # from chromadb import Client
    # client = Client()
    # collection = client.get_collection("voter_survey")
    # results = collection.query(query_texts=[question], n_results=5)
    return """
    Retrieved records:
    - Total families: 4, Total members: 18
    - Eligible (age>=18): 14
    - Have Voter ID: 9 (64.3% coverage)
    - Missing: 5 (35.7%)
    - Female missing: 3, Male missing: 2
    - Worst district: Muzaffarnagar (50% coverage)
    """

def generate_district_report(district: str) -> str:
    """
    Report Agent Tool:
    Generates AI-written report for a district.
    """
    return f"""
    VOTER ID SURVEY REPORT — {district.upper()}
    Date: {__import__('datetime').date.today()}
    
    Executive Summary:
    District {district} shows moderate voter registration coverage.
    Key findings: 3 out of 6 eligible citizens lack Voter ID.
    Gender gap: Female registration is 20% lower than male.
    
    Recommendations:
    1. Conduct Voter ID drive in low-coverage villages
    2. Focus on female outreach programs
    3. Set up mobile enrollment camps
    """

# ─── Tools wrapped for CrewAI ───────────────────────────────────────────────
doc_extract_tool = Tool(
    name="DocumentExtractor",
    func=extract_document_info,
    description="Extracts Name, DOB, Gender, Address from Aadhaar/ID documents using Claude Vision"
)

survey_store_tool = Tool(
    name="SurveyStorage",
    func=store_family_survey,
    description="Validates and stores family survey data into PostgreSQL database"
)

rag_query_tool = Tool(
    name="RAGQueryTool",
    func=query_survey_database,
    description="Retrieves relevant voter survey data from ChromaDB vector database to answer questions"
)

report_gen_tool = Tool(
    name="ReportGenerator",
    func=generate_district_report,
    description="Generates a comprehensive AI-written report for a given district"
)

# ─── AGENT 1: Document Processing Agent ─────────────────────────────────────
document_agent = Agent(
    role="Document Processing Specialist",
    goal="Extract and validate citizen information from uploaded ID documents",
    backstory="""You are an expert document analyst. You receive Aadhaar cards,
    birth certificates, and other ID proofs. You extract key information (name, 
    age, gender, address) and validate if the person is eligible for Voter ID 
    (age >= 18). You are precise and never make errors in age calculation.""",
    tools=[doc_extract_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ─── AGENT 2: Survey Management Agent ───────────────────────────────────────
survey_agent = Agent(
    role="Survey Data Manager",
    goal="Collect, validate, and store complete family survey data",
    backstory="""You are a meticulous data manager for the state's voter ID 
    survey program. You receive extracted document info from the Document Agent,
    ask for any missing details, check for duplicate entries, and store complete
    family records in the database. You flag eligible citizens who don't have 
    Voter IDs for follow-up.""",
    tools=[survey_store_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ─── AGENT 3: RAG Query Agent ────────────────────────────────────────────────
rag_agent = Agent(
    role="Survey Data Analyst",
    goal="Answer questions about voter ID survey data using RAG retrieval",
    backstory="""You are an intelligent data analyst with access to the complete
    voter ID survey database. When asked a question, you retrieve the most relevant
    records using semantic search (ChromaDB), then synthesize an accurate, 
    data-grounded answer. You never guess — you always base answers on retrieved data.""",
    tools=[rag_query_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ─── AGENT 4: Report Generation Agent ───────────────────────────────────────
report_agent = Agent(
    role="Government Report Writer",
    goal="Generate comprehensive, actionable district-level survey reports",
    backstory="""You are a senior analyst writing official reports for district
    election officers. You take survey statistics and write clear, structured 
    reports highlighting coverage gaps, gender disparities, and actionable 
    recommendations for improving voter registration in each district.""",
    tools=[report_gen_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ─── Tasks ───────────────────────────────────────────────────────────────────
def create_survey_crew(document_text: str, district: str):
    """
    Creates and runs the full multi-agent survey pipeline.
    Agents work sequentially: Doc → Survey → RAG → Report
    """

    task_extract = Task(
        description=f"Extract all citizen information from this document: {document_text}. Validate age eligibility.",
        agent=document_agent,
        expected_output="Extracted citizen info: name, age, gender, address, eligibility status"
    )

    task_survey = Task(
        description="Take the extracted citizen info and store a complete family survey record. Check for duplicates.",
        agent=survey_agent,
        expected_output="Confirmation of data stored with family ID and summary of eligible/missing members"
    )

    task_rag = Task(
        description=f"Query the survey database for current statistics on district {district}. Summarize key findings.",
        agent=rag_agent,
        expected_output="Summary of voter ID coverage stats for the district"
    )

    task_report = Task(
        description=f"Generate a full district survey report for {district} based on current data and statistics.",
        agent=report_agent,
        expected_output="Complete formatted report with executive summary, findings, and recommendations"
    )

    # ─── Crew Orchestration ──────────────────────────────────────────────────
    crew = Crew(
        agents=[document_agent, survey_agent, rag_agent, report_agent],
        tasks=[task_extract, task_survey, task_rag, task_report],
        process=Process.sequential,  # Agents work one after another
        verbose=True
    )

    result = crew.kickoff()
    return result


# ─── Example Usage ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = create_survey_crew(
        document_text="Aadhaar card image of Ramesh Kumar, DOB 12/04/1985, Male, Kairana",
        district="Shamli"
    )
    print("\n=== CREW RESULT ===")
    print(result)
