#DAT 1
#creating a chatbot using langchain and groq
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient #this is the library that is used to connect the momgo DB to the chatbot
from datetime import datetime, timezone
#importing the library to deploy our chatbot using fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
#Adding the MongoDB connection string to the environment variables
mongo_uri = os.getenv("MONGO_URI")

#We are setting up a mongo_db connection by creating a client using pymongo
client = MongoClient(mongo_uri)
#this is the name of the database we will use
db = client["chatbot_db"] 
#this is the name of the collection we will use to store the conversations
collection = db["users"] 

#we are going to initialize the FastAPI app
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str
#we are going to allow all ips to access the fastapi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


prompt = ChatPromptTemplate.from_messages(
    [
("system", "You are a helpful assistant that helps users ask questions related to study topics only"),
("placeholder", "{history}"),
  ("human", "{question}")
    ]
)

chat = ChatGroq(api_key=groq_api_key,model="openai/gpt-oss-120b")
chain = prompt | chat



def get_history(user_id,limits=5):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1).limit(limits)
    history = []

    for chat in chats:
        history.append((chat["role"], chat["message"]))
    return history

#we are initializing some routes
@app.get("/")#root route
def home():
    return {"message": "Welcome to the study chatbot API!"}

@app.post("/chat")#route to ask questions
def chat(request: ChatRequest):
    history = get_history(request.user_id)

    response = chain.invoke({"history": history, "question": request.question})

    #store the user question in the database
    collection.insert_one({
        "user_id": request.user_id,
        "question": request.question,
        "role":"user",
        "message": request.question,
        "timestamp":datetime.utcnow()
    })

    #store the assistant response in the database
    collection.insert_one({
        "user_id": request.user_id,
        "role":"assistant",
        "message": response.content,
        "timestamp":datetime.utcnow()
    })

    return {"response": response.content}

