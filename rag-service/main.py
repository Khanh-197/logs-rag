import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from datetime import datetime, timedelta
import json
from elasticsearch import Elasticsearch
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Logs RAG API")

# Elasticsearch configuration
es_host = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
es_port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
es_user = os.getenv("ELASTICSEARCH_USER", "")
es_password = os.getenv("ELASTICSEARCH_PASSWORD", "")

# Ollama configuration
ollama_host = os.getenv("OLLAMA_HOST", "ollama")
ollama_port = os.getenv("OLLAMA_PORT", "11434")
ollama_url = f"http://{ollama_host}:{ollama_port}"
ollama_model = os.getenv("OLLAMA_MODEL", "llama3:8b")

# ChromaDB configuration
chroma_host = os.getenv("CHROMA_HOST", "chroma")
chroma_port = os.getenv("CHROMA_PORT", "8000")

# Initialize embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Initialize Elasticsearch client
es_client = None
if es_user and es_password:
    es_client = Elasticsearch(
        f"http://{es_host}:{es_port}",
        basic_auth=(es_user, es_password)
    )
else:
    es_client = Elasticsearch(f"http://{es_host}:{es_port}")

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))

# Create collection if it doesn't exist
try:
    logs_collection = chroma_client.get_collection("logs_collection")
    logger.info("Connected to existing ChromaDB collection: logs_collection")
except:
    logs_collection = chroma_client.create_collection(
        name="logs_collection",
        embedding_function=embedding_function
    )
    logger.info("Created new ChromaDB collection: logs_collection")

# Text splitter for processing logs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

class QueryRequest(BaseModel):
    query: str
    time_range: Optional[str] = "24h"  # Default to last 24 hours

class TransactionRequest(BaseModel):
    transaction_id: str

class LogSyncRequest(BaseModel):
    index_pattern: str = "logstash-*"
    hours_back: int = 24

@app.get("/")
async def root():
    return {"message": "Logs RAG API is running"}

@app.post("/query")
async def query_logs(request: QueryRequest):
    """
    Query the logs using natural language and get AI-powered responses
    """
    try:
        # Step 1: Convert query to embeddings and search in ChromaDB
        results = logs_collection.query(
            query_texts=[request.query],
            n_results=5
        )
        
        # Step 2: Get relevant log contexts
        contexts = results.get("documents", [[]])[0]
        
        if not contexts:
            # Fallback to direct Elasticsearch search if no context in ChromaDB
            time_filter = get_time_filter(request.time_range)
            es_results = await search_elasticsearch(request.query, time_filter)
            contexts = [json.dumps(hit["_source"]) for hit in es_results.get("hits", {}).get("hits", [])]
        
        # Step 3: Prepare prompt with contexts
        prompt = f"""
You are an AI assistant specialized in analyzing system logs. 
Based on the following log data, please answer this question: "{request.query}"

Log data:
{contexts}

Please provide a clear, concise answer based only on the provided logs. If the logs don't contain relevant information, please state that.
"""
        
        # Step 4: Get response from Ollama
        response = await query_ollama(prompt)
        
        return {
            "query": request.query,
            "response": response,
            "sources": contexts[:3]  # Include top sources for reference
        }
    
    except Exception as e:
        logger.error(f"Error in query_logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/transaction")
async def get_transaction(request: TransactionRequest):
    """
    Get details for a specific transaction ID
    """
    try:
        # Search for the transaction in logs
        query_body = {
            "query": {
                "multi_match": {
                    "query": request.transaction_id,
                    "fields": ["transId", "transaction_id", "transactionId", "message", "*"]
                }
            }
        }
        
        response = es_client.search(
            index="logstash-*",
            body=query_body,
            size=20
        )
        
        hits = response["hits"]["hits"]
        
        if not hits:
            return {"message": f"No logs found for transaction ID: {request.transaction_id}"}
        
        # Extract relevant log entries
        log_entries = [hit["_source"] for hit in hits]
        
        # Create a context from logs
        log_context = "\n".join([json.dumps(entry) for entry in log_entries])
        
        # Query the LLM to analyze the transaction
        prompt = f"""
You are an AI assistant specialized in analyzing system logs.
Analyze these logs for transaction ID {request.transaction_id} and provide a summary:

{log_context}

Please provide:
1. Transaction status (success/failed)
2. Timestamp information (when it started/completed)
3. Any errors or issues found
4. A brief summary of the transaction flow
"""
        
        analysis = await query_ollama(prompt)
        
        return {
            "transaction_id": request.transaction_id,
            "analysis": analysis,
            "logs": log_entries
        }
    
    except Exception as e:
        logger.error(f"Error in get_transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving transaction: {str(e)}")

@app.post("/sync-logs")
async def sync_logs(request: LogSyncRequest):
    """
    Sync logs from Elasticsearch to ChromaDB for vector search
    """
    try:
        # Calculate time range
        time_from = datetime.now() - timedelta(hours=request.hours_back)
        time_to = datetime.now()
        
        # Query Elasticsearch for recent logs
        query_body = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": time_from.isoformat(),
                        "lte": time_to.isoformat()
                    }
                }
            },
            "sort": [
                {"@timestamp": {"order": "desc"}}
            ]
        }
        
        # Use scroll API for large datasets
        response = es_client.search(
            index=request.index_pattern,
            body=query_body,
            scroll="2m",
            size=1000
        )
        
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        total_hits = len(hits)
        
        # Process logs in batches
        all_logs = []
        batch_count = 1
        
        while hits:
            logger.info(f"Processing batch {batch_count} with {len(hits)} logs")
            
            # Extract log content
            for hit in hits:
                source = hit["_source"]
                log_id = hit["_id"]
                
                # Convert log to string representation
                log_content = json.dumps(source)
                all_logs.append((log_id, log_content))
            
            # Get next batch
            response = es_client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
            total_hits += len(hits)
            batch_count += 1
        
        # Clear scroll
        es_client.clear_scroll(scroll_id=scroll_id)
        
        # Process logs and add to ChromaDB
        added_count = 0
        
        # Process in chunks to avoid memory issues
        chunk_size = 100
        for i in range(0, len(all_logs), chunk_size):
            chunk = all_logs[i:i+chunk_size]
            
            ids = [item[0] for item in chunk]
            documents = [item[1] for item in chunk]
            
            # Add to ChromaDB
            logs_collection.add(
                ids=ids,
                documents=documents,
                metadatas=[{"source": "elasticsearch"} for _ in chunk]
            )
            
            added_count += len(chunk)
            logger.info(f"Added {added_count}/{len(all_logs)} logs to ChromaDB")
        
        return {
            "message": f"Successfully synced {added_count} logs from Elasticsearch to ChromaDB",
            "logs_processed": total_hits,
            "logs_added": added_count
        }
    
    except Exception as e:
        logger.error(f"Error in sync_logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error syncing logs: {str(e)}")

async def search_elasticsearch(query: str, time_filter: Dict) -> Dict:
    """
    Search Elasticsearch directly using the query
    """
    try:
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["message", "*"],
                                "type": "best_fields"
                            }
                        },
                        time_filter
                    ]
                }
            },
            "size": 20
        }
        
        response = es_client.search(
            index="logstash-*",
            body=search_body
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error searching Elasticsearch: {str(e)}")
        return {"hits": {"hits": []}}

def get_time_filter(time_range: str) -> Dict:
    """
    Convert time range string (e.g. '24h', '7d') to Elasticsearch filter
    """
    now = datetime.now()
    
    if time_range.endswith('h'):
        hours = int(time_range[:-1])
        from_time = now - timedelta(hours=hours)
    elif time_range.endswith('d'):
        days = int(time_range[:-1])
        from_time = now - timedelta(days=days)
    elif time_range.endswith('m'):
        minutes = int(time_range[:-1])
        from_time = now - timedelta(minutes=minutes)
    else:
        # Default to 24 hours
        from_time = now - timedelta(hours=24)
    
    return {
        "range": {
            "@timestamp": {
                "gte": from_time.isoformat(),
                "lte": now.isoformat()
            }
        }
    }

async def query_ollama(prompt: str) -> str:
    """
    Query Ollama LLM with the given prompt
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.text}")
                return "Sorry, I encountered an issue processing your request."
            
            result = response.json()
            return result.get("response", "")
    
    except Exception as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        return f"Error: Could not connect to LLM service. {str(e)}"

@app.get("/health")
async def health_check():
    """
    Check the health of all connected services
    """
    health_status = {
        "api": "healthy",
        "elasticsearch": "unknown",
        "chroma": "unknown",
        "ollama": "unknown"
    }
    
    # Check Elasticsearch
    try:
        es_health = es_client.cluster.health()
        health_status["elasticsearch"] = es_health["status"]
    except Exception as e:
        health_status["elasticsearch"] = f"unhealthy: {str(e)}"
    
    # Check ChromaDB
    try:
        chroma_collections = chroma_client.list_collections()
        health_status["chroma"] = "healthy"
    except Exception as e:
        health_status["chroma"] = f"unhealthy: {str(e)}"
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                health_status["ollama"] = "healthy"
            else:
                health_status["ollama"] = f"unhealthy: status {response.status_code}"
    except Exception as e:
        health_status["ollama"] = f"unhealthy: {str(e)}"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
