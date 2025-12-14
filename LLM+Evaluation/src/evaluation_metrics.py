import time
from typing import Dict, Any
import json
from deepeval.test_case import LLMTestCase

actual_output = ''

# Custom Metric for Latency and Cost Estimation
def performance_metrics(latency, cost_per_token_usd) -> Dict[str, Any]:

    # Simple token estimation: 1 word ~ 1.3 tokens
    token_count = len(actual_output.split()) * 1.3 
    mock_cost_usd = token_count * cost_per_token_usd
    
    # Performance passes if latency is below 2.0 seconds
    success = latency < 2.0

    perf_results = {
                    "success":success, 
                    "score":latency, 
                    "metric_metadata":{
                        "Latency (seconds)": round(latency, 4),
                        "Mock Cost (USD)": round(mock_cost_usd, 6)
                        },
                        "name":"Performance",
                        "reason":"Latency and cost metrics calculated."
                        }
    
    return perf_results


def load_data_prepare_test_case(chat_json_path: str, context_json_path: str, user_turn , ai_turn) -> Dict[str, Any]:
    """Loads and simulates the input JSON data."""
    user_query = None
    ai_response = None
    global actual_output
    try:
        with open(chat_json_path, 'r') as f:
            chat_data = json.load(f)
        with open(context_json_path, 'r') as f:
            context_data = json.load(f)
        chat_history = chat_data["conversation_turns"]
        context_data = context_data["data"]["vector_data"]  

        for turn in chat_history:
            if turn['turn'] == user_turn and turn['role'] == 'User':
                user_query = turn['message']
            elif turn['turn'] == ai_turn and turn['role'] == 'AI/Chatbot':
                ai_response = turn['message']
            
            # Optimization: stop searching once both are found
            if user_query is not None and ai_response is not None:
                break

        context_vectors = [v['text'] for v in context_data if 'text' in v]
        actual_output = ai_response
        data = {
                    "user_query":user_query, 
                    "actual_output":ai_response, 
                    "context_vectors":context_vectors,
                        }
        return data
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None
