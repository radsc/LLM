import time
from typing import Dict, Any
import json
from deepeval.test_case import LLMTestCase

# --- Custom Metric for Latency and Cost ---

# class PerformanceMetric(BaseMetric):
#     """Custom metric to calculate and evaluate latency and mock cost."""
    
#     def __init__(self, cost_per_token_usd: float = 0.000002, threshold: float = 1.0):
#         super().__init__(threshold=threshold)
#         self.cost_per_token_usd = cost_per_token_usd
#         self.latency_seconds = 0.0
#         self.mock_cost_usd = 0.0
#         self.name = "Performance"

#     def measure(self, test_case: LLMTestCase) -> TestResult:
#         """
#         Calculates Latency and Mock Cost. 
#         In production, the start_time should be captured *before* the LLM call.
#         """
#         start_time = time.time()
        
#         # NOTE: This time.sleep(0.5) MOCKS the actual LLM generation time.
#         time.sleep(0.5) 
#         self.latency_seconds = time.time() - start_time
        
#         # --- Cost Calculation ---
#         # Simple token estimation: 1 word ~ 1.3 tokens
#         token_count = len(test_case.actual_output.split()) * 1.3 
#         self.mock_cost_usd = token_count * self.cost_per_token_usd
        
#         # Performance passes if latency is below 2.0 seconds
#         success = self.latency_seconds < 2.0
        
#         return TestResult(
#             success=success,
#             score=self.latency_seconds, # Storing latency as the score
#             metric_metadata={
#                 "Latency (seconds)": round(self.latency_seconds, 4),
#                 "Mock Cost (USD)": round(self.mock_cost_usd, 6)
#             },
#             name=self.name,
#             reason="Latency and cost metrics calculated."
#         )

# --- Utility Function to Prepare Deepeval TestCase ---

# Calculates Latency and Mock Cost    
# --- Cost Calculation ---
# Simple token estimation: 1 word ~ 1.3 tokens

actual_output = ''
# score = None
# metric_metadata = None
# name = None
# reason = None
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


def load_data_prepare_test_case(chat_json_path: str, context_json_path: str, user_turn , ai_turn) -> LLMTestCase:
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
        return LLMTestCase(input=user_query, actual_output=ai_response, retrieval_context=context_vectors,)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None

# def prepare_test_case(chat_data: Dict, context_data: Dict, target_turn: int) -> LLMTestCase:
#     """Extracts data from the JSON structures and creates a Deepeval TestCase."""
    
#     user_query = ""
#     ai_response = ""
#     for turn in chat_data.get('conversation_turns', []):
#         # User query is target_turn - 1
#         if turn.get('turn') == target_turn - 1 and turn.get('role') == 'User':
#             user_query = turn['message']
#         # AI response is target_turn
#         elif turn.get('turn') == target_turn and turn.get('role') == 'AI/Chatbot':
#             ai_response = turn['message']
#             if "evaluation_note" in turn and ai_response.endswith(turn.get('evaluation_note', '')):
#                  ai_response = ai_response.replace(turn['evaluation_note'], "").strip()

#     if not user_query or not ai_response:
#         raise ValueError(f"Could not find turn {target_turn} or its preceding query.")

#     # Retrieval Context is the data fetched from the Vector Database
#     retrieval_context = [
#         v['text'] 
#         for v in context_data.get('data', {}).get('vector_data', []) 
#         if 'text' in v
#     ]

#     return LLMTestCase(
#         input=user_query,
#         actual_output=ai_response,
#         retrieval_context=retrieval_context,
#     )