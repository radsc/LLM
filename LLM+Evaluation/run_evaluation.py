import os
import time
from dotenv import load_dotenv
from deepeval.models import OllamaModel
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from src.evaluation_metrics import performance_metrics, load_data_prepare_test_case
import src.context_rag as rag
from src.rag_retriever import get_retriever
from config import settings


os.environ["DEEPEVAL_DEBUG"] = "true"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1800"
USER_TURN = 4
AI_TURN = 5
OLLAMA_URL = settings.OLLAMA_URL
MODEL = settings.MODEL
# OLLAMA_URL="http://localhost:11434"
# MODEL="llama3:8b"
cost_per_token_usd=0.000002 # Token/Cost Rates (Very rough estimation for demonstration)
chat_json_path = "data/sample-chat-conversation-02.json"
context_json_path = "data/sample_context_vectors-02.json"


rag.build_context_rag(chat_json_path, context_json_path, USER_TURN, AI_TURN)

# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("Please set the OPENAI_API_KEY environment variable for the Judge LLM.")

# Instantiate the Ollama Model
llama3_judge = OllamaModel(
    model=MODEL,
    # model="phi3:mini",
    # model="gemma:2b",
    # timeout=600,
    base_url=OLLAMA_URL # Default Ollama URL
    # timeout=300  # Increased timeout for larger models
)

# Find the messages


# chat_data_cleaned = {
#     "conversation_turns": [
#         {"turn": 13, "role": "User", "message": "Can you tell me about the cost of accommodation/stay near your clinic?"},
#         {
#             "turn": 14,
#             "role": "AI/Chatbot",
#             "message": "We have two accommodation options near the clinic. 1. Gopal Mansion: Your best choice is Gopal Mansion, which offers an air-conditioned room with TV and bath for only Rs 800 per night. This is run as a charitable service by Mr. R K Damani. 2.Happy Home Hotel: This hotel is a 5-minute walk from the clinic and offers rooms at Rs 1400/- for a Single Room and Rs 2000/- for a Double Room.",
#         }
#     ]
# }

# context_data_cleaned = {
#     "data": {
#         "vector_data": [
#             {"text": "Happy Home Hotel... Room Charges 1400/- Single Room 2000/- Double Room. This is a 5 min walk away from the clinic."},
#             {"text": "Your best choice would be Gopal Mansion... An airconditioned room with TV and bath is only Rs 800 per night, and is run as a charitable service by the famous philanthropist, Mr R K Damani."},
#         ]
#     }
# }
# --- 2. Define Metrics ---

metrics = [
    
    # AnswerRelevancyMetric(threshold=0.5, model='gpt-4.1',async_mode=False), 
    AnswerRelevancyMetric(threshold=0.5, model=llama3_judge,async_mode=True), # Response Relevance & Completeness
    
    # FaithfulnessMetric(threshold=0.8, model='gpt-4.1',async_mode=False) 
    FaithfulnessMetric(threshold=0.8, model=llama3_judge,async_mode=True) # Hallucination / Factual Accuracy
    
    # Latency & Costs (Custom Metric)
    # PerformanceMetric(cost_per_token_usd=0.000002) 
]

# --- 3. Main Execution ---

if __name__ == '__main__':
    print(f"--- LLM Evaluation Pipeline: Starting analysis for Turn {USER_TURN} ---")

    try:
        # Prepare the data and test case
        # data = load_data_prepare_test_case("sample-chat-conversation-02.json", "sample_context_vectors-02.json", USER_TURN, AI_TURN)
        # chat_history = data["chat_history"]["conversation_turns"]
        # context_data = data["context_vectors"]["data"]["vector_data"]

        # for turn in chat_history:
        #     if turn['turn'] == USER_TURN and turn['role'] == 'User':
        #         user_query = turn['message']
        #     elif turn['turn'] == AI_TURN and turn['role'] == 'AI/Chatbot':
        #         ai_response = turn['message']
            
        #     # Optimization: stop searching once both are found
        #     if user_query is not None and ai_response is not None:
        #         break

        # context_vectors = [v['text'] for v in context_data if 'text' in v]   
        # print("user_query", user_query)
        # print("ai_response", ai_response)
        # # print("context_data", context_data)
        # print("context_vectors", context_vectors)

        # test_case = LLMTestCase(input=user_query, actual_output=ai_response, retrieval_context=context_vectors,)

        # data = load_data_prepare_test_case(chat_json_path, context_json_path, USER_TURN, AI_TURN)
        user_query = rag.user_query
        ai_response = rag.ai_response
        print("rag user_query", user_query)
        print("rag ai_response", ai_response)
        retriever = get_retriever()
        docs = retriever.invoke(user_query)
        context_list = [doc.page_content for doc in docs]
        print("Retrieved Context:\n", context_list)

        test_case=LLMTestCase(
        input=user_query,
        actual_output=ai_response,
        retrieval_context=context_list,
        )

        
        start_time = time.time()
        evaluation_results = evaluate(test_cases=[test_case], metrics=metrics) # Run the tests
        latency_seconds = time.time() - start_time

        test_result_metrics = performance_metrics(latency_seconds, cost_per_token_usd)

        # Print the structured report
        print("\n\n#################################################")
        print(f"## FINAL EVALUATION REPORT (Turn {USER_TURN}) ##")
        print("#################################################")
        print(f"\nUser Query: {test_case.input}")
        print(f"AI Response: {test_case.actual_output}\n")
            
        if test_result_metrics['name'] == "Performance":
            print(f"Latency: {test_result_metrics['metric_metadata']['Latency (seconds)']}s")
            print(f"Cost: ${test_result_metrics['metric_metadata']['Mock Cost (USD)']}")
        # 1. Access the list of TestResult objects
        test_results_list = evaluation_results.test_results 

        # 2. Iterate through each TestResult (for all test cases run)
        for test_result in test_results_list:
            
            # 3. Access the metrics_data attribute, which is a list of MetricData objects
            metrics_data = test_result.metrics_data
            
            # 4. Iterate through each MetricData object to get the individual metric results
            for metric_data in metrics_data:
                
                # Retrieve the required attributes directly from the MetricData object
                metric_name = metric_data.name
                metric_score = metric_data.score
                metric_success = metric_data.success
                metric_reason = metric_data.reason
                # Print the formatted result
                print("-----------------------------------------")
                print(f"--- Metric: {metric_name} ---")
                print(f"Score: {round(metric_score, 4)}")
                print(f"Status: {'✅ PASS' if metric_success else '❌ FAIL'}")
                print(f"Justification: {metric_reason}")
                print(f"Model Used: {metric_data.evaluation_model}") # Optional: Useful info
                
            print("-----------------------------------------")    

        print("\n#################################################")
        print("## Execution Complete ###########################")
        print("#################################################")

    except Exception as e:
        print(f"\n❌ ERROR: Evaluation failed. Error: {e}")