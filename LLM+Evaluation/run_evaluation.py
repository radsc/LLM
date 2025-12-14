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
cost_per_token_usd=0.000002 # Token/Cost Rates (Very rough estimation for demonstration)
chat_json_path = "data/sample-chat-conversation-02.json"
context_json_path = "data/sample_context_vectors-02.json"

# Build RAG for the specified turn
rag.build_context_rag(chat_json_path, context_json_path, USER_TURN, AI_TURN)

# Instantiate the Ollama Model
llama3_judge = OllamaModel(
    model=MODEL,
    base_url=OLLAMA_URL # Default Ollama URL
)

# Define Metrics
metrics = [
    AnswerRelevancyMetric(threshold=0.5, model=llama3_judge,async_mode=True), # Response Relevance & Completeness
    FaithfulnessMetric(threshold=0.8, model=llama3_judge,async_mode=True) # Hallucination / Factual Accuracy
]

# Main Execution
if __name__ == '__main__':
    print(f"--- LLM Evaluation Pipeline: Starting analysis for Turn {USER_TURN} ---")

    try:
        user_query = rag.user_query # The user query for the specified turn
        ai_response = rag.ai_response # The AI response for the specified turn
        retriever = get_retriever() # Initialize the RAG retriever
        docs = retriever.invoke(user_query) # Retrieve context documents
        context_list = [doc.page_content for doc in docs] # Extract text content from documents
        
        # Prepare the test case
        test_case=LLMTestCase(
        input=user_query,
        actual_output=ai_response,
        retrieval_context=context_list,
        )

        # Calculate latency in seconds for the evaluation
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
        # Access the list of TestResult objects
        test_results_list = evaluation_results.test_results 

        # Iterate through each TestResult (for all test cases run)
        for test_result in test_results_list:
            
            # Access the metrics_data attribute, which is a list of MetricData objects
            metrics_data = test_result.metrics_data
            
            # Iterate through each MetricData object to get the individual metric results
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