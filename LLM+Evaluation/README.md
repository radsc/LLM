# LLM Evaluation Pipeline (Deepeval)

This repository contains a pipeline for automatically evaluating LLM responses against core quality and performance parameters: Response Relevance, Factual Accuracy (Hallucination), Latency, and Cost.

It leverages the **Deepeval** framework for automated quality scoring via an evaluator LLM and includes a custom metric for performance tracking.

## 🚀 Setup and Installation

1.  Clone the repository
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables:**
    Create a file named `.env` in the root directory and add your OPEN AI API key. This is required for `AnswerRelevancyMetric` and `FaithfulnessMetric` to function if using 'gpt-4.1' model.

## ⚙️ Usage

The pipeline is set up to evaluate a single turn (Turn 14) from the sample data, which contains a known hallucination for demonstration.

1.  **Run the pipeline:**
    ```bash
    py run_evaluation.py
    ```

2. **Architecture of the Evaluation Pipeline Flow**
    Data Extraction: A script reads the conversation history (sample-chat-conversation-01.json) to extract test cases, pairing a User Question (input) with the immediately following AI Answer (actual_output).

    RAG System Integration (White-Box): The raw text file is first loaded into the system to create a document object. The document is then broken down into smaller, manageable text chunks (context) to ensure the retrieved piece is relevant and fits within the LLM's context window.Each text chunk is converted into a numerical vector embedding using an embedding model. These vectors are then stored in the ChromaDB vector store, allowing for fast, semantic similarity searches during the retrieval phase of the RAG application.

    Retrieval: The function queries ChromaDB with the input to fetch the top-k most relevant documents.

    Context Capture: The exact text content of these retrieved documents is captured as the retrieval_context.

    Generation: The LLM synthesizes the actual_output using the captured context.

    Test Case Construction: A LLMTestCase is constructed using the input, actual_output, and the captured retrieval_context.

    Metric Evaluation (DeepEval): The test case is passed to the evaluate() function along with the defined metrics:
        AnswerRelevancyMetric:
        FaithfulnessMetric
        PerformanceMetric(custom)

3. **Design Justification (Why This Approach?)**
    This dynamic white-box integration approach over simpler file-based or black-box methods to ensure the integrity and relevance of evaluation data.

    Holistic System Test: By making a live call to the RAG system and ChromaDB, we are testing the entire chain—from embedding and retrieval quality to final generation. This prevents false positives that occur when only testing the generation layer against manually created "ideal" context.

    True Faithfulness: The FaithfulnessMetric relies entirely on the provided retrieval_context. Capturing the exact context returned by ChromaDB during the live query ensures that the metric correctly identifies any hallucinations based on the actual, flawed or perfect, documents the LLM received.

    Debugging Efficiency: If a test case fails (e.g., low Faithfulness score), we immediately have the three components—the query, the answer, and the exact context—required to debug the vector database configuration, the embedding model, or the LLM prompt.

4. **Scaling Strategy for Real-Time Evaluation**
    When running this script at scale (millions of daily conversations), minimizing latency and cost for real-time evaluations is paramount.
    For this demo llama3:8b llm model was choosen as there were few issues while purchasing key. But for real time GPT-4 LLM is recommended.

    **A. Cost Minimization via Tiered LLM Usage**
    The primary cost driver is the use of powerful LLMs (like GPT-4) for the metrics themselves.

    Metric LLM Selection: Configure DeepEval to use more cost-effective models (e.g., GPT-3.5-turbo) as the judge for high-volume metrics like AnswerRelevancy. High-cost, high-reasoning models are reserved only for critical metrics, if needed.

    Custom Performance Metric: The PerformanceMetric is run entirely without making an external LLM API call for its calculation. It uses a simple word-to-token ratio and a fixed cost_per_token_usd to provide an immediate, accurate cost estimate based on the output length, minimizing latency overhead.

    **B. Latency Minimization through Parallelism**
    Asynchronous Evaluation: The evaluate() function in DeepEval should be executed asynchronously. By submitting test cases in large batches, we leverage parallel processing, utilizing the high concurrent request limits of the external LLM APIs and minimizing I/O waiting time.

    Batching and Sampling: Instead of attempting to evaluate all millions of conversations, implement intelligent sampling to maintain statistical relevance without incurring massive latency or cost:

    Random Sampling: Evaluate a statistically significant subset of daily conversations (e.g., 0.1%).

    Targeted Sampling: Prioritize the evaluation of conversations that show potential issues (e.g., high retrieval latency from ChromaDB, long conversation chains, or negative user feedback) for high-signal testing.

## 🧪 Evaluation Metrics Implemented

| Parameter | Deepeval Metric | Description |
| :--- | :--- | :--- |
| **Relevance & Completeness** | `AnswerRelevancyMetric` | Measures how well the AI's response answers the user's query. |
| **Hallucination / Factual Accuracy** | `FaithfulnessMetric` | Measures how much of the AI's response is factually supported by the provided `retrieval_context`. |
| **Latency & Costs** | `PerformanceMetric` (Custom) | Tracks the generation time and calculates a mock token cost based on a fixed rate. |