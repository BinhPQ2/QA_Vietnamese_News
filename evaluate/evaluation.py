import json
from src.api.controllers.controller import *

def evaluate_pipeline(question):
    question_embedding = embedding_text(question)
    list_id, list_url = retrieval_context(question_embedding,3)
    print("list_id", list_id)
    print("list_url", list_url)
    context = mapping_data(list_id,list_url)
    print("context", context)
    result, _ = chatbot(question,context)
    return result

def evaluate_model(data_file):
    with open(data_file, "r") as file:
        dataset = [json.loads(line) for line in file]
        
    results = []
    
    # Iterate through each entry in the dataset
    for entry in dataset:
        text = entry["text"]
        questions_answers = entry["qa"]
        
        for qa in questions_answers:
            question = qa["question"]
            expected_answers = qa["answers"]

            model_response = evaluate_pipeline(question)  
            
            # Store the results for evaluation
            result_entry = {
                "question": question,
                "expected_answers": expected_answers,
                "model_response": model_response,
            }
            results.append(result_entry)
    
    return results

data_file = "evaluate/data/evaluation_data.txt"

# Call the evaluate_model function
evaluation_results = evaluate_model(data_file)

# Print the results
for result in evaluation_results:
    print(f"Question: {result['question']}")
    print(f"Expected Answers: {result['expected_answers']}")
    print(f"Model Response: {result['model_response']}")
    print("="*50)
