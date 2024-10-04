import json
import sys
# import os
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(root_dir)
# from src.api.controllers.controller import *
from controller import *

def evaluate_pipeline(question):
    rephrased_question = chatbot_rephrase(question)
    question_embedding = embedding_text(rephrased_question)
    list_id, list_url = retrieval_context(question_embedding, 3)
    context = mapping_data(list_id, list_url)
    result, _ = chatbot_answering(rephrased_question, context)
    return result

def evaluate_answer_accuracy(model_response, expected_answers):
    """Evaluate the model's response against expected answers."""
    # Convert both model_response and expected_answers to lowercase for case insensitive comparison
    model_response = model_response.lower()
    expected_answers = [answer.lower() for answer in expected_answers]

    # Check if the model's response contains any of the expected answers
    for answer in expected_answers:
        if answer in model_response:
            return 1  # Score of 1 if any expected answer is found

    return 0  # Score of 0 if no expected answer is found

def evaluate_model(data_file, limit=200):
    with open(data_file, "r") as file:
        dataset = [json.loads(line) for line in file]
        
    results = []
    total_score = 0  # To keep track of the total score
    evaluated_count = 0  # To count how many questions have been evaluated

    # Iterate through each entry in the dataset
    for entry in dataset:
        text = entry["text"]
        questions_answers = entry["qa"]
        
        for qa in questions_answers:
            if limit is not None and evaluated_count >= limit:
                break  # Stop if the limit is reached
            
            question = qa["question"]
            expected_answers = qa["answers"]

            model_response = evaluate_pipeline(question)  
            
            # Evaluate the model's response and accumulate the score
            score = evaluate_answer_accuracy(model_response, expected_answers)
            total_score += score

            # Store the results for evaluation
            result_entry = {
                "question": question,
                "expected_answers": expected_answers,
                "model_response": model_response,
                "score": score
            }
            results.append(result_entry)

            evaluated_count += 1  # Increment the evaluated count

    return results, total_score, evaluated_count

data_file = "evaluate/data/evaluation_data.txt"
limit = 10  # Set the limit for evaluation

# Call the evaluate_model function
evaluation_results, total_score, evaluated_count = evaluate_model(data_file, limit)

# Print the results
for result in evaluation_results:
    print(f"Question: {result['question']}")
    print(f"Expected Answers: {result['expected_answers']}")
    print(f"Model Response: {result['model_response']}")
    print(f"Score: {result['score']}")
    print("="*50)

# Print the total score and number of evaluated questions
print(f"Total Score: {total_score}")
print(f"Evaluated Questions: {evaluated_count}")
