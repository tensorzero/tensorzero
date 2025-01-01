import requests
import json

BASE_URL = "http://localhost:3000/v1/functions"

def retrieve_context(topic):
    response = requests.post(f"{BASE_URL}/retrieve_context", json={"topic": topic})
    if response.status_code == 200:
        return response.json().get("context", "No context found.")
    return f"Error: {response.status_code} - {response.text}"

def answer_question(question, context):
    response = requests.post(f"{BASE_URL}/answer_question", json={"question": question, "context": context})
    if response.status_code == 200:
        return response.json().get("answer", "No answer found.")
    return f"Error: {response.status_code} - {response.text}"

def main():
    with open("jeopardy_dataset.json", "r") as f:
        dataset = json.load(f)

    for item in dataset:
        topic = item["topic"]
        question = item["question"]
        
        print(f"Topic: {topic}")
        context = retrieve_context(topic)
        print(f"Context: {context}")

        answer = answer_question(question, context)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
