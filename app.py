from flask import Flask, render_template, request, jsonify
from groq import Groq
from database import init_db, insert_chat, chat_history
from read_documents import read_and_split_pdfs
import yaml
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

app = Flask(__name__)
init_db()
# GROQ_API_KEY 
# client = Groq(api_key=api_key)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
model = None

def load_model():
    global model
    if model is None:
        model = ChatGroq(
            temperature=0.8,
            model="llama3-8b-8192",
            groq_api_key= config["GROQ_API_KEY"]
        )
    return model

documents_dir = os.path.join(os.path.dirname(__file__), "documents")
vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
vector_store, _, _, _, _ = read_and_split_pdfs(documents_dir, vector_store_path)
model = load_model()
retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever(), return_source_documents=True)
chat_history = chat_history()

def get_limited_chat_history(chat_history, limit=5):
    return chat_history[-limit:]

# def generate(state, model):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
#     response = model.chat.completions.create(
#      messages=[{"role": "user", "content": messages.text}],
#      model="llama3-8b-8192",
#      temperature=0.8,
#      stream=False)
#     return {"answer": response.choices.message.content}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if user_input.lower() == "stop":
        return jsonify({'response': 'Exiting chat...', 'stop': True})
    # chat_response = client.chat.completions.create(
    #     messages=[{"role": "user", "content": user_input}],
    #     model="llama3-8b-8192",
    #     temperature=0.8,
    #     stream=False
    # )
    # response = chat_response.choices[0].message.content

    limited_chat_history_tuples = get_limited_chat_history(chat_history, limit=5)
    response = retrieval_chain.invoke({"question": user_input, "chat_history": limited_chat_history_tuples})

    # insert_chat(user_input, response['answer'])
    # return jsonify({'response': response['answer'], 'stop': False})

    answer = response['answer']
    source_documents = response['source_documents']
    # source_info = "\n".join(
    #     f"{i+1}. {doc.metadata['source']}, Page: {doc.metadata['page']}"
    #     for i, doc in enumerate(source_documents)
    # )
    sources = {}
    for doc in source_documents:
        source = doc.metadata['source']
        page = doc.metadata['page']
        if source not in sources:
            sources[source] = set()  
        sources[source].add(page)
    source_info = []
    for source, pages in sources.items():
        sorted_pages = sorted(pages)  
        page_str = ", ".join(map(str, sorted_pages))  

        file_name = source.replace(" ", "%20")
        link = f"https://github.com/brpuneet898/bdm-project-flask/blob/main/documents/{file_name}"
        source_info.append(f"{len(source_info) + 1}. [{source}]({link}), Page Number: {page_str}")

        # source_info.append(f"{len(source_info) + 1}. {source}, Page Number: {page_str}")
    answer_with_sources = f"{answer}\n\n**Sources:**\n" + "\n".join(source_info)
    insert_chat(user_input, answer_with_sources)
    return jsonify({'response': answer_with_sources, 'stop': False})    


if __name__ == "__main__":
    documents_dir = os.path.join(os.path.dirname(__file__), "documents")
    vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
    if os.path.exists(documents_dir):
        vector_store, total_characters, total_pdfs, split_documents, total_splits = read_and_split_pdfs(documents_dir, vector_store_path)
        # print(f"Total number of characters read from all PDFs: {total_characters}") #2165673
        # print(f"Total number of PDFs present: {total_pdfs}") #32
        # print(f"Total number of split documents: {len(split_documents)}") #1256
        # print(f"Total number of split documents stored: {total_splits}") #1256
    else:
        print(f"Error: Directory '{documents_dir}' does not exist.")
    app.run(debug=False)