from flask import Flask, render_template, request, jsonify, send_from_directory
from database import init_db, insert_chat, chat_history
from read_documents import read_and_split_pdfs
import yaml
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from urllib.parse import quote
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
init_db()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
model = None


def load_model():
    global model
    if model is None:
        model = ChatGroq(
            temperature=0.8,
            model="llama-3.3-70b-versatile",
            groq_api_key=config["GROQ_API_KEY"],
            streaming=True,
        )
    return model

def load_vector_store(vector_store_path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    
documents_dir = os.path.join(os.path.dirname(__file__), "documents")
vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
# vector_store, _, _, _, _, _ = read_and_split_pdfs(documents_dir, vector_store_path)
vector_store = load_vector_store(vector_store_path)
model = load_model()

SYSTEM_PROMPT = """
You are Alswhin, a bot designed to help the student with their queries related
to Business Data Management (BDM) project. You use advanced language models and
document retrieval techniques to provide accurate and relevant responses to
user questions. For irrelevant questions, answer them by telling them you don't
know. For relevant questions, give the best possible answer."""

system_message = SystemMessage(content=SYSTEM_PROMPT)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    model, retriever=vector_store.as_retriever(), return_source_documents=True
)
chat_history = chat_history()


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/documents/<filename>')
def serve_pdf(filename):
    documents_dir = 'documents' 
    return send_from_directory(documents_dir, filename)


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if user_input.lower() == "stop":
        return jsonify({"response": "Exiting chat...", "stop": True})

    limited_chat_history_tuples = chat_history[-int(request.args.get('history', '5')):]

    messages = [system_message, HumanMessage(content=user_input)]

    response = retrieval_chain.invoke(
        {
            "question": user_input,
            "chat_history": limited_chat_history_tuples,
            "messages": messages,
        }
    )

    answer = response["answer"]
    source_documents = response["source_documents"]
    sources = {}
    for doc in source_documents:
        source = doc.metadata["source"]
        # page = doc.metadata["page"]
        page = doc.metadata.get("page", 1)
        if source not in sources:
            sources[source] = set()
        sources[source].add(page)
    source_info = []
    for source, pages in sources.items():
        sorted_pages = sorted(pages)
        page_str = ", ".join(map(str, sorted_pages))

        link = f"{config['BASE_URL']}/documents/{quote(source)}"
        source_info.append(
            f"{len(source_info) + 1}. [{source}]({link}), Page Number: {page_str}"
        )
    answer_with_sources = f"{answer}\n\n**Sources:**\n" + "\n".join(source_info)
    insert_chat(user_input, answer_with_sources)
    return jsonify({"response": answer_with_sources, "stop": False})


if __name__ == "__main__":
    documents_dir = os.path.join(os.path.dirname(__file__), "documents")
    vector_store_path = os.path.join(os.path.dirname(__file__), "vector_store")
    # vector_store, total_characters, total_pdfs, total_csvs, split_documents, total_splits = (
    #     read_and_split_pdfs(documents_dir, vector_store_path)
    # )
    # print(f"Total number of PDFs present: {total_pdfs}")
    # print(f"Total number of CSVs present: {total_csvs}")
    # print(f"Total number of split documents: {len(split_documents)}")
    # print(f"Total number of split documents stored: {total_splits}")
    app.run(debug=False)
