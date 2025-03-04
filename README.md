# Alshwin - AI-Driven Bot for BDM Project Queries

Alshwin is a personalized AI-driven bot designed to assist with queries related to BDM (Business Data Management) projects. It leverages advanced language models and document retrieval techniques to provide accurate and relevant responses to user questions.

## Features

- **Conversational Interface**: Interact with the bot through a web-based chat interface.
- **Document Retrieval**: Retrieves relevant information from a collection of PDF documents.
- **Chat History**: Maintains a history of chat interactions.
- **Source Referencing**: Provides references to the source documents for the answers given.

## Project Structure

- **app.py**: Main application file containing the Flask routes and chat logic.
- **database.py**: Handles database operations for storing and retrieving chat history.
- **read_documents.py**: Reads and processes PDF documents, and creates a vector store for document retrieval.
- **static/style.css**: Contains the CSS styles for the web interface.
- **templates/index.html**: HTML template for the chat interface.
- **requirements.txt**: Lists the Python dependencies required for the project.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/brpuneet898/bdm-project-flask.git
    cd alshwin
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    venv\Scripts\activate  
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up the database:

    Automatically set up inside the `app.py` file. 

5. Configure the API keys and Base URL:

    - Create a `config.yaml` file in the project root directory with the following content:

    ```yaml
    GROQ_API_KEY: "your_groq_api_key_here"
    GOOGLE_CLIENT_ID: "your_google_client_id_here"
    GOOGLE_CLIENT_SECRET: "your_google_client_secret_here"
    BASE_URL: "http://127.0.0.1:5000"
    ```

6. Place your PDF documents in the `documents` directory. (Already setup, if you want new PDFs, for that place them here.)

### Running the Application

1. Start the Flask application:

    ```sh
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000` to interact with the bot.

3. Log in using your Google account to access the application.

## Usage

- Type your questions in the input box and press Enter or click the send button.
- The bot will respond with answers and provide references to the source documents.
- To end the chat, type "stop".

## Example Prompts

- "What are the rubrics to submit project proposal for BDM Project?"
- "How to formulate the title for my BDM Project?"
- "What are the rubrics for submitting the final submission for my BDM project?"
- "What do we have to submit in the metadata and descriptive statistics?"
- "What is Metadata in Mid Term Submission?"

## License

This project is licensed under the IIT Madras BS Degree License. 

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Groq](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace](https://huggingface.co/)

## Contributing

Currently, we are not accepting any contributions.

## Contact

For any questions or inquiries, please contact [Puneet](21f3002005@ds.study.iitm.ac.in).