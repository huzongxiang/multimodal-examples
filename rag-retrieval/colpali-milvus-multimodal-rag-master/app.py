import gradio as gr
import tempfile
import os
import fitz  # PyMuPDF
import uuid
import socket
from pathlib import Path
from dotenv import load_dotenv


from middleware import Middleware
from rag import Rag

load_dotenv(Path(__file__).with_name(".env"))

rag = Rag()


def find_available_port(start_port: int, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port

    raise OSError(
        f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}"
    )

def generate_uuid(state):
    if state is None:
        return str(uuid.uuid4())

    return state


class PDFSearchApp:
    def __init__(self):
        self.indexed_docs = {}
        self.current_pdf = None
    
        
    def upload_and_convert(self, state, file, max_pages):
        id = generate_uuid(state)

        if file is None:
            return "No file uploaded"

        print(f"Uploading file: {file.name}, id: {id}")
            
        try:
            self.current_pdf = file.name

            middleware = Middleware(id, create_collection=True)

            pages = middleware.index(pdf_path=file.name, id=id, max_pages=max_pages)

            self.indexed_docs[id] = True
            
            return f"Uploaded and extracted {len(pages)} pages", id
        except Exception as e:
            return f"Error processing PDF: {str(e)}", state
    
    
    def search_documents(self, state, query, num_results=1):
        print(f"Searching for query: {query}")
        id = generate_uuid(state)
        
        if not self.indexed_docs.get(id):
            print("Please index documents first")
            return "Please index documents first", "--"
        if not query:
            print("Please enter a search query")
            return "Please enter a search query", "--"
            
        try:

            middleware = Middleware(id, create_collection=False)
            
            search_results = middleware.search([query])[0]

            page_num = search_results[0][1] + 1

            print(f"Retrieved page number: {page_num}")

            img_path = f"pages/{id}/page_{page_num}.png"

            print(f"Retrieved image path: {img_path}")

            rag_response = rag.get_answer_from_glm(query, [img_path])

            return img_path, rag_response
            
        except Exception as e:
            return f"Error during search: {str(e)}", "--"

def create_ui():
    app = PDFSearchApp()
    
    with gr.Blocks() as demo:
        state = gr.State(value=None)

        gr.Markdown("# Colpali Milvus Multimodal RAG Demo")
        gr.Markdown("This demo showcases how to use [Colpali](https://github.com/illuin-tech/colpali) embeddings with [Milvus](https://milvus.io/) and utilizing GLM multimodal RAG for pdf search and Q&A.")
        
        with gr.Tab("Upload PDF"):
            with gr.Column():
                file_input = gr.File(label="Upload PDF")
                
                max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=10,
                    label="Max pages to extract and index"
                )
                
                status = gr.Textbox(label="Indexing Status", interactive=False)
        
        with gr.Tab("Query"):
            with gr.Column():
                query_input = gr.Textbox(label="Enter query")
                # num_results = gr.Slider(
                #     minimum=1,
                #     maximum=10,
                #     value=5,
                #     step=1,
                #     label="Number of results"
                # )
                search_btn = gr.Button("Query")
                llm_answer = gr.Textbox(label="RAG Response", interactive=False)
                images = gr.Image(label="Top page matching query")
        
        # Event handlers
        file_input.change(
            fn=app.upload_and_convert,
            inputs=[state, file_input, max_pages_input],
            outputs=[status, state],
            api_name=False,
            show_api=False,
        )
        
        search_btn.click(
            fn=app.search_documents,
            inputs=[state, query_input],
            outputs=[images, llm_answer],
            api_name=False,
            show_api=False,
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    default_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_kwargs = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        "server_port": find_available_port(default_port),
        "share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
    }

    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise

        print("Localhost launch is not accessible. Retrying with share=True.")
        launch_kwargs["share"] = True
        demo.launch(**launch_kwargs)
