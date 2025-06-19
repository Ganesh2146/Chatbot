from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
import PyPDF2
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI(
    title="Chatbot API",
    description="API for the Chatbot Application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)

class GeminiChat:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        api_key = "AIzaSyCppQnAUISSEdCEZ8AT43waffzDoxqt-kk"
        if not api_key:
            raise ValueError("API key not found.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=[])
        self.pdf_content = ""  # Store PDF content here

    def get_response(self, user_input: str) -> str:
        try:
            # If we have PDF content, include it as context
            if self.pdf_content:
                prompt = f"""Here is the content of the uploaded PDF:
                {self.pdf_content[:2000]}... (truncated)
                
                Question: {user_input}"""
            else:
                prompt = user_input
            
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        self.chat = self.model.start_chat(history=[])
        self.pdf_content = ""  # Clear stored PDF content

    def extract_text_from_pdf_bytes(self, file_bytes: bytes) -> str:
        text = ""
        try:
            # Create a BytesIO object from the bytes
            pdf_stream = io.BytesIO(file_bytes)
            
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(pdf_stream)
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text()
            
            # Clean up the text
            text = text.strip()
            if not text:
                return "No text could be extracted from the PDF"
            
            # Store the PDF content
            self.pdf_content = text
            return "PDF uploaded successfully. You can now ask questions about its content."
        except Exception as e:
            return f"Failed to read PDF: {str(e)}"

chatbot = GeminiChat()

# Pydantic model for chat input
class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    """
    Process a chat message and return the bot's response.
    If a PDF has been uploaded, its content will be used as context.
    """
    try:
        if not chat_input.message or not chat_input.message.strip():
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "Message cannot be empty"}
            )
            
        response = chatbot.get_response(chat_input.message)
        return {
            "status": "success",
            "response": response,
            "has_pdf_context": bool(chatbot.pdf_content)
        }
    except Exception as e:
        error_msg = f"Error processing chat message: {str(e)}"
        print(error_msg)
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": error_msg}
        )

# Add OPTIONS endpoint for /chat
@app.options("/chat")
async def chat_options():
    return {"message": "CORS preflight request handled"}

@app.options("/clear")
async def clear_options():
    return {"message": "CORS preflight request handled"}

@app.post("/clear")
async def clear_chat():
    chatbot.clear_history()
    return {"status": "chat history cleared"}

@app.post("/detach-pdf")
async def detach_pdf():
    chatbot.clear_history()
    return {"status": "PDF detached successfully"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and extract its text content for context.
    The extracted text will be used to provide context for subsequent chat messages.
    """
    try:
        # Get file info
        filename = file.filename or 'unnamed.pdf'
        content_type = file.content_type
        
        # Validate file
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail={"status": "error", "message": "Only PDF files are allowed"}
            )
        
        # Check file size (limit to 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file_bytes = await file.read()
        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": "File too large. Maximum size is 10MB"}
            )
        
        # Reset chat history when uploading a new PDF
        chatbot.clear_history()
        
        # Extract text from PDF
        result = chatbot.extract_text_from_pdf_bytes(file_bytes)
        
        if result.startswith("Failed to read PDF") or result.startswith("No text could be extracted"):
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": result}
            )
        
        return {
            "status": "success",
            "message": result,
            "filename": filename,
            "text_length": len(chatbot.pdf_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        print(error_msg)
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": error_msg}
        )
    except HTTPException as e:
        print(f"HTTP Exception: {str(e)}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Simple check if the chatbot is responding
        test_response = chatbot.get_response("Test message")
        return {
            "status": "healthy",
            "model": "gemini-2.0-flash",
            "has_pdf_context": bool(chatbot.pdf_content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
