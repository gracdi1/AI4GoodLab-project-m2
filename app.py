# app.py
from flask import Flask, request, jsonify, render_template, Response
import os
import io
from PIL import Image
import base64
import cv2
import time
import numpy as np
import re
import requests
import shutil
'''
    Flask: web framework in Python
        request: access data sent in an HTTP request
        jsonify: Python objects -> JSON (send to front end)
        r_t: renders HTML templates
        Response: customizes what HTTP response from Flask server is sent to the client
    os: access to OS functions
    io: lets you treat in-memory data as file-like objects
    Pillow: lets you open/manipulate/save image files
    base64: encode/decode data in Base64 (turn binary data into plain text)
    OpenCV: library for computer vision (read/write, apply filters, use webcam, detect objs)
    time: use time-based functions (sleep (pause), time (get current time))
    np: NumPy library for numerical computing in Python
'''

# Load environment variables from .env file (if using dotenv)
from dotenv import load_dotenv
load_dotenv() # Call this at the very beginning

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Google Generative AI imports
import google.generativeai as genai
'''
    prepare the app to use Google's GenAi models, and load environment variables from .env
    allows you to securely manage sensitive data like API keys without hardcoding 
'''

# LangChain imports (for RAG)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI # For text-only initial recommendations
from langchain_google_genai import ChatGoogleGenerativeAI # Often better for multimodal and conversational
# For processing frames with Vision model
from langchain_core.messages import HumanMessage
'''
    this block is needed to build a RAG system with Gemini inside a Flask app
        LangChain for RAG (Retrieval-Augmented Generation): connect LLMs to external data (PDFs, dbs, docs) + querying that data
    Chroma: vector store; save and search docs based on semantic embedding
    PyPDFLoader: load PDF file and split it into pages or chunks as docs so LC can process
    RecCharTSp: splits long text into smaller overlapping chunks (recursively splitting)
    RetrievalQA: user question -> search relevant docs -> LLM answers using those docs
    GoogGenAI: instantiates text-only Google LLM 
    ChatGoogGenAI: chat-based Google LLM 
    HumanMessage: used when interacting with chat model (how you send a message from user into the chat)
        AIMessage is used for the model's response
        we are capturing video frames (and encoding them) -> sending them to a Vision model (Gemini)
        instead of the human chatting, it has a VIDEO response
'''

app = Flask(__name__) # create Flask app instance, which will be used later to define routes
'''
    What is Flask? a Python web framework that lets you turn Python code into a web app
        API (Application Programming Interface): a way for different softwar programs to talk to each other
            the rules, formats, and endpoints that different programs use
            API client: an individual software program
            our Flask app's API endpoint: a special URL route
        Recap of HTTP methods: GET (request), POST (send), PUT (update), DELETE (remove)
    What does Flask do? runs a web server, maps URLs to functions, sends back responses
        overall: builds routes, handle requests, and return responses
    What is a Flask app instance? a Flask application object (called app)
        acts as the central registry for your routes
        keeps track of configuration settings + starts the server when you run it
'''

# --- Configure Google Gemini API Key ---
if os.getenv("GOOGLE_API_KEY") is None:
    print("Warning: GOOGLE_API_KEY environment variable not set. Please set it. Exiting.")
    exit() # Exit if API key is not set, as it's critical.

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
'''
    A Google Gemini API key: a secret token that Google gives you to access their API services
    must be configured in order for our program to communicate with Google's AI models     
'''

# --- Global Variables for Exercise State ---
prosthetic_types = ["transtibial", "transfemoral", "no-prosthetic"]
current_exercise_steps = []
current_exercise_index = 0
vlm_model = None # To be initialized with gemini-pro-vision
llm_to_vlm = None
global_mode = None
'''
    initialize variables that will be used for the step-by-step instructions feature
'''

# --- RAG Setup (Load PDFs and Create Vector Store) ---
vectorstore = None # store vector database (from contents of PDFs) + search for info
llm = None # store LLM (text-only Gemini model)
current_retriever = None

def setup_llm_and_rag(pdf_file_path=None, mode='default_doc'):
    global vectorstore, llm, vlm_model, global_mode
    
    if mode == 'default_doc':
        # Initialize Google Gemini LLM (for text-based recommendation)
        try:
            llm = GoogleGenerativeAI(model="gemini-2.0-flash")
            print("Google Gemini LLM for text initialized successfully.")
        except Exception as e:
            print(f"Error initializing text LLM: {e}")
            return False

        # Initialize Google Gemini VLM (for vision-based analysis)
        try:
            vlm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            print("Google Gemini VLM initialized successfully.")
        except Exception as e:
            print(f"Error initializing VLM: {e}")
            return False

    # Load your PDF files
    documents = []
    
    if mode == 'uploaded_pdf': # if user uploads a pdf
        global_mode = mode
        if pdf_file_path and os.path.exists(pdf_file_path): # use pdf instead
            #print(f"Loading uploaded PDF: {pdf_file_path}")
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["category"] = mode
        else: # error
            print("No valid PDF path provided.")
            return False

    elif mode == 'default_doc': # if user lists exercises but no pdf
        global_mode = mode
        for prosthetic_type in prosthetic_types:
            #TODO: CHANGE HERE for no-prosthetic
            pdf_folder = os.path.join("data", prosthetic_type)
            print("Using pdf_folder path:", pdf_folder)
            pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
            
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path):
                    try:
                        loader = PyPDFLoader(pdf_path)
                        loaded_docs = loader.load()
                        
                        # set category metadata for each loaded doc
                        for doc in loaded_docs:
                            doc.metadata["category"] = mode + prosthetic_type

                        documents.extend(loaded_docs)
                        #print(f"Path: \n {pdf_path}")
                        #print(len(documents))
                    
                    except Exception as e:
                        print(f"Failed to load {pdf_path}: {e}")
                else:
                    print(f"Warning: PDF file not found at {pdf_path}")

    else:
        print("Unknown mode or missing PDF.")
        return False
    
    ''' 
        loads booklet as a list of pages 
        each page is a separate Document obj
    '''
    
    # split and embed documents based on mode
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    texts = text_splitter.split_documents(documents)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(texts, embeddings) 
        print(f"Stored {len(texts)} chunks in mode '{mode}'")
        return True
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return False

# upload PDF endpoint and run RAG setup (loads pDF, embeds docs, initialize LLMs)
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['pdf_file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    try:
        save_path = os.path.join("uploads", file.filename)
        file.save(save_path)

        success = setup_llm_and_rag(pdf_file_path=save_path, mode='uploaded_pdf')

        if success:
            return jsonify({
                "message": f"PDF '{file.filename}' uploaded and RAG system initialized.",
                "filename": file.filename
            }), 200
        else:
            return jsonify({"error": "Failed to initialize RAG system with uploaded PDF"}), 500

    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route('/remove_pdf', methods=['POST'])
def remove_pdf():
    global global_mode
    data = request.get_json()
    filename = data.get('filename')
    user_prosthetic = data.get('user_prosthetic')

    print("Received request to remove PDF")

    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    file_path = os.path.join("uploads", filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"PDF '{filename}' removed successfully")

            # Reset to mode = default_doc
            success = setup_llm_and_rag(mode='default_doc')
            if not success:
                return jsonify({"error": f"Uploaded File '{filename}' is removed."}), 500
            return jsonify({"message": f"PDF '{filename}' removed and reset to use database (default_mode)."})
        
        except Exception as e:
            return jsonify({"error": f"Error deleting file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File does not exist"}), 404


# display main webpage (html file from templates folder)
@app.route('/')
def index():
    # clean up uploaded files
    upload_folder = "uploadeds"
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
        os.makedirs(upload_folder)  # recreate empty folder
        
    return render_template('index.html')

# main function: POST reqs, take in user prompt, queries RAG, parses LLM output
'''
    Frontend -> POST /ask_llm -> Flask backend -> build prompt -> qa_chain (RAG)
    -> LLM -> parse response (steps + supervision) -> return JSON -> Frontend
'''
@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    global current_exercise_steps, current_exercise_index, global_mode # Use globals

    # 1. get symptoms input -> change to get exercises/type of prosthetic
    ''' 
        when a client sends a POST request to the Flask route (/ask_llm) the data 
        must be in JSON format and must contain a key called "symptoms" 
    '''
    
    user_prosthetic = request.json.get('prosthetic')
    if not user_prosthetic:
        return jsonify({"error": "No details provided"}), 400
    user_leg = request.json.get('leg')
    if ((not user_leg) and (user_prosthetic != "no-prosthetic")):
        return jsonify({"error": "No details provided"}), 400
    
    # if able bodied or prosthetic user
    if user_prosthetic == 'no-prothetic':
        user_details = user_prosthetic
    else:
        user_details = user_prosthetic + user_leg
    
    user_exercises = request.json.get('exercises')
    exercise_purpose = ['Strength', 'Mobility', 'Balance', 'Agility'] # needed?
    if not user_exercises:
        return jsonify({"error": "No exercises provided"}), 400

    # 2. build the prompt: instructs the LLM what to do 
    ''' 
        - retrieve relevant exercises froms doc based on user's provided names of exercises 
        - explain the exercise + include common errors

        still need to add: 
        - state how to position selves in front of camera
        - add feedback option (provide alternatives) ?
    '''
    try:
        prompt = (
            f"You are a virtual rehabilitation assistant "
            f"to help guide users with lower limb prosthetics through their rehabilitation exercises. "
            f"Given the following exercise(s): {user_exercises} and"
            f"the user's type and location of prosthetic: {user_details}, " 
            f"use the provided documents to find step-by-step instructions for each exercise."

            f"IMPORTANT INSTRUCTIONS:\n"
            f"1. Only provide ONE VERSION of steps for each exercise.\n"
            f"2. If multiple descriptions exist, merge them into one clean and non-redundant step-by-step list,"
            f"   avoiding duplicated or alternative step formats. Do NOT list both versions."
            f"3. ONLY IF the exercise(s) inputted by the user are NOT found, suggest the closest alternative from the documents.\n"
            f"4. For EACH exercise, include:\n"
            f"   - Prosthetic limb type(s) that use this exercise\n"
            f"   - The purpose (choose from {exercise_purpose})\n"
            f"   - commmon mistakes that individuals make when performing the exercises\n"
            f"   - Clear step-by-step instructions\n"
            f"5. If nothing relevant is found, please use general knowledge.\n"

            f"Please format your response like this:\n\n"

            f"Exercise: [Name of Exercise]\n"
            f"Prosthetic limb type(s): [e.g., transfemoral, transtibial]\n"
            f"Purpose: [e.g., Balance, Mobility]\n"
            f"Common mistakes: [Common mistakes that individuals make]\n"
            f"Steps:\n"
            f"1. [Step one]\n"
            f"2. [Step two]\n"
            f"...\n\n"

            f"IMPORTANT: only provide ONE VERSION of steps for EACH exercise."
             
            )
        #print(prompt)
        
        # 3. querry LLM chain via qa_chain (run the RAG process)
        '''
            RAG process: search the vectorstore + pass results to LLM
            returns a dictionary with "result" as key based on PDFs
        '''
        print('global mode: ',global_mode)
        if global_mode == 'uploaded_pdf': # upload pdf mode
            retriever = vectorstore.as_retriever(search_kwargs={
                "filter": {"category": global_mode}
                }
            )
        elif global_mode == 'default_doc': # if default mode then have the category = mode + prosthetic
            retriever = vectorstore.as_retriever(search_kwargs={
                "filter": {"category": global_mode + user_prosthetic}
                }
            )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": prompt})
        print("\n Result: ", result)
        exercise_recommendation_full_text = result.get("result", "Could not find a suitable exercise.")
        sources = result.get("source_documents", [])
        source_names = list({doc.metadata.get("source", "Unknown document") for doc in sources})
        print("Source PDFs used:", source_names)

        # extract using regex patterns
        exercise = re.search(r"Exercise:\s*(.*?)\s*Prosthetic", exercise_recommendation_full_text)
        prosthetic = re.search(r"Prosthetic limb type\(s\):\s*(.*?)\s*Purpose", exercise_recommendation_full_text)
        purpose = re.search(r"Purpose:\s*(.*?)\s*Common mistakes", exercise_recommendation_full_text)
        mistakes = re.search(r"Common mistakes:\s*(.*?)\s*Steps", exercise_recommendation_full_text)
        print(exercise, prosthetic, purpose, mistakes)
        
        # 4. parse response to get supervision and steps 
        ''' this parsing is highly dependent on the LLM's output format.
            you might need to fine-tune the prompt or use more sophisticated parsing.'''
        extracted_steps = []

        response_lines = exercise_recommendation_full_text.split('\n') # split into lines
        in_steps_section = False
        count = 1
        
        for line in response_lines: # look for key words (supervision, steps, etc.)
            if "steps:" in line.lower():
                in_steps_section = True
                extracted_steps.append(f"Exercise {count}:\n")
                count += 1
            elif in_steps_section and line.strip() and (line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')) or line.strip().startswith(('- ' , '* '))):
                extracted_steps.append(line.strip())
            elif in_steps_section and not line.strip(): # End of steps if empty line
                 in_steps_section = False
        
         # Fallback if steps aren't parsed well
        if not extracted_steps:
            extracted_steps = ["Could not parse specific steps. Please refer to the general description."]
            ''' as a fallback, you could try to summarize the 'Description' part if it exists
                for simplicity, we'll just put a generic message if structured steps aren't found. '''


        # 5. reset exercise state for new recommendation (save to global state)
           # -> this allows the rest of the app to know what was just recommended
        current_exercise_steps = extracted_steps
        current_exercise_index = 0

        # return JSON (exercises, steps, etc) to Frontend (to be displayed on screen)
        global llm_to_vlm
        response_data = {
            "exercise": exercise.group(1) if exercise else "",
            "prosthetic": prosthetic.group(1) if prosthetic else "",
            "purpose": purpose.group(1) if purpose else "",
            "mistake": mistakes.group(1) if mistakes else "",
            "steps": current_exercise_steps,
            "user_info": user_prosthetic,
            "sources": source_names
        }

        llm_to_vlm = response_data
        #print(response_data)
        return jsonify(llm_to_vlm)
    
    # fail safe for any error (eg: LLM API failure)
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return jsonify({"error": f"Failed to get LLM response: {str(e)}. Please check API key and model availability."}), 500

# --- New Endpoint for VLM Processing ---
@app.route('/process_frame', methods=['POST']) # endpoint design to analyze user's pose (base64-encoded image sent by front end)

# main function: accepts image + current step -> determines if user correct 
#                 -> provides feedback -> advance to next step
def process_frame():
    global current_exercise_steps, current_exercise_index, vlm_model

    # 1. check for model and exercise steps
    if not vlm_model:
        return jsonify({"error": "VLM not initialized. Cannot process frame."}), 500
    if not current_exercise_steps:
        return jsonify({"error": "No exercise steps available. Please get an exercise recommendation first."}), 400

    # 2. get image (grabs image sent from frontend, remove header, decode string into raw image bytes)
    image_data_b64 = request.json.get('image')
    if not image_data_b64:
        return jsonify({"error": "No image data provided."}), 400

    # The image data comes as 'data:image/jpeg;base64,...', so split off the header
    if ',' in image_data_b64:
        header, image_data_b64 = image_data_b64.split(',', 1)

    # 3. get current exercise step based on image + provide feedback
    try:
        image_bytes = base64.b64decode(image_data_b64)
        # In a real scenario, you might save this image temporarily or process it
        # For direct VLM use, we just need the bytes.

        # 3a. 
        current_step_text = current_exercise_steps[current_exercise_index]
        print('Curentttt', current_step_text, ("Exercise " in current_step_text))
        # input('')
        # Construct the VLM prompt
        # IMPORTANT: This prompt needs to be very specific for good results.
        # You would integrate pose estimation here to get structured data for the VLM.
        # For a basic example, we'll ask it to describe the pose and compare.
        '''
            breakdown: 
            - mix the image + text -> multimodal prompt 
            - asks model if the user image pose is correct for this step 

            change to/add: 
        '''
        prompt_parts = [
            {"mime_type": "image/jpeg", "data": image_bytes},
            f"You are an exercise coach. Analyze the person's pose in this image. "
            f"Are they correctly performing the following exercise step? "
            f"Current Step: \"{current_step_text}\".\n"
            f"Provide feedback: Is the step done correctly? If not, what adjustments are needed? "
            f"Conclude with 'Status: [Completed/Not Completed]'."
        ]

        # 3b. handle response to get model's answer 
        # Use LangChain's ChatModel for multimodal
        response = vlm_model.invoke(
            [HumanMessage(content=prompt_parts)] # send image + instructions to Gemini Vision (VLM model)
        )

        feedback = response.content # Access the content of the AI's message
        # Basic check for completion status (refine this based on actual VLM output)
        if ("status: completed" in feedback.lower()) or ("Exercise " in current_step_text):
            status = "Completed"
            current_exercise_index += 1 # Move to next step
            if current_exercise_index >= len(current_exercise_steps):
                status = "All Steps Completed!"
                current_exercise_index = len(current_exercise_steps) - 1 # Stay on last step
        else:
            status = "Not Completed"

        next_step_text = "Exercise Finished!" if current_exercise_index >= len(current_exercise_steps) else current_exercise_steps[current_exercise_index]

        # 4. send response to Frontend 
        return jsonify({
            "feedback": feedback,
            "current_step": current_step_text,
            "next_step": next_step_text,
            "status": status,
            "is_last_step": (current_exercise_index >= len(current_exercise_steps))
        })

    # 5. fail safe
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": f"Failed to process frame: {str(e)}"}), 500

# --- Webcam Streaming (for direct display, not VLM processing) ---
# This part is separate from the VLM processing.
camera = None # global variable for webcam

# 1. webcam setup + streaming function
def generate_frames():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0) # open camera, 0 for default webcam
        if not camera.isOpened():
            print("Error: Could not open webcam.")
            return

    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame) # reads frame from webcam
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # reads frame from the webcam (convert to JPEG bytes)
        time.sleep(3) # Adjust for desired frame rate (seconds b/w each frame)

# 2. flask route for video feed (sends to frontend in order to display as video)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    # multipart is used to send continuous images (stream)

# 3. stop the webcam
@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
        print("Webcam released.")
    return jsonify({"status": "Webcam stopped"})

# --- Process Uploaded Videos --- #
@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    print('Analyzing video')
    try: 

        global llm_to_vlm
        # Get JSON data from frontend
        data = request.get_json()
            
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        video_data = data.get('videoData', {})
        base64_data = video_data.get('base64')
        mime_type = video_data.get('mimeType', 'video/mp4')

        '''llm_to_vlm['steps'] = f"""Steps:
                    1. Lie on your operative side.
                    2. Lift your non-residual limb straight up, keeping your residual limb straight in line with your hip.
                    3. Relax.
                    4. Repeat."""
                    '''

        # define prompt    
        prompt = f""" 
        You are a physiotherapist reviewing a video of a person performing the exercise: "{llm_to_vlm['exercise']}".
        
        Your tasks are:
            1. Assess whether the person is correctly following these prescribed steps:
            {llm_to_vlm['steps']}. If any steps are performed incorrectly, please state which steps
            and provide corrections.
            2. Identify if the person makes any of these mistakes: {llm_to_vlm['mistake']}. 
            Make sure to identify incorrect form, posture, and positioning of the body.
            If any mistakes are made, please state which mistakes and provide corrections.
        
        Please highlight if there are any safety concerns or hazards
        
        IMPORTANT: Ensure the feedback is specific, concise, and supportive, as if you were
        coaching the user in person. 

        Please format your response like this:

        Does the user have a amputation? [Yes/No]
        What is the amputation? 
        What is the user doing?

        Steps done correctly: [Yes/No]
            Corrections: [N/A if not]
        Correct form and posture: [Yes/No]
            Corrections: [N/A if not]


        """

        # with a limb amputation
        # IMPORTANT: This person has this amputation: {llm_to_vlm['user_info']} The person may or may not be wearing a prosthetic in this video. Make sure 
        # your response is cognizant of this.


        print(prompt)

        # put into parts for gemini
        parts = [{"text": prompt}]

        # add video to prompt
        parts.append({
            "inlineData": {
                "mimeType": mime_type,
                "data": base64_data
            }
        })

        # Construct the payload for Gemini API
        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }]
        }

        # API endpoint URL
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
        
        # Make the API request to Gemini
        response = requests.post(
            api_url,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=120  # Longer timeout for video processing
        )

        print(response)
        
        # Check if request was successful
        if not response.ok:
            error_data = response.json() if response.content else {}
            return jsonify({
                'success': False,
                'error': f'Gemini API error: {response.status_code} {response.reason}',
                'details': error_data
            }), 500
        
        # Parse the response
        result = response.json()

        print('parsed')
        
        # Extract the generated text
        if (result.get('candidates') and 
            len(result['candidates']) > 0 and
            result['candidates'][0].get('content') and
            result['candidates'][0]['content'].get('parts') and
            len(result['candidates'][0]['content']['parts']) > 0):
            
            feedback = result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({
                'success': True,
                'feedback': feedback
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No meaningful response from Gemini. Please try a different prompt or video.'
            })
            
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'Request timed out. The video might be too large or complex.'
        }), 408
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f'Network error: {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    if os.path.exists("./data"):
        print("Initializing with default PDFs from /data...")
        setup_llm_and_rag(mode='default_doc')
    else:
        print("No default PDF folder found. Upload required.")
    app.run(debug=True)
