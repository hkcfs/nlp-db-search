import os
import mysql.connector
from flask import Flask, request, render_template, redirect, url_for
import ollama
import json
import logging
import time
import re

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
# User requested qwen3 1B. phi3:mini is a good small alternative.
# If you have qwen models pulled, you can use:
# OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2:0.5b') # or 'qwen:0.5b'
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:0.6b')


# --- Database Connection ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'db'), # 'db' is the service name in docker-compose
            user=os.getenv('MYSQL_USER', 'jobuser'),
            password=os.getenv('MYSQL_PASSWORD', 'jobpassword'),
            database=os.getenv('MYSQL_DB', 'jobdb'),
            port=int(os.getenv('MYSQL_PORT', 3306)) # Ensure port is an integer
        )
        if connection.is_connected():
            logging.info("Successfully connected to MySQL database")
            return connection
    except mysql.connector.Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        return None
    except Exception as ex:
        logging.error(f"A non-MySQL error occurred during DB connection: {ex}")
        return None

# --- Ollama Interaction ---
try:
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    # Test connection by listing models
    ollama_client.list()
    logging.info(f"Successfully connected to Ollama at {OLLAMA_BASE_URL} using model {OLLAMA_MODEL}")
except Exception as e:
    logging.error(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}: {e}")
    ollama_client = None # Set to None if connection fails

def get_skill_variations_from_ollama(user_query):
    if not ollama_client:
        logging.warning("Ollama client not initialized. Falling back to basic skill extraction.")
        return list(set([word.strip(".,!?:;'\"") for word in user_query.lower().split() if len(word) > 2]))

    prompt = f"""
    Job Requirement: "{user_query}"
    Identify key technical skills from the requirement. List common variations for each.
    Examples:
    - Python: python, Python, python3, py
    - HTML: html, HTML, html5, html 5
    - JavaScript: JavaScript, javascript, js, ES6
    - SQL: SQL, sql, MySQL, PostgreSQL
    - AWS: AWS, Amazon Web Services, EC2, S3

    Your entire response MUST be ONLY a comma-separated list of these skill keywords and their variations.
    DO NOT include any other text, explanations, introductory phrases, labels, XML-like tags (e.g. <think>), or your thought process.
    For "HTML expert", the response should be: html,HTML,html5,html 5,HTML5

    Based on the Job Requirement "{user_query}", provide ONLY the comma-separated skills list:
    """
    try:
        logging.info(f"Querying Ollama ({OLLAMA_MODEL}) for skill variations for query: '{user_query}'")
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 80} # Temp 0, limit tokens
        )
        raw_skills_text = response['response'].strip()
        logging.info(f"Ollama raw skills response: {raw_skills_text}")

        lines = raw_skills_text.splitlines()
        candidate_list_str = ""

        for i in range(len(lines) -1, -1, -1): # Iterate backwards to find the last good list
            line = lines[i].strip()
            looks_like_list = (',' in line) or (len(line.split()) <= 4 and len(line) > 1) # Allow slightly longer single "skills" if no comma
            is_not_convo = not line.lower().startswith(("okay", "sure", "here are", "the skills are", "i think", "first", "the user wants", "<think>", "based on the job", "example output", "your entire response", "my apologies"))
            is_not_prompt_echo = "job requirement" not in line.lower() and "comma-separated skills" not in line.lower()

            if looks_like_list and is_not_convo and is_not_prompt_echo:
                candidate_list_str = line
                break
        
        if not candidate_list_str:
            for line in reversed(lines):
                if line.strip():
                    candidate_list_str = line.strip()
                    break
            if not candidate_list_str:
                 logging.warning(f"Ollama response could not be parsed into a skills list. Raw: {raw_skills_text}")
                 skills_list_from_llm = []
            else:
                 skills_list_from_llm = [skill.strip(" .,!?:;'\"()[]{}<>") for skill in candidate_list_str.split(',') if skill.strip()]
        else:
            skills_list_from_llm = [skill.strip(" .,!?:;'\"()[]{}<>") for skill in candidate_list_str.split(',') if skill.strip()]
        
        logging.info(f"LLM skills after initial parse: {skills_list_from_llm}")

        # --- Additional filtering for conversational remnants ---
        temp_filtered_skills_stage1 = []
        conversational_markers = ["wait", "need", "focus", "shows that", "i will", "let me", "the core skill", "variations would be"]
        for skill_cand in skills_list_from_llm:
            skill_cand_lower = skill_cand.lower()
            is_conversational = False
            for marker in conversational_markers:
                if marker in skill_cand_lower:
                    is_conversational = True
                    break
            if len(skill_cand.split()) > 4 and is_conversational: # If it's long and has conversational markers
                logging.warning(f"Skill candidate '{skill_cand}' looks conversational, skipping.")
                continue
            
            # Remove specific problematic suffixes or prefixes if they appear due to bad splitting
            if skill_cand.endswith('.'): 
                skill_cand = skill_cand[:-1].strip()
            if skill_cand.lower().startswith("like "):
                skill_cand = skill_cand[5:].strip()
            
            temp_filtered_skills_stage1.append(skill_cand)
        # --- End of additional filtering stage 1 ---
        
        # Final filtering stage
        filtered_skills = []
        undesired_keywords = [
            "etc", "and", "or", "the", "is", "a", "for", "with", "core", "skill", "skills", "first",
            "variation", "variations", "list", "them", "version", "versions", "expert", "wait",
            "technical", "key", "identify", "mentioned", "requirement", "include", "common",
            "job", "output", "response", "comma-separated", "provide", "only", "actual",
            "example", "examples", "expert", "things", "like" # "expert" from "html expert", "things like"
        ]
        user_query_words = [word.lower() for word in user_query.split()]
        
        for skill_candidate in temp_filtered_skills_stage1: # Use the output of stage 1 filtering
            if not skill_candidate:
                continue
            
            skill_lower = skill_candidate.lower()
            
            # Skip if it's an undesired keyword 
            if skill_lower in undesired_keywords:
                logging.info(f"Skipping '{skill_candidate}' as it's an undesired keyword.")
                continue

            # Skip if it's a generic part of a multi-word query, unless it's a specific variant itself
            if len(user_query_words) > 1 and skill_lower in user_query_words:
                # Allow if it's something like "html" from "html expert" because "html" is a valid skill
                # But if skill is "expert" from "html expert", skip "expert"
                if skill_lower == "expert" and "expert" in user_query_words : # More specific check
                     logging.info(f"Skipping '{skill_candidate}' as it's a generic part of the query '{user_query}'")
                     continue

            if len(skill_candidate) < 2 or len(skill_candidate) > 25:
                if skill_candidate.lower() not in ["c#", "c++", "js", "r", "ai", "ml", "go", "ui", "ux", "5"]: # Allow common short skills and '5' for html 5
                    logging.info(f"Skipping '{skill_candidate}' due to length constraints.")
                    continue
            if skill_candidate.isdigit() and skill_candidate != "5": 
                 logging.info(f"Skipping '{skill_candidate}' as it's purely numeric (and not '5').")
                 continue
            if "example output for" in skill_lower or "my apologies" in skill_lower: 
                logging.info(f"Skipping '{skill_candidate}' as it echoes prompt or is an apology.")
                continue
            
            # Regex to check if the skill_candidate is a plausible skill (alphanumeric, may contain spaces, #, +)
            # Avoids skills that are just punctuation or odd characters.
            if not re.fullmatch(r"^[a-zA-Z0-9#+()\s.-]+$", skill_candidate):
                if skill_candidate: # only log if not empty after this check
                    logging.info(f"Skipping '{skill_candidate}' as it doesn't match plausible skill pattern.")
                continue

            filtered_skills.append(skill_candidate)
        
        normalized_query = user_query.lower()
        html_query_keywords = ["html", "frontend", "web design", "ui developer"]
        if any(kw in normalized_query for kw in html_query_keywords):
            html_variants = ["html", "html5", "html 5", "HTML", "HTML5", "HTML 5"]
            for hv in html_variants:
                if hv not in filtered_skills: 
                    filtered_skills.append(hv)
        
        final_skills = list(set(s for s in filtered_skills if s)) 
        
        if not final_skills and user_query: 
            logging.warning("LLM skill extraction and filtering yielded no skills. Using basic split of user query as fallback.")
            final_skills = list(set([
                kw.strip(".,") for kw in user_query.lower().split() 
                if len(kw.strip(".,")) > 1 and kw.lower() not in undesired_keywords
            ]))
        
        logging.info(f"Processed skill variations after ALL cleaning: {final_skills}")
        return final_skills
    except Exception as e:
        logging.error(f"Error during skill variations from Ollama: {e}", exc_info=True)
        simple_skills = [word.strip(".,!?:;'\"") for word in user_query.lower().split() if len(word) > 2]
        logging.warning(f"Falling back to simple keyword extraction due to error: {simple_skills}")
        return list(set(simple_skills))

def get_pros_cons_from_ollama(candidate_document_text, required_skills_list):
    if not ollama_client:
        logging.warning("Ollama client not initialized. Skipping pros/cons generation.")
        return {"pros": ["LLM analysis unavailable."], "cons": ["LLM analysis unavailable."]}

    if not candidate_document_text:
        return {"pros": ["N/A - No candidate document (resume/cover letter) provided"], 
                "cons": ["N/A - No candidate document (resume/cover letter) provided"]}
    
    max_len = 1800 # Max characters for the snippet
    doc_snippet = (candidate_document_text[:max_len] + "...") if len(candidate_document_text) > max_len else candidate_document_text
    
    required_skills_str = ", ".join(required_skills_list) if required_skills_list else "general technical qualifications"

    prompt = f"""
    You are an expert HR analyst. Analyze the following candidate's document snippet.
    The role requires skills such as: {required_skills_str}.
    Candidate Document Snippet: "{doc_snippet}"

    Based SOLELY on the provided document snippet and its relevance to the required skills, list:
    1. Two to three potential PROS of this candidate for such a role. (e.g., "Demonstrates experience with [Skill X]", "Mentions project relevant to [Skill Y]")
    2. Two to three potential CONS or areas where information is lacking for such a role. (e.g., "Limited details on [Skill Z]", "Experience with [Skill A] not explicitly stated")

    Format your response STRICTLY as a JSON object with two keys: "pros" (a list of strings) and "cons" (a list of strings).
    Do not include any explanations, introductory/concluding text, or markdown code fences (like ```json) outside the JSON structure itself.
    Provide only the JSON object.

    Example:
    {{"pros": ["Shows experience with Python", "Worked on cloud platforms (AWS mentioned)"], "cons": ["Specifics of Docker usage unclear", "No mention of CI/CD tools"]}}
    """
    try:
        logging.info(f"Querying Ollama ({OLLAMA_MODEL}) for pros/cons (target skills: {required_skills_str[:100]}...).")
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            format="json", # Request JSON output directly
            options={"temperature": 0.3, "num_predict": 150} # Adjust num_predict as needed
        )
        
        response_content = response['response'].strip()
        logging.info(f"Ollama pros/cons raw response: {response_content}")

        # Attempt to parse the JSON
        # Clean potential markdown fences or other non-JSON text if format="json" didn't perfectly strip it
        json_start = response_content.find('{')
        json_end = response_content.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response_content[json_start:json_end+1]
            analysis = json.loads(json_str)
        else: # If no clear JSON object is found
            logging.error(f"Valid JSON object not found in Ollama response: {response_content}")
            return {"pros": ["Error: LLM response was not valid JSON."], "cons": [f"Content: {response_content[:100]}..."]}

        # Validate structure
        if not isinstance(analysis, dict) or "pros" not in analysis or "cons" not in analysis:
             logging.error(f"Ollama response for pros/cons is not in the expected dict format. Got: {analysis}")
             raise ValueError("Ollama response for pros/cons is not in the expected dict format with 'pros' and 'cons' keys.")
        if not isinstance(analysis.get("pros"), list) or not isinstance(analysis.get("cons"), list):
             logging.error(f"Ollama response for 'pros' or 'cons' are not lists. Got: {analysis}")
             raise ValueError("Ollama response for 'pros' or 'cons' are not actual lists.")

        logging.info(f"Ollama pros/cons parsed: {analysis}")
        return analysis
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from Ollama for pros/cons: {e}. Response was: {response_content}")
        return {"pros": ["Error parsing analysis from LLM."], "cons": [f"Could not parse LLM's response: {str(e)}. Raw: {response_content[:100]}..."]}
    except Exception as e:
        logging.error(f"Error getting pros/cons from Ollama: {e}")
        return {"pros": ["Error generating analysis via LLM."], "cons": [str(e)]}

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_candidates():
    user_query = request.form.get('query', '').strip()
    if not user_query:
        return render_template('index.html', error="Please enter a search query.")

    logging.info(f"Received search query: {user_query}")

    target_skills = []
    if ollama_client:
        try:
            target_skills = get_skill_variations_from_ollama(user_query)
            if not target_skills: 
                logging.warning("Ollama returned no specific skills, using user query split for search.")
                target_skills = [kw.strip(".,") for kw in user_query.lower().split() if len(kw.strip(".,")) > 1]
                if not target_skills:
                     return render_template('index.html', query=user_query, error="Could not extract any meaningful skills from your query. Please try rephrasing.")
        except Exception as e:
            logging.error(f"Failed to get skills from Ollama: {e}", exc_info=True)
            target_skills = [word.strip(".,!?:;'\"") for word in user_query.lower().split() if len(word) > 2]
            if not target_skills:
                return render_template('index.html', query=user_query, error=f"LLM for skill extraction error, and no keywords found from query. Error: {e}")
            logging.warning(f"Using basic keyword extraction due to LLM error: {target_skills}")
    else: 
        logging.warning("Ollama client not available. Using basic keyword extraction for search.")
        target_skills = [word.strip(".,!?:;'\"") for word in user_query.lower().split() if len(word) > 2]
        if not target_skills:
            return render_template('index.html', query=user_query, error="LLM unavailable and no keywords found from query.")

    logging.info(f"Effective skills to search for: {target_skills}")

    db_conn = get_db_connection()
    if not db_conn:
        return render_template('index.html', query=user_query, error="Database connection failed.")

    candidates_found = []
    cursor = None 

# Inside search_candidates function:
    # ... (after target_skills are finalized) ...
    try:
        cursor = db_conn.cursor(dictionary=True)

        sql_query_parts = []
        sql_params = []

        if not target_skills:
            logging.warning("No target skills to search for in database.")
        else:
            for skill_item in target_skills:
                skill_item_stripped = skill_item.strip()
                if not skill_item_stripped:
                    continue

                # Sanitize the skill item to be used in a regex pattern
                # Remove characters that are not alphanumeric, space, #, +, -
                # This is a stricter cleaning for the regex pattern itself.
                pattern_core = re.sub(r'[^\w\s#+-]', '', skill_item_stripped)
                if not pattern_core: # If cleaning removes everything
                    continue

                # Special handling for C++ (escape the '+')
                if "C++" in pattern_core: # Use "in" in case of "C++ skill"
                    pattern_core = pattern_core.replace("C++", "C\\+\\+")
                
                # Replace spaces with [[:space:]]+ for multi-word skills
                if ' ' in pattern_core:
                    pattern_core = pattern_core.replace(' ', '[[:space:]]+')


                # Construct a REGEXP pattern that looks for the skill as a whole word.
                # This pattern means:
                # (^|[^[:alnum:]_])  -- beginning of string OR not an alphanumeric/underscore character
                # (pattern_core)     -- our actual skill pattern
                # ([^[:alnum:]_]|$)  -- not an alphanumeric/underscore character OR end of string
                # This effectively creates word boundaries around our pattern_core.
                # Example: for "html", pattern is (^|[^[:alnum:]_])(html)([^[:alnum:]_]|$)
                # Example: for "html 5", pattern_core is "html[[:space:]]+5"
                # resulting pattern is (^|[^[:alnum:]_])(html[[:space:]]+5)([^[:alnum:]_]|$)
                
                # MySQL's REGEXP is case-insensitive by default if the column uses a _ci collation.
                # We'll rely on that.
                final_regexp_pattern = f"(^|[^[:alnum:]_])({pattern_core})([^[:alnum:]_]|$)"

                sql_query_parts.append("(`resumetext` REGEXP %s OR `cover_letter` REGEXP %s)")
                sql_params.append(final_regexp_pattern)
                sql_params.append(final_regexp_pattern)
                logging.info(f"Using REGEXP for skill: '{skill_item_stripped}', pattern: '{final_regexp_pattern}'")


        if not sql_query_parts:
             logging.info("No valid SQL query parts generated from skills.")
             return render_template('index.html', query=user_query, candidates=[])
        
        # Combine individual conditions with OR
        where_clause = " OR ".join(sql_query_parts)
        sql_full_query = f"""
            SELECT `id`, `full_name`, `email`, `phone`, `skills`, `resumetext`, `cover_letter`
            FROM `job_applications` 
            WHERE {where_clause}
            ORDER BY `id` DESC
            LIMIT 10; 
        """
        
        logging.info(f"FINAL SQL Query Structure: {sql_full_query.split('WHERE')[0]} WHERE ...") # Log structure
        logging.info(f"FINAL SQL Params: {tuple(sql_params)}")

        cursor.execute(sql_full_query, tuple(sql_params)) # Ensure params is tuple
        db_results = cursor.fetchall()
        logging.info(f"Found {len(db_results)} candidates from DB.")

    # ... (rest of the try-except-finally block) ...

        for candidate_row in db_results:
            candidate_data = dict(candidate_row) 
            resume_text = candidate_data.get('resumetext', '') or ""
            cover_letter_text = candidate_data.get('cover_letter', '') or ""
            
            if not resume_text.strip() and cover_letter_text.strip():
                combined_document_text = cover_letter_text.strip()
            elif resume_text.strip() and not cover_letter_text.strip():
                combined_document_text = resume_text.strip()
            elif resume_text.strip() and cover_letter_text.strip():
                combined_document_text = f"{cover_letter_text.strip()}\n\nRESUME DETAILS:\n{resume_text.strip()}"
            else: 
                if candidate_data.get('skills'):
                    combined_document_text = f"Skills listed in database: {candidate_data.get('skills')}"
                else:
                    combined_document_text = "No textual information (resume/cover letter/skills) available for analysis."

            try:
                if ollama_client:
                    candidate_data['pros_cons'] = get_pros_cons_from_ollama(
                        combined_document_text,
                        target_skills 
                    )
                else:
                    candidate_data['pros_cons'] = {"pros": ["LLM analysis unavailable."], "cons": ["LLM analysis unavailable."]}
            except Exception as e:
                logging.error(f"Error generating pros/cons for candidate {candidate_data.get('id')}: {e}", exc_info=True)
                candidate_data['analysis_error'] = str(e)
            
            candidate_data['skills_display'] = candidate_data.get('skills', 'N/A')
            if isinstance(candidate_data['skills_display'], str) and candidate_data['skills_display'].startswith('['):
                try: 
                    skill_ids = json.loads(candidate_data['skills_display'])
                    candidate_data['skills_display'] = ", ".join(skill_ids) if isinstance(skill_ids, list) else candidate_data['skills_display']
                except json.JSONDecodeError:
                    pass 

            candidates_found.append(candidate_data)
            if ollama_client: 
                time.sleep(0.2) 

    except mysql.connector.Error as e:
        logging.error(f"Database query error: {e}", exc_info=True)
        return render_template('index.html', query=user_query, error=f"Database error: {e}")
    except Exception as e_global:
        logging.error(f"An unexpected error occurred during search: {e_global}", exc_info=True)
        return render_template('index.html', query=user_query, error=f"An unexpected error occurred: {e_global}")
    finally:
        if cursor:
            cursor.close()
        if db_conn and db_conn.is_connected():
            db_conn.close()
            logging.info("MySQL connection closed.")

    return render_template('index.html', query=user_query, candidates=candidates_found)

if __name__ == '__main__':
    # This block is for local development without Docker.
    # For Docker, CMD ["flask", "run"] in Dockerfile is used.
    # Ensure Ollama is running (e.g. `ollama serve`) and the model is pulled (e.g. `ollama pull phi3:mini`).
    # Set environment variables for DB connection if not using Docker defaults:
    # export MYSQL_HOST=localhost
    # export MYSQL_USER=your_local_user
    # export MYSQL_PASSWORD=your_local_password
    # export MYSQL_DB=jobdb
    # export MYSQL_PORT=3306 # or your local MySQL port
    # export OLLAMA_BASE_URL=http://localhost:11434 # If Ollama is local
    
    # Check Ollama model and pull if necessary (example for local setup)
    if ollama_client:
        try:
            models = ollama_client.list()
            model_names = [m['name'] for m in models['models']]
            if OLLAMA_MODEL not in model_names:
                logging.warning(f"Ollama model {OLLAMA_MODEL} not found locally. Attempting to pull...")
                # This pull might take time and is better done outside the app startup in production
                # For development, this can be convenient.
                # ollama.pull(OLLAMA_MODEL) # ollama.pull is not in the client library in this way
                # For manual pull: `docker exec -it ollama_service ollama pull your_model_name`
                # or if running ollama locally: `ollama pull your_model_name`
                logging.info(f"Please ensure model {OLLAMA_MODEL} is pulled in your Ollama instance.")
        except Exception as e:
            logging.error(f"Could not check/pull Ollama model: {e}")
    else:
        logging.error("Ollama client failed to initialize. LLM features will be disabled.")

    app.run(debug=True, host='0.0.0.0', port=5000)
