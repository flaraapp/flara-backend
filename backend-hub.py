import modal
import pathlib
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import json
import subprocess
import re
from typing import Optional, List
from supabase import create_client, Client
from pydantic import BaseModel
import webvtt
from pydub import AudioSegment
import google.generativeai as genai

from modal import (
    App,
    Image,
    Secret,
    asgi_app,
)

app_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "uvicorn",
        "fastapi",
        "pydantic",
        "requests",
        "supabase",
        "ffmpeg-python",
        "google-generativeai",
        "webvtt-py",
        "pydub"
    )
)

app = App(
    "flara-backend",
    image=app_image,
)

# FastAPI app
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase configuration
url: str = "https://jmlaffbdapwhgwovikel.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImptbGFmZmJkYXB3aGd3b3Zpa2VsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxOTY5Njg1OSwiZXhwIjoyMDM1MjcyODU5fQ.YoFiCmxkgM4u9XFN1Su_qxXtNsv4MIgsx-Sh7UfxIXM"
supabase: Client = create_client(url, key)

# URLs for external services
WHISPER_URL = "https://rouge606--whisperx-transcriber-fastapi-endpoint.modal.run/transcribe/"

class User(BaseModel):
    email_verified: Optional[bool] = None
    name: Optional[str] = None
    nickname: Optional[str] = None
    org_id: Optional[str] = None
    picture: Optional[str] = None
    sub: str
    updated_at: Optional[str] = None

# ---------------------- SPEECH FUNCTIONS ----------------------

@fastapi_app.post("/process_speech/")
async def process_speech(file: UploadFile = File(...), context: str = Form(...), user_sub: str = Form(...), title: Optional[str] = Form(None), isVideo: bool = Form(...)):
    user_id = user_sub
    speech_title = title

    if not await check_user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    if not await check_user_has_credits(user_id, "Speech"):
        raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        response = supabase.table("speech_reports").insert({"user_id": user_id, "pending": True, "title": speech_title}).execute() if speech_title else supabase.table("speech_reports").insert({"user_id": user_id, "pending": True}).execute()
        response_data = response.data[0]
        PRIMARY_KEY = response_data["id"]
        filename = f"{PRIMARY_KEY}_{user_id}{'.mp4' if isVideo else '.wav'}"

        temp_file_path = f"/tmp/{filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        transcription = await get_transcription(file.filename, temp_file_path, file.content_type)
        transcription_text = await vtt_to_text(transcription)
        words_per_minute = await calculate_wpm(temp_file_path, transcription_text)
        response = await get_response(transcription_text, words_per_minute, context)
        parsed_response = await parse_response(response)

        words_per_minute = float(words_per_minute)
        words_per_minute = round(words_per_minute)
        words_per_minute = int(words_per_minute)

        response = supabase.table("speech_reports").update({
            "user_id": user_id,
            "transcription": transcription,
            "rating": parsed_response.get('Overall Score'),
            "feedback": parsed_response.get('Feedback'),
            "wpm": words_per_minute,
            "pending": False
        }).eq("id", PRIMARY_KEY).execute()

        return PRIMARY_KEY

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def get_response(transcription, wpm, context):
    GEMINI_PROMPT = f""""Act as a speech and interview coach. I will provide you with transcriptions of speeches and interviews, along with the words per minute (WPM) of the speech/interview. The transcripts will likely contain filler words, and it is your job to analyze them. 

Definition of a filler word:

Filler words mostly comprise adverbs such as very, really, actually, basically, quite, honestly, literally, and seriously. Because adverbs are essential to speech, they should not be completely eliminated but used only when necessary.
Other filler words include like and just, depending on their usage since they can have more than one meaning.
Filler phrases such as I mean, I want to say that, I suppose that, you see, …or stuff like that, and so on are unnecessary and should be refrained from.
Fillers can also be sounds like “uh, um, ”"uhm," "hmm," "er," and "ah," which can be used for interaction or dramatic effect but should be limited to avoid conveying unpreparedness and boredom.

Use the following rubric to score the speeches/interviews:

Words Per Minute:

Strong (10 points): 135-165 WPM
Satisfactory (7 points): 1-20 WPM off
Needs Improvement (4 points): 20+ WPM off
Filler Words:

Strong (10 points): 0-2 filler words per minute
Satisfactory (7 points): 3-5 filler words per minute
Needs Improvement (4 points): 6+ filler words per minute
Content Quality:

Strong (10 points): Clear, engaging, well-organized, and insightful
Satisfactory (7 points): Generally clear and organized but lacks depth
Needs Improvement (4 points): Unclear, disorganized, or lacks relevance
Total Score:

Outstanding (25-30)
Competent (18-24)
Developing (12-17)
For the rubric rows "Filler Words" and "Content Quality," it is your job to detect and score the filler words and the quality of the content. Additionally, the user may provide 1-3 sentences of context surrounding the speech/interview, detailing the situation, definitions of words used, or other relevant information. After this, output the information using the following format:

Feedback: [Provide 2-4 sentences on the overall feedback of the speech or interview. Ensure the criticism is constructive, offering both praise and suggestions for improvement. Reference the filler words found, the quality of the content, and the provided words per minute. Mention the score given as well.]

Filler Words: [Provide the category grade for Filler Words from Strong, Satisfactory, Needs Improvement]
Quality of Content: [Provide the category grade for Quality of Content from Strong, Satisfactory, Needs Improvement]
Overall Score: [Provide the overall grade from Outstanding, Competent, Developing]

Do NOT output anything else other than using the above template as this will mess up the return of the scoring.

Here are the transcription, words per minute, and optional context for the speech/interview:

Transcription: {transcription}

Words Per Minute (WPM): {wpm}

Context (optional): {context}"""

    google_key = await get_key()
    genai.configure(api_key=google_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(GEMINI_PROMPT)
    return response.text

# ---------------------- INTERVIEW FUNCTIONS ----------------------

@fastapi_app.post("/process_interview/")
async def process_interview(files: List[UploadFile] = File(...), questions: List[str] = Form(...), user_sub: str = Form(...), title: Optional[str] = Form(None), isVideo: bool = Form(...), isTechnical: bool = Form(...), job_description: str = Form(...), resume: str = Form(...)):
    user_id = user_sub
    interview_title = title

    if not await check_user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    if not await check_user_has_credits(user_id, "Interview"):
        raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        response = supabase.table("interview_reports").insert({"user_id": user_id, "pending": True, "title": interview_title}).execute() if interview_title else supabase.table("interview_reports").insert({"user_id": user_id, "pending": True}).execute()
        response_data = response.data[0]
        PRIMARY_KEY = response_data["id"]

        transcriptions = []
        unformatted_transcriptions = []
        wpms = []
        file_paths = []

        # Process each file
        for count, (file, question) in enumerate(zip(files, questions), 1):
            filename = f"{PRIMARY_KEY}_{user_id}_{count}{'.mp4' if isVideo else '.wav'}"
            temp_file_path = f"/tmp/{filename}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(await file.read())
            file_paths.append(temp_file_path)

            # Get transcription and WPM
            transcription = await get_transcription(file.filename, temp_file_path, file.content_type)
            transcription_text = await vtt_to_text(transcription)
            wpm = await calculate_wpm(temp_file_path, transcription_text)
            wpm = int(wpm)

            # Format the transcription with the question
            formatted_transcription = f"Question {count}: {question}\nAnswer: {transcription_text}"

            transcriptions.append(formatted_transcription)
            unformatted_transcriptions.append(transcription)
            wpms.append(wpm)

        resume = resume.lower()
        # Determine if resume is provided
        resumeExists = resume != "nothing"

        # Get interview response from the model
        model_response = await get_interview_response(transcriptions, wpms, isTechnical, job_description, resume, resumeExists)

        # Parse the model's response
        feedback_list, score_list, sample_responses, overall_feedback, overall_score, average_wpm = await parse_interview_response(model_response, len(files))

        average_wpm = float(average_wpm)
        average_wpm = round(average_wpm)
        average_wpm = int(average_wpm)

        # Insert data into the database
        response = supabase.table("interview_reports").update({
            "user_id": user_id,
            "questions": json.dumps(questions),
            "transcriptions": json.dumps(unformatted_transcriptions),
            "wpms": json.dumps(wpms),
            "feedbacks": json.dumps(feedback_list),
            "ratings": json.dumps(score_list),
            "sample_responses": json.dumps(sample_responses),
            "overall_feedback": overall_feedback,
            "overall_rating": overall_score,
            "overall_wpm": average_wpm,
            "pending": False
        }).eq("id", PRIMARY_KEY).execute()

        return PRIMARY_KEY

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


async def get_interview_response(transcriptions, wpms, isTechnical, job_description, resume, resumeExists):
    num_questions = len(transcriptions)
    question_blocks = ""

    for i in range(num_questions):
        question_blocks += f"{transcriptions[i]}\nWPM: {wpms[i]}\n"

    GEMINI_PROMPT = f"""Act as an interview coach and grade interviews based on the transcriptions provided for each interview question and the words per minute (WPM) for each audio response. The interviews will simulate a real interview, and your role is to assess the quality of the candidate’s responses. When giving your feedback, write the response as if directly speaking to the interviewee. 

Your grading should focus on the following aspects:

Filler Words: Detect and assess the use of filler words in each response. Filler words include adverbs such as very, really, actually, basically, quite, honestly, literally, and seriously. Other filler words include like and just, depending on their usage. Filler phrases and sounds like “uh, um, ”"uhm," "hmm," "er," and "ah" should also be evaluated. These should be used sparingly to avoid conveying unpreparedness or lack of confidence.

Content Quality: Evaluate the content of each response, focusing on the relevance, clarity, and depth of the answer in relation to the interview question. The content should demonstrate a thorough understanding of the topic, clear articulation of ideas, and a structured approach to answering the question.

Technical Knowledge (Optional): If the interview is a technical interview (True), additionally grade the responses on whether they showcase sufficient technical knowledge for the field as displayed in the question. The evaluation should consider the depth and accuracy of the technical content in relation to the job description and resume (if provided).

Words Per Minute (WPM): Assess the pacing of the response based on the provided WPM. The ideal range is 135-165 WPM. Points should be deducted if the response is too fast (indicating potential nervousness or lack of clarity) or too slow (indicating hesitancy or lack of confidence).

Overall Performance: Provide an overall score for the interview based on the combined assessment of each question, including the use of filler words, content quality, technical knowledge (if applicable), and WPM. The overall score should reflect the candidate's ability to effectively communicate and answer the interview questions.

Grading Rubric:

Filler Words:

Outstanding: 0-2 filler words per minute
Competent: 3-5 filler words per minute
Developing: 6+ filler words per minute

Content Quality:

Outstanding: Clear, relevant, and insightful response that fully addresses the question
Competent: Generally clear and relevant but lacks depth or completeness
Developing: Unclear, disorganized, or fails to fully address the question

Technical Knowledge (if applicable):

Outstanding: Demonstrates deep and accurate technical knowledge relevant to the job description
Competent: Shows adequate technical understanding but lacks depth or detail
Developing: Limited or inaccurate technical knowledge, fails to address key technical aspects

Words Per Minute (WPM):

Outstanding: 135-165 WPM
Competent: 1-20 WPM off
Developing: 20+ WPM off
Feedback Instructions:

When providing feedback, follow these guidelines:

Rubric-Related Feedback: Explain how the candidate’s response aligns with or deviates from the expectations set in the rubric categories. Mention specific instances of filler words, clarity, relevance, technical knowledge (if applicable), and pacing.

Job Description and Resume Relevance: Tailor your feedback to the specific job description and resume (if provided). Highlight how well the candidate’s response reflects the skills and knowledge required for the role and how their experience or background (as detailed in the resume) is reflected in their answers. However, don't reference imaginary or blank job descriptions/experiences when the user does not provide a resume. Only reference details that exist in the resume given by the user.

Constructive Criticism: Offer balanced feedback that includes both praise for strengths and constructive suggestions for improvement. Ensure that the feedback is actionable and relevant to the candidate’s potential performance in the role.

Output Format:

Feedback:

Feedback #1: [Provide feedback for Question 1]

Feedback #2: [Provide feedback for Question 2]

...

Scores:

Score #1: [Provide the grade for Question 1 as Developing, Competent, or Outstanding]

Score #2: [Provide the grade for Question 2 as Developing, Competent, or Outstanding]

...

Sample Responses:

Sample Response #1: [Provide a sample response for Question 1]

Sample Response #2: [Provide a sample response for Question 2]

...

Overall Score: [Provide the overall score for the entire interview and categorize it as Outstanding, Competent, or Developing]

Overall Feedback: [Summarize the candidate’s performance, highlighting strengths and areas for improvement]

Average WPM: [Provide the average WPM across all questions]

Do NOT generate any output other than the feedback, scores, sample responses, overall score, overall feedback, and average WPM using the format provided.

Here are the inputs:

Job Description: {job_description}

Resume Provided (True/False): {resumeExists}

Interview Questions and Transcriptions:

{question_blocks}
Technical Interview (True/False): {"True" if isTechnical else "False"}

Resume ("Nothing" if there is no resume): {resume}"""

    google_key = await get_key()
    genai.configure(api_key=google_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(GEMINI_PROMPT)
    return response.text


async def parse_interview_response(model_response, num_questions):
    feedback_list = []
    score_list = []
    sample_responses = []
    overall_feedback = None
    overall_score = None
    average_wpm = None

    # Parse feedback
    for i in range(num_questions):
        feedback_pattern = f"Feedback #{i+1}:(.*?)(Scores|Sample Responses|Feedback #{i+2}:|Overall Score|Overall Feedback)"
        feedback_match = re.search(feedback_pattern, model_response, re.DOTALL)
        if feedback_match:
            feedback_list.append(feedback_match.group(1).strip())

    # Parse scores
    valid_scores = ["Developing", "Competent", "Outstanding"]
    for i in range(num_questions):
        score_pattern = f"Score #{i+1}:(.*?)(Sample Responses|Score #{i+2}:|Overall Score|Overall Feedback)"
        score_match = re.search(score_pattern, model_response, re.DOTALL)
        if score_match:
            score = score_match.group(1).strip()
            # Ensure only valid scores are returned
            if any(valid_score in score for valid_score in valid_scores):
                for valid_score in valid_scores:
                    if valid_score in score:
                        score_list.append(valid_score)
                        break
            else:
                score_list.append("Unknown")  # This is a fallback

    # Parse sample responses
    for i in range(num_questions):
        sample_response_pattern = f"Sample Response #{i+1}:(.*?)(Overall Score|Sample Response #{i+2}:|Overall Feedback)"
        sample_response_match = re.search(sample_response_pattern, model_response, re.DOTALL)
        if sample_response_match:
            sample_responses.append(sample_response_match.group(1).strip())

    # Parse overall feedback and score
    overall_feedback_match = re.search(r"Overall Feedback:(.*?)Average WPM:", model_response, re.DOTALL)
    if overall_feedback_match:
        overall_feedback = overall_feedback_match.group(1).strip()

    overall_score_match = re.search(r"Overall Score:(.*?)Overall Feedback:", model_response, re.DOTALL)
    if overall_score_match:
        overall_score_text = overall_score_match.group(1).strip()
        for valid_score in valid_scores:
            if valid_score in overall_score_text:
                overall_score = valid_score
                break

    average_wpm_match = re.search(r"Average WPM:(.*?)$", model_response, re.DOTALL)
    if average_wpm_match:
        average_wpm = average_wpm_match.group(1).strip()

    return feedback_list, score_list, sample_responses, overall_feedback, overall_score, average_wpm

@fastapi_app.post("/generate_questions/")
async def generate_questions(job_description: str = Form(...), numbQuestions: int = Form(...), resume: str = Form(...), isTechnical: bool = Form(...)):
    GEMINI_PROMPT = f""""Act as an interviewer and generate questions for a candidate based on the provided job description, resume, the number of questions requested, and whether the interview is a technical interview or not. The interview should be designed to test the candidate's knowledge of the necessary skills for the job as outlined in the job description.
If the interview is a technical interview (True), focus on generating questions that assess the candidate's technical ability for the role, as well as the depth and breadth of their knowledge in the field as described in the job description.
The questions should be tailored to the candidate’s resume and fall under four main topics:

1. General Interview Questions (e.g., "Tell me about yourself")
2. Background and Skills
3. Work Experience
4. Projects (if applicable)

The questions should:
- Be distinct and not repeat any previous question.
- Flow in a logical order, simulating a real interview with a human interviewer.
- Relate to the job description and the context of the candidate’s resume.

Output Formatting:
- Each question should be on the same line as "Question #:"
 -Start each question with "Question #:" where # is the question number.
 -Do not generate any output other than the interview questions.

Example Output Formatting:

Question 1: [First question here]
Question 2: [Second question here]
Question 3: [Third question here]

Here are the inputs:

Job Description: {job_description}

Number of Questions (3-10): {numbQuestions}

Candidate’s Resume: {resume}

Technical Interview (True/False): {isTechnical}"""

    # Get response from the gemini model
    google_key = await get_key()
    genai.configure(api_key=google_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(GEMINI_PROMPT)
    print(response.text)
    response = response.text

    questions = await extract_questions(response)

    return json.dumps(questions, indent=4)

# ---------------------- SHARED UTILITY FUNCTIONS ----------------------

async def get_transcription(filename, audio_file, content_type):
    headers = {'accept': 'application/json'}
    files = {'file': (filename, open(audio_file, 'rb'), content_type)}
    response = requests.post(WHISPER_URL, headers=headers, files=files)

    with open('/tmp/transcription.vtt', 'wb') as f:
        f.write(response.content)

    with open('/tmp/transcription.vtt', 'r') as f:
        transcript = f.read()

    return transcript


async def parse_response(feedback):
    feedback_dict = {}
    overall_feedback = re.search(r'Feedback:(.*?)Filler Words:', feedback, re.DOTALL)
    if overall_feedback:
        feedback_dict['Feedback'] = overall_feedback.group(1).strip()

    overall_score = re.search(r'Overall Score: (.*?)(Here are some suggestions for improvement:|$)', feedback, re.DOTALL)
    if overall_score:
        score_only = re.match(r'(Outstanding|Competent|Developing)', overall_score.group(1).strip())
        if score_only:
            feedback_dict['Overall Score'] = score_only.group(1)

    return feedback_dict


async def extract_questions(text):
    pattern = r"Question\s(\d+):\s(.+?)(?=Question\s\d+:|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    questions_dict = {}
    for match in matches:
        question_number = int(match[0])
        question_text = match[1].strip()
        questions_dict[question_number] = question_text

    return questions_dict


async def check_user_exists(user_id: str) -> bool:
    response = supabase.table("user").select("user_id").eq("user_id", user_id).execute()
    response_data = response.data[0]
    exists = response_data["user_id"]
    return bool(exists)

# Subtract credits from user based on submission type
async def subtract_user_credits(user_id: str, submission_type: str):
    # Define the credit cost based on submission type
    credit_cost = 300 if submission_type == "Interview" else 100

    # Get the current credits as a list [organization_credits, premium_credits, daily_credits]
    org_credits, premium_credits, free_credits = await get_current_credits(user_id)

    # Subtract from organization credits first
    if org_credits >= credit_cost:
        new_org_credits = org_credits - credit_cost
        supabase.table("user").update({"organization_credits": new_org_credits}).eq("user_id", user_id).execute()
        return
    else:
        credit_cost -= org_credits
        supabase.table("user").update({"organization_credits": 0}).eq("user_id", user_id).execute()

    # Subtract from premium credits next
    if premium_credits >= credit_cost:
        new_premium_credits = premium_credits - credit_cost
        supabase.table("user").update({"premium_credits": new_premium_credits}).eq("user_id", user_id).execute()
        return
    else:
        credit_cost -= premium_credits
        supabase.table("user").update({"premium_credits": 0}).eq("user_id", user_id).execute()

    # Finally, subtract from daily credits (free credits)
    if free_credits >= credit_cost:
        new_free_credits = free_credits - credit_cost
        supabase.table("user").update({"daily_credits": new_free_credits}).eq("user_id", user_id).execute()
    else:
        # This else block should not be reached due to prior credit checks
        raise ValueError("Not enough credits to subtract. This state should not occur.")

async def check_user_has_credits(user_id: str, submission_type: str) -> bool:
    # Get the current credits as a list [organization_credits, premium_credits, daily_credits]
    org_credits, premium_credits, free_credits = await get_current_credits(user_id)

    # Calculate total credits
    total_credits = org_credits + premium_credits + free_credits

    # Determine if the user has enough credits based on the submission type
    required_credits = 300 if submission_type == "Interview" else 100
    return total_credits >= required_credits

# Get all current credits as a list [organization_credits, premium_credits, daily_credits]
async def get_current_credits(user_id: str) -> List[int]:
    org_credits = await get_organization_credits(user_id)
    premium_credits = await get_premium_credits(user_id)
    daily_credits = await get_daily_credits(user_id)
    return [org_credits, premium_credits, daily_credits]

# Fetch organization credits
async def get_organization_credits(user_id: str) -> int:
    response = supabase.table("user").select("organization_credits").eq("user_id", user_id).execute()
    response_data = response.data[0]
    return response_data["organization_credits"]

# Fetch premium credits
async def get_premium_credits(user_id: str) -> int:
    response = supabase.table("user").select("premium_credits").eq("user_id", user_id).execute()
    response_data = response.data[0]
    return response_data["premium_credits"]

# Fetch daily (free) credits
async def get_daily_credits(user_id: str) -> int:
    response = supabase.table("user").select("daily_credits").eq("user_id", user_id).execute()
    response_data = response.data[0]
    return response_data["daily_credits"]

async def calculate_wpm(audio_file, transcription):
    audio_duration = await get_audio_duration(audio_file)
    word_count = await count_words_from_vtt(transcription)
    return word_count / audio_duration


async def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return float(audio.duration_seconds) / 60


async def count_words_from_vtt(vtt_content):
    text_content = await vtt_to_text(vtt_content)
    words = [word for word in text_content.split() if word]
    return len(words)


async def vtt_to_text(vtt_content):
    vtt_content = vtt_content.replace('WEBVTT', '')
    vtt_content = re.sub(r'\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}\.\d{2}', '', vtt_content)
    return re.sub(r'[^\w\s]', '', vtt_content)


async def get_key():
    return os.environ["GOOGLE_API_KEY"]


@app.function(image=app_image, secrets=[modal.Secret.from_name("GOOGLE_API_KEY")], allow_concurrent_inputs=20)
@asgi_app()
def fastapi_endpoint():
    return fastapi_app


# Local entry point for testing
@app.local_entrypoint()
def main():
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
