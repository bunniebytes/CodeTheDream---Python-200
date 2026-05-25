import json

from dotenv import load_dotenv
from openai import OpenAI

# --- Task 1: Setup and System Prompt ---

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
	response = client.chat.completions.create(
		model=model,
		messages=messages,
		temperature=temperature,
		max_completion_tokens=400
	)
	return response.choices[0].message.content

system_prompt = """
You are an experienced hiring manager who understands ATS systems and helps people pivot into software engineering roles.

Your job is to take the user's existing experience and help them present it in a way that shows how it translates to being a strong software engineer.

Keep these rules in mind:
- Stay focused only on job application materials, especially resumes
- Do not make up skills, experience, or achievements the user does not have
- Do not change their work history or exaggerate anything
- Keep the resume concise and within one page
- Optimize wording for ATS by using clear, relevant keywords from the target role (without keyword stuffing)

When helping:
- Focus on clarity, relevance, and impact
- Highlight transferable skills where appropriate

Also:
- Remind the user to review and edit everything before submitting it anywhere
- Acknowledge that you may not fully understand their original industry, so they should use their own judgment when applying your suggestions
"""

# I made the deliberate decision to ask them take on the personality as a hiring manager as those are who decides if a candidate is offered an OA/interview.  I also mentioned that they should be well-versed in ATS in case the filtering is automated.  I also broke it down to bullets and sections to help the model follow it more reliably.

# --- Task 2: Bullet Point Rewriter ---
# Write a standalone rewrite_bullets() function that takes a list of resume bullet points and returns improved versions. This function will later be called from inside the chatbot loop.

# Your function should:
	# Use delimiters to clearly separate the user's bullet points from your instructions
	# Ask for the output as a JSON list where each item has "original" and "improved" keys
	# Parse the JSON response and print both versions of each bullet side by side

# What makes these bullets weak, and what kinds of changes did the model suggest?

def rewrite_bullets(bullets: list[str]) -> list[dict]:
	# Format the bullets into a delimited block
	bullet_text = "\n".join(f"- {b}" for b in bullets)

	# Added some rules because my original output was adding percentages and statistics to the improved bullets.
	prompt = f"""
	You are a professional resume coach helping a career changer.
	Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.

	Rules:
 		- Use strong action verbs
		- Do not add any metrics, numbers, percentages, or quantified results unless explicitly provided
		- Do not invent or assume achievements, outcomes, or impact
		- Do not add new skills or responsibilities not present in the original
		- Preserve factual accuracy over sounding impressive
	
	Bad example (DO NOT DO THIS):
		Improved production by 20%

	Good example:
		Supported efforts to improve production efficiency through process optimization and collaboration.

	Original:
		Helped improve production efficiency.

	Return ONLY a valid JSON list. Each item should have two keys:
	"original" (the original bullet) and "improved" (your rewritten version).

	Bullet points:
	```
	{bullet_text}
	```
	"""

	messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
	 # Your code here: call get_completion(), parse the JSON, and return the result
	results = get_completion(messages)
 
	try:
		parsed = json.loads(results.replace("```", "").replace("json", ""))
	except Exception as e:
		print(f"Unable to Parse : {e}")
		print(results)
	
	for bullet in parsed:
		if bullet.get("original"):
			print(f'Original : {bullet["original"]}')
		if bullet.get("improved"):
			print(f'Improved : {bullet["improved"]}')

bullets = [
	"Helped customers with their problems",
	"Made reports for the management team",
	"Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# The bullets were weak because they did not really express how the user's skills apply to someone transitioning to tech.  They also were very vague in general which doesn't explain what skills they actually have, just what they did.  The imroved ones were better because they showcased skills that the user would have used to accomplish the vague statements without making anything up.

# --- Task 3: Cover Letter Generator ---
# Write a generate_cover_letter() function that takes a job title and a brief description of the user's background, and returns a cover letter opening paragraph.
# Use few-shot prompting: include at least two examples of strong cover letter openings in your prompt before asking for the new one. Your examples should demonstrate the tone and style you want — confident, specific, and not generic.
# Why did you choose those particular examples? What does the few-shot pattern help control in the output?

def generate_cover_letter(job_title: str, background: str) -> str:
	prompt = f"""
	You write strong cover letter opening paragraphs for career changers.
	The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

	Here are two examples of the style and tone you should match:

	Example 1:
	Role: Software Engineer 1 at a video streaming big tech company
	
	Background: Streamer who documents their journey in learning to program.  Self taught programmer.
	
	I am writing to express my interest for the Software Engineer position at [Company].  I believe that the potential someone has isn't reflected in a resume, and I believe my journey shows this.  As someone who transitioned into programming from a non-traditional background that streams on [Company], I deeply resonate with [Company]'s mission of “empower communities to build together”. I am a Twitch streamer and stream the process of learning to program with my community.  My experience has made me appreciate Twitch's initiatives like Creator Clubs and the Stream Together program, which foster connection and collaboration among creators.  I previously developed an interactive game with another streamer where Twitch viewers could plant, water, accessorize, and even playfully attack each other.  This project not only enhanced community engagement in chat but also allowed me to apply my programming skills in a creative and interactive way.
 
	Example 2:
	Role: Software Engineer at a collaborative, social media-focused tech company
 
	Background: Former figure skating coach and competitive figure skater, self-taught programmer

	Opening: I'm excited to apply for the Software Engineer I position at [Company]. As someone passionate about programming and continuous learning, I truly believe this role would be an incredible fit for me. [Company's] unique combination of creativity, technology, and its positive impact on people really resonates with me, and I'm eager to contribute to the team. One of the things I admire most about [Company] is how it provides a platform for people to find inspiration while staying organized. The way [Company] blends visual appeal with the ability to categorize and curate ideas is something I deeply appreciate. I love how the platform encourages creativity, allowing users to explore new ideas while keeping everything organized in a way that feels accessible and personal. 

	Now write an opening paragraph for this person:
	Role: {job_title}
	Background: {background}
	Opening:
	"""

	messages = [{"role": "user", "content": prompt}]
	# Your code here: call get_completion() and return the result
	results = get_completion(messages)
	print(f"Cover Letter :\n{results}")
	
job_title = "Junior Data Engineer"

background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."

generate_cover_letter(job_title, background)

# Using the few-shot pattern helps control the length and format of the bot's response.  It also gives a tone and voice for the bot to use.  I chose these examples as they are cover letters to specific companies I have applied to and received invites to OA.  I want the bot to keep the opening in my voice.

# --- Task 4: Moderation Check ---
# Before sending any user input to the model in your chatbot loop, run it through OpenAI's moderation endpoint first.
# Write an is_safe(text) function that:
	# Calls client.moderations.create() with model="omni-moderation-latest"
	# Returns True if the input is not flagged, False if it is
	# Prints a short, respectful message if the input is flagged, asking the user to rephrase

def is_safe(text: str) -> bool:
	result = client.moderations.create(
		model="omni-moderation-latest",
		input=text
	)
	flagged = result.results[0].flagged
	# Your code here: return True if safe, False if flagged, and print a message if flagged
	if not flagged:
		return True
	else:
		return "I can't process that request as written. Could you rephrase it?"
	
safe_input = "How do I improve my resume?"
unsafe_input = "How can I hack into someone's email account?"

print(f"Safe Test : {is_safe(safe_input)}")
print(f"Unsafe Test : {is_safe(unsafe_input)}")

# --- Task 5: The Chatbot Loop ---
# Now assemble everything into a working chatbot. Use the starter code below as your structure — your job is to fill in the marked sections.

def run_chatbot():
	# 1. Initialize conversation history with your system prompt
	messages = [
		{"role": "system", "content": system_prompt}
	]

	print("=" * 50)
	print("Job Application Helper")
	print("=" * 50)
	print("I can help you with:")
	print("  1. Rewriting resume bullet points")
	print("  2. Drafting a cover letter opening")
	print("  3. Any other questions about your application")
	print("\nType 'quit' at any time to exit.\n")

	while True:
		user_input = input("You: ").strip()

		# 2. Handle exit
		if user_input.lower() in {"quit", "exit"}:
			print("\nJob Application Helper: Good luck with your applications!")
			break

		# 3. Skip empty input
		if not user_input:
			continue

		# 4. Run moderation check before doing anything else
		if not is_safe(user_input):
			continue  # is_safe() already printed the warning message

		# 5. Check if the user wants to rewrite bullets
		#    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
		if "bullet" in user_input.lower() or "resume" in user_input.lower():
			print("\nJob Application Helper: Paste your bullet points below, one per line.")
			print("When you're done, type 'DONE' on its own line.\n")
			raw_bullets = []
			while True:
				line = input().strip()
				if line.upper() == "DONE":
					break
				if line:
					raw_bullets.append(line)
			rewrite_bullets(raw_bullets)

		# 6. Check if the user wants a cover letter
		elif "cover letter" in user_input.lower():
			job_title = input("Job Application Helper: What is the job title? ").strip()
			background = input("Job Application Helper: Briefly describe your background: ").strip()
			generate_cover_letter(job_title, background)

		# 7. Otherwise, handle it as a regular chat turn
		else:
			# Append the user's message
			messages.append({
				"role": "user",
				"content": user_input
			})

			# Get assistant response
			reply = get_completion(messages)

			# Print the reply
			print(reply)

			# Append assistant response to messages
			messages.append({
				"role": "assistant",
				"content": reply
			})
		

if __name__ == "__main__":
	run_chatbot()
 
# --- Task 6: Ethics Reflection ---
# Option A — Comment block: At the bottom of project_05.py, add a comment block responding to the questions below. Write at least 3-5 sentences total.
# Option B — Short video: Record a 2-3 minute Loom or YouTube video walking through the same questions and paste the link as a comment at the bottom of project_05.py. This can be submitted as your second LMS link.
# Respond to at least two of the following three questions:
    # Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
    # What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
    # What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)
    
# Something that could go wrong if a job-seeker submits a bot's output directly without reviewing it is that experience of the user may be embelished to the point that it is not true.  If the recruiter and hiring manager sees the resume and believe that it is true and the person is invited to an interview, it may waste everyone's time.  The bot being trained on certain text can produce biased advice as it could give off the wrong tone based on where a person is applying.  There is also more information out there based on big corporate roles that smaller skilled roles and the way the bot structures it's suggestions may not be appropriate for the lesser known roles.  The bot could provide the wrong information and misunderstand the experience due to not having a lot of information readily available.
