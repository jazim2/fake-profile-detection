import os

# This is a placeholder for a real LLM call.
# In a real scenario, you would use OpenAI API, Anthropic API, or a local LLM via Ollama/LM Studio.

def explain_verdict(profile_data, prediction, probability):
    """
    Generates a natural language explanation for the verdict.
    """
    verdict_text = "FAKE" if prediction == 1 else "REAL"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    # Simple rule-based logic to mimic LLM reasoning for the demo
    reasons = []
    
    if profile_data['profile_pic'] == 0:
        reasons.append("the profile lacks a profile picture")
    
    if profile_data['num_posts'] == 0 and profile_data['num_following'] > 100:
        reasons.append("creates zero content but follows many users (typical bot behavior)")
    
    if profile_data['num_followers'] < 10 and profile_data['num_following'] > 500:
        reasons.append("has a very high following-to-follower ratio")
        
    if profile_data['ratio_numlen_username'] > 0.5: # Assuming this feature exists or similar
        reasons.append("the username contains a high ratio of numbers")
        
    # Construct the explanation
    if not reasons:
        if verdict_text == "FAKE":
            explanation = f"The model is {confidence:.0%} confident this is a FAKE account, likely due to patterns in metadata not visible to the naked eye."
        else:
            explanation = f"The model is {confidence:.0%} confident this is a REAL account as it exhibits normal user behavior."
    else:
        reason_str = ", ".join(reasons)
        explanation = f"The model flagged this profile as **{verdict_text}** ({confidence:.0%} confidence). " \
                      f"Key indicators include: {reason_str}."

    return explanation

# NOTE: To use a real LLM (e.g., OpenAI), you would replace the above with:
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# prompt = f"Analyze this profile metadata: {profile_data}. Verdict: {verdict_text}. Explain why."
# response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
# return response.choices[0].message.content
