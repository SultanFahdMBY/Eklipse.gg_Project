import pandas as pd
from google import generativeai as genai
import time
import json
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize API client
genai.configure(api_key='AIzaSyC6ITA6N-dykc4xHUkbmMRSIaf9KieSH_Y') 
model = genai.GenerativeModel('gemini-1.5-flash')

# Load data
df = pd.read_csv("Game Thumbnail.csv")
DELAY_BETWEEN_REQUESTS = 2 #in second
MAX_RETRIES = 3

# Storage for all respone in the JSON format 
all_json_responses = {}

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_gemini(title):
    """Call Gemini API with combined genre, description, and player mode prompt"""
    prompt = f"""You are an expert game analyst. Analyze the game title below and respond with a strict JSON format.

GAME TITLE: {title}

RESPONSE FORMAT (JSON ONLY, NO EXPLANATION, NO MARKDOWN):
{{
  "genre": "[EXACTLY ONE from: Action, Adventure, RPG, Strategy, Simulation, Sports, Puzzle, Horror, FPS, Platformer, Fighting, Racing, MMORPG, Sandbox, Survival, BattleRoyale]",
  "description": "[Concise, 20-30 word third-person game description, active voice, no pronouns]",
  "player_mode": "[Singleplayer / Multiplayer / Both]"
}}

EXAMPLE:
{{
  "genre": "FPS",
  "description": "Tactical first-person shooter emphasizing agent abilities and precise shooting mechanics across objective-based competitive multiplayer matches.",
  "player_mode": "Multiplayer"
}}"""

    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )

    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()

    return json.loads(text)

def get_game_analysis(title):
    """Wrap API call and error handling"""
    try:
        result = call_gemini(title)
        return result
    except Exception as e:
        print(f"⚠️ Failed for {title}. Error: {e}")
        return {
            "genre": "Unknown",
            "description": f"Error: {e}",
            "player_mode": "Unknown"
        }

# Process each row with delay(sec) and collect results
for index, row in df.iterrows():
    title = row['game_title']
    print(f"\n Processing {index+1}/{len(df)}: {title}")

    start_time = time.time()
    result = get_game_analysis(title)

    # Assign to dataframe
    df.at[index, 'genre'] = result['genre']
    df.at[index, 'short_description'] = result['description']
    df.at[index, 'player_mode'] = result['player_mode']

    # Simpan hasil JSON mentah
    all_json_responses[title] = result

    elapsed = time.time() - start_time
    print(f"✅ Done in {elapsed:.2f}s | Genre: {result['genre']} | Mode: {result['player_mode']}")

    time.sleep(DELAY_BETWEEN_REQUESTS)

# Save CSV output
df.to_csv("Enhanced_Games_Combined3.csv", index=False)
print("\n✅ All data processed and saved to Enhanced_Games_Combined.csv")

# Save JSON output
Path("json_outputs").mkdir(exist_ok=True)
with open("json_outputs/gemini_game_output.json", "w", encoding='utf-8') as f:
    json.dump(all_json_responses, f, indent=2, ensure_ascii=False)

# Show sample
print("\n🔎 Sample Output:")
print(df[['game_title', 'genre', 'player_mode', 'short_description']].head().to_string(index=False))
