import pandas as pd
import numpy as np
import os
import json
import io
import csv
import time # Used to simulate the "training" time
import random # Used to simulate the forecast
from flask import Flask, request, jsonify
from flask_cors import CORS # Handles browser security
from openai import OpenAI # The LLM client

# --- 1. INITIALIZE FLASK SERVER ---
app = Flask(__name__)
CORS(app) 
print("--- Flask Server Initialized (MVP SIMULATION MODE) ---")

# --- 2. SECURELY LOAD API KEY ---
# The server will read the key from your environment variables
# For local testing, you must set this in your terminal first:
# export OPENAI_API_KEY='sk-your-real-key-goes-here'
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if os.environ.get("OPENAI_API_KEY") is None:
        raise Exception("OPENAI_API_KEY environment variable not set.")
    print("--- OpenAI Client Initialized Successfully ---")
except Exception as e:
    print(f"--- CRITICAL WARNING: OpenAI API Key not found. AI features will be disabled. Error: {e} ---")
    client = None

# --- 3. DATA PARSING FUNCTION (*** FINAL ROBUST VERSION ***) ---
def parse_user_csv(csv_text):
    """
    Robustly parses the user's uploaded CSV to validate it AND extract item names.
    This version handles missing values, flexible column names, AND bad/corrupt lines.
    """
    file = io.StringIO(csv_text)
    # Filter out any blank lines
    raw_lines = [line for line in file.readlines() if line.strip()]
    
    if len(raw_lines) < 2:
        raise ValueError("File must have a header and at least one data row.")
    
    try:
        delimiter = csv.Sniffer().sniff(raw_lines[0]).delimiter
    except csv.Error:
        delimiter = ','

    # *** FINAL ROBUSTNESS FIX ***
    # on_bad_lines='skip' tells pandas to ignore any row that is corrupt (e.g., has the wrong number of columns)
    # This fixes the error you saw with the messy CSV file.
    df = pd.read_csv(io.StringIO("\n".join(raw_lines)), delimiter=delimiter, on_bad_lines='skip')
    
    # Clean column headers (remove spaces, make lowercase)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Flexible Column Name Matching
    DATE_SYNONYMS = ['ds', 'date', 'day', 'timestamp']
    SALES_SYNONYMS = ['y', 'sales', 'quantity', 'qty', 'revenue', 'total', 'units_sold']
    ITEM_SYNONYMS = ['item', 'product', 'sku', 'name', 'item_name', 'product_name']

    ds_col = next((col for col in df.columns if col in DATE_SYNONYMS), None)
    y_col = next((col for col in df.columns if col in SALES_SYNONYMS), None)
    item_col = next((col for col in df.columns if col in ITEM_SYNONYMS), None)

    if not ds_col:
        raise ValueError(f"CSV must contain a Date column. Expected one of: {DATE_SYNONYMS}")
    if not y_col:
        raise ValueError(f"CSV must contain a Sales column. Expected one of: {SALES_SYNONYMS}")
    if not item_col:
        raise ValueError(f"CSV must contain an Item column. Expected one of: {ITEM_SYNONYMS}")
    
    # --- Data Cleaning ---
    final_df = df[[ds_col, y_col, item_col]].copy()
    final_df.columns = ['ds', 'y', 'item'] # Standardize names
    
    # *** FINAL DATE PARSING FIX ***
    # We now tell pandas to try and guess the format (e.g., MM/DD/YYYY vs YYYY-MM-DD)
    try:
        # 'format="mixed"' is the most flexible way to handle US/EU dates
        final_df['ds'] = pd.to_datetime(final_df['ds'], format='mixed', errors='coerce')
    except TypeError:
        # Fallback for older pandas versions
        final_df['ds'] = pd.to_datetime(final_df['ds'], infer_datetime_format=True, errors='coerce')
    
    # Drop any rows where the date could not be understood
    final_df = final_df.dropna(subset=['ds'])

    # Clean and convert sales column (removes $ and ,)
    final_df['y'] = final_df['y'].astype(str).str.replace(r'[$,]', '', regex=True)
    final_df['y'] = pd.to_numeric(final_df['y'], errors='coerce') 

    # Handle Missing Values (replaces blanks with 0)
    final_df['y'] = final_df['y'].fillna(0) 
    
    if len(final_df) < 30:
        raise ValueError("At least 30 days of valid, parsable data are required.")

    print(f"Successfully validated and cleaned {len(final_df)} rows from user CSV.")
    
    unique_items = final_df['item'].unique()
    return unique_items

# --- 4. AI ANALYST FUNCTION (This is REAL) ---
def get_ai_recommendation(forecast_json, metrics_json, user_question=None):
    if not client:
        return "AI Analyst is offline. (API Key not configured on server)."

    try:
        # Format the simulated forecast data into a simple table for the LLM
        forecast_table = "| Item | Recommended Reorder Qty |\n|---|---|\n" + \
                         "\n".join([f"| {item['item_name']} | {item['reorder_qty']} units |" for item in forecast_json])
        
        # The System Prompt sets the AI's persona
        system_prompt = (
            "You are an expert Inventory Analyst advising a small business owner. Be concise, clear, and actionable. "
            "Answer based ONLY on the provided forecast table and metrics. Do not use complex jargon. "
            "Your responses should be formatted in HTML (use <br> for new lines, <strong> for bold)."
        )
        
        # Build the message history
        messages = [{ "role": "system", "content": system_prompt }]
        
        # Add the context (the forecast data)
        messages.append({
            "role": "user", 
            "content": f"""
            Here is my 7-day reorder plan:
            My Forecast Accuracy is: {metrics_json['ACCURACY']}%
            
            Reorder Plan:
            {forecast_table}
            """
        })

        # Add the user's specific question
        if user_question:
            messages.append({"role": "user", "content": user_question})
        else:
            # This is the initial prompt for the main insight
            messages.append({"role": "user", "content": "Analyze this reorder plan. Provide a one-paragraph summary of the key actions for the week."})

        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        ai_text = response.choices[0].message.content
        return ai_text 

    except Exception as e:
        print(f"--- LLM Error: {e} ---")
        return f"<strong>AI Analyst error:</strong><br>Could not contact the AI service. Please check the server logs. {str(e)}"


# --- 5. THE API ENDPOINT (This is the "Actor") ---
@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    """
    Receives user CSV text, VALIDATES it, simulates a DYNAMIC forecast,
    calls the LLM for insights, and returns a complete package.
    """
    try:
        # 1. Get and Validate User Data (This part is REAL)
        user_csv_text = request.data.decode('utf-8')
        unique_item_names = parse_user_csv(user_csv_text) 
        
        # 2. Simulate Model Training (This is the "acting" for the demo)
        print("--- Simulating complex item-level Prophet training... ---")
        time.sleep(5) # Wait 5 seconds to feel realistic
        print("--- Simulation complete. ---")

        # 3. Create the DYNAMIC SIMULATED result
        simulated_forecast_data = []
        for item in unique_item_names:
            simulated_forecast_data.append({
                "item_name": item,
                "reorder_qty": random.randint(30, 150) # Generate a random order qty
            })
        
        # Limit to top 10 items for a clean chart
        if len(simulated_forecast_data) > 10:
             simulated_forecast_data = simulated_forecast_data[:10]

        
        # *** BUG FIX HERE ***
        # Find the top reorder quantity from the simulated data
        top_reorder_qty = 0
        if simulated_forecast_data:
            # This finds the maximum 'reorder_qty' from the list
            top_reorder_qty = max(item['reorder_qty'] for item in simulated_forecast_data)

        simulated_metrics = { 
            "MAE": 12.5, 
            "ACCURACY": 92.3, 
            "DEMAND": top_reorder_qty # <-- This is now the CORRECT maximum value
        }

        # 4. Get LIVE AI insight based on the DYNAMIC SIMULATED data
        ai_insight = get_ai_recommendation(simulated_forecast_data, simulated_metrics, None)

        # 5. Send all data back to the frontend
        return jsonify({
            'forecast': simulated_forecast_data,
            'metrics': simulated_metrics,
            'ai_insight': ai_insight
        })

    except Exception as e:
        print(f"--- Prediction Error: {e} ---")
        # Send a user-friendly error back to the browser
        return jsonify({'error': f'Data Processing Error: {str(e)}'}), 400

# API endpoint for the LIVE AI CHAT
@app.route('/api/chat', methods=['POST'])
def post_chat():
    try:
        data = request.json
        user_question = data['question']
        # Get the context data (the forecast) from the frontend
        forecast_context = data['forecast'] 
        metrics_context = data['metrics']
        
        # Get a LIVE AI response
        ai_response = get_ai_recommendation(forecast_context, metrics_context, user_question)
        
        return jsonify({'response': ai_response})

    except Exception as e:
        print(f"--- Chat Error: {e} ---")
        return jsonify({'error': f'Chat Error: {str(e)}'}), 400

# --- 6. SERVER START ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

