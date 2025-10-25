InventoryAI

An AI-Powered Inventory Forecasting and Optimization Platform (MVP)

Description

InventoryAI is a cloud-native web application that empowers small businesses to make smarter, data-driven inventory decisions.

This MVP prototype demonstrates a complete, end-to-end architecture:

A modern, responsive HTML/JavaScript frontend (deployed on Netlify).

A robust Python backend API (deployed on Render).

A live, secure integration with GPT-4o Mini for real-time AI analysis and chat.

Users can upload their daily sales CSV, and the application will validate the data, generate a (simulated) item-level forecast, and provide live, actionable insights from an AI analyst.

Key Features (MVP)

Modern & Responsive UI: A clean, clutter-free dashboard built with Tailwind CSS.

Robust CSV Validation: Python backend validates any user-uploaded CSV for required columns (date, sales, item) and data length (>30 days).

Item-Level Forecast Simulation: Simulates a complex, item-level Prophet forecast to demonstrate the final product's UX without the 20-minute live training delay.

Dynamic Bar Chart: A user-friendly, interactive Chart.js bar chart showing the recommended reorder quantity for each actual item from the user's file.

Live AI-Powered Insights: Securely calls the OpenAI (GPT-4o Mini) API from the backend to provide a real, live analysis of the forecast data.

Live AI Chat Assistant: A fully functional, ChatGPT-style chat window that allows users to ask follow-up questions about their reorder plan.

Tech Stack (MVP Architecture)

This project is built using a modern, secure, split-deployment (Jamstack) architecture.

1. Frontend (The "Face")

Technology: HTML, Tailwind CSS, JavaScript (ES6+)

Libraries: Chart.js (for visualization)

Deployment: Netlify (for continuous deployment from Git)

2. Backend (The "Brain")

Technology: Python, Flask (as a lightweight API server)

Libraries: Pandas (for data validation), OpenAI (for AI chat)

Deployment: Render (as a production-grade web service)

Project Progress and Status Tracking 📊

For the most current status, task assignments, and detailed progress on the InventoryAI project, please refer to our dedicated, live project board:

➡️ InventoryAI Live Project Tracker on Notion ⬅️

How to Run the Project (Midterm Delivery)

1. Live Deployed MVP (Recommended)

The full, interactive MVP is deployed to the cloud. The frontend (Netlify) is connected to the live backend API (Render).

Simply click this link to run the application:
https://inventoryaipro.netlify.app (Or your specific Netlify URL)

2. How to Run Locally

To run the app on your local machine, you must run both servers in two separate terminals:

Terminal 1 (Run the "Brain"):

### Set your secret key
export OPENAI_API_KEY='sk-YOUR_KEY_HERE'
### Run the Python server
python3 src/app_api.py


Terminal 2 (Run the "Face"):

### Run the simple web server
python3 -m http.server 8000
### Open http://localhost:8000/index.html in your browser
