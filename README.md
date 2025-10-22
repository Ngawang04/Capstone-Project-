# InventoryAI

**AI-Powered Inventory Forecasting and Optimization Platform**

---

## Description

InventoryAI is a streamlined, cloud-based web application that empowers small and medium businesses to make smarter inventory decisions. Users upload sales or inventory data and instantly receive advanced, AI-powered forecasts, optimal reorder recommendations, and simple plain-English explanations—all delivered through an intuitive and secure dashboard.

---

## Key Features

- **Data Upload:** Drag-and-drop CSV/Excel sales/inventory files with validation and instant preview.
- **Automated Forecasting:** Accurate, multi-product demand forecasts powered by Prophet and scikit-learn.
- **Smart Reorder Recommendations:** Clear reorder quantities and timing for every product.
- **Interactive Visualization:** Dynamic charts and tables for sales, forecasts, and inventory health.
- **AI Explanations & Q&A:** Business-friendly, GPT-4-powered explanations of forecasts and stock alerts.
- **Downloadable Reports:** Export recommendations and analytics as PDF or Excel files.
- **Secure Accounts:** AWS Cognito-based signup/login, with user-specific data isolation.
- **Persistent Cloud Storage:** AWS S3 file storage and PostgreSQL database for uploads, results, and user info.
- **Mobile-Friendly UI:** Responsive Streamlit dashboard for desktop, tablet, and mobile devices.

---

## Tech Stack

### Frontend (and Fullstack UI)
- **Python 3.12.6**
- **Streamlit** (UI components, routing, application logic)
- **Plotly/Streamlit Native Charts** (visualizations)
- **AWS Cognito** (authentication)

### Backend & Infrastructure
- **Prophet** (AI forecasting engine)
- **scikit-learn** (feature engineering, ML experiments, benchmarking)
- **OpenAI GPT-4 API** (AI explanations and chat)
- **PostgreSQL** (via AWS RDS, managed database service)
- **AWS S3** (secure file/storage for uploads and reports)
- **AWS EC2 / Elastic Beanstalk** (cloud app deployment)
- **APIs for holidays, events, and optionally weather** (data enrichment)

---

## Project Progress and Status Tracking 📊

For the most current status, task assignments, and detailed progress on the **InventoryAI** project, please refer to our dedicated, live project board. We are using a Kanban board to manage our workflow.

➡️ **[InventoryAI Live Project Tracker on Notion](https://www.notion.so/27d4d5b4186f8013aefbfa84767d86d4?v=27d4d5b4186f80e2ba4d000cc763ec25&source=copy_link)** ⬅️

---



---

## How to Run the Project (Midterm Delivery)



---

## Cost Breakdown(estimated)

- **OpenAI GPT-4 API:** ~$0.02–$0.04 per explanation/Q&A (~$5–10 for class demo)
- **AWS Hosting (EC2, RDS, S3):** $25–50/month (after free tier, for small-scale use)
- **Domain (if used):** ~$10–30/year
- **Open-source dependencies:** Free

---

## Business Use Case

InventoryAI enables SMBs to:
- Avoid stockouts and lost revenue by predicting future product demand
- Prevent overstocking and excess cash tied up in inventory
- Save time and reduce errors—transitioning from guesswork/spreadsheet management to data-driven, AI-supported recommendations
- Understand the “why” behind every reorder or forecast, thanks to AI-powered natural-language explanations

---

## Authors & Acknowledgements

**Authors**
-  Ngawang Choega
-  Dhruv Mane

**Acknowledgements**
- Prof. Darsh Joshi
- Thanks to Prophet, OpenAI, Streamlit, AWS, and the open-source/data science community

---

_For questions or issues, open a GitHub issue or reach out to a project author._
