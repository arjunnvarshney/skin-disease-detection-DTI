 🧠 SkinAI — Intelligent Skin Disease Detection and Assistant

**SkinAI** is a lightweight, intelligent web application that assists users in the early identification of skin-related conditions using AI. It combines image-based classification with a natural language Q&A assistant to improve awareness and encourage timely consultation with medical professionals.

⚠️ **Disclaimer**: This application is **not a diagnostic tool**. It is intended to provide educational insight and encourage professional medical advice.


## 🚀 Features

- *📷 Image-Based Diagnosis**: Upload a photo of a skin lesion and get an AI-generated prediction of possible skin conditions.
- 📑 Condition Information**: Get explanations for predicted conditions and suggested next steps.
- 🤖 AI-Powered Q&A Assistant**: Ask dermatology-related questions and get informative responses using a language model.
- 📊 Top-5 Predictions**: View confidence scores and alternative diagnoses for a clearer picture.
- 🔐 Google Sign-In**: Secure login using Google authentication.
- 📄 Downloadable Reports**: Export results as PDF for record-keeping or sharing with doctors.


 💡 Motivation

Skin conditions are widespread and can go undiagnosed for too long, especially in remote or underserved areas. **SkinAI bridges the gap** by providing instant, AI-powered insights to guide users toward timely medical care.



### ⚙️ **Tech Stack**

| Component          | Technology                                                          |
|--------------------|---------------------------------------------------------------------|
| **Backend**         | Python (Flask), OpenCV, PyTorch                                    |
| **Frontend**        | HTML, CSS, JavaScript                                              |
| **ML Model**        | Vision Transformer (ViT) fine-tuned using DINOv2 on ISIC dataset   |
| **Authentication**  | Google OAuth 2.0 via `Flask-OAuthlib`                              |
| **Report Export**   | `ReportLab` (Python library for generating PDFs server-side)       |




