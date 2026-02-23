from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
import traceback
from authlib.integrations.flask_client import OAuth
from huggingface_hub import InferenceClient
from PIL import Image
import os
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from dotenv import load_dotenv


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(16))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts for 7 days

# =======================================
# 📊 Database Configuration
# =======================================
# Handle Render/Heroku DATABASE_URL which often uses postgres:// (unsupported by SQLAlchemy 1.4+)
db_url = os.environ.get("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url or "sqlite:///skin_disease.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# =======================================
# 📝 Database Models
# =======================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    prediction = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'date': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# =======================================
# 🔐 Hugging Face LLM Token + InferenceClient
# =======================================
load_dotenv()
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

# Client for LLM inference
client = InferenceClient(
    provider="hf-inference",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=HUGGINGFACE_TOKEN
)

# Client for image classification - using provider for API access
skin_client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACE_TOKEN
)

# =======================================
# 🔍 Google OAuth Configuration
# =======================================
#app.config['SERVER_NAME'] = 'localhost:5000' for local use
app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get('GOOGLE_CLIENT_SECRET')

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={"scope": "openid email profile"},
)

# =======================================
# 📚 Skin Disease Class Labels
# =======================================
class_names = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa',
    'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans',
    'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid', 'Lichen Planus',
    'Lupus Erythematosus Chronicus Discoides', 'Melanoma', 'Molluscum Contagiosum',
    'Mycosis Fungoides', 'Neurofibromatosis', 'Papilomatosis Confluentes And Reticulate',
    'Pediculosis Capitis', 'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis',
    'Tinea Corporis', 'Tinea Nigra', 'Tungiasis', 'actinic keratosis', 'dermatofibroma',
    'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma',
    'vascular lesion'
]

# =======================================
# 📁 Make Upload Directory
# =======================================
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =======================================
# 🌐 Frontend Routes
# =======================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload")
def upload():
   if not session.get('logged_in'):
        flash('Please log in to access this page', 'warning')
        return redirect(url_for('login'))
   return render_template("upload.html")

@app.route("/result")
def result():
    if not session.get('logged_in'):
        flash('Please log in to access this page', 'warning')
        return redirect(url_for('login'))
    return render_template("result.html")
@app.route("/terms")
def terms():
    return render_template("terms.html")
@app.route("/privacy")
def privacy():
    return render_template("privacy.html") 

@app.route("/dashboard")
def dashboard():
    if not session.get('logged_in'):
        flash('Please log in to access this page', 'warning')
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    # Get all user analyses
    analyses = Analysis.query.filter_by(user_id=user_id).order_by(Analysis.created_at.desc()).all()
    
    # Calculate statistics
    total_analyses = len(analyses)
    
    # Format the last analysis date if analyses exist
    last_analysis_date = "None yet"
    if analyses and total_analyses > 0:
        last_analysis_date = analyses[0].created_at.strftime('%b %d')
    
    return render_template(
        "dashboard.html", 
        analyses=[analysis.to_dict() for analysis in analyses],
        total_analyses=total_analyses,
        last_analysis_date=last_analysis_date
    )

# analysis result
@app.route("/view-result/<int:analysis_id>")
def view_result(analysis_id):
    if not session.get('logged_in'):
        flash('Please log in to access this page', 'warning')
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')
    
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user_id).first_or_404()
    
    # Convert to dict for template
    analysis_data = analysis.to_dict()
    analysis_data['image_path'] = analysis.image_path  # Add image path which might not be in to_dict()
    
    return render_template("result.html", analysis=analysis_data)

@app.route("/download-report/<int:analysis_id>")
def download_report(analysis_id):
    if not session.get('logged_in'):
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session.get('user_id')
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user_id).first_or_404()

    try:
        from flask import make_response
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from io import BytesIO
        import datetime
        import os
        import textwrap

        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y_position = height - 50  # start from top

        # Title
        p.setFont("Helvetica-Bold", 24)
        p.drawString(50, y_position, "SkinAI Analysis Report")
        y_position -= 30

        # Date
        p.setFont("Helvetica", 12)
        p.drawString(50, y_position, f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}")
        y_position -= 40

        # Patient Info
        user = User.query.get(user_id)
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_position, "Patient Information:")
        y_position -= 20
        p.setFont("Helvetica", 12)
        p.drawString(70, y_position, f"Name: {user.fullname}")
        y_position -= 20
        p.drawString(70, y_position, f"ID: {user_id}")
        y_position -= 30

        # Analysis Details
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_position, "Analysis Details:")
        y_position -= 20
        p.setFont("Helvetica", 12)
        p.drawString(70, y_position, f"Date: {analysis.created_at.strftime('%B %d, %Y at %H:%M')}")
        y_position -= 20
        p.drawString(70, y_position, f"Diagnosis: {analysis.prediction}")
        y_position -= 20
        p.drawString(70, y_position, f"Confidence: {round(analysis.confidence * 100, 1)}%")
        y_position -= 30

        # Condition Info
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_position, "Condition Information:")
        y_position -= 20
        p.setFont("Helvetica", 12)

        condition_info = "Condition description not available."
        try:
            condition_prompt = f"Provide a concise (3-4 sentences) medical description of {analysis.prediction} as a skin condition. Include key symptoms, causes, and when to seek medical attention."
            messages = [{"role": "user", "content": condition_prompt}]
            response = client.chat_completion(messages=messages, max_tokens=150)
            condition_info = response.choices[0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error getting condition info from model: {str(e)}")

        # Wrap condition info
        wrapped_text = textwrap.wrap(condition_info, width=90)
        text_obj = p.beginText(70, y_position)
        text_obj.setFont("Helvetica", 12)
        for line in wrapped_text:
            text_obj.textLine(line)
        p.drawText(text_obj)
        y_position -= (len(wrapped_text) * 14 + 30)

        # Add Image if available
        try:
            image_path = os.path.join(app.root_path, 'static', analysis.image_path)
            if os.path.exists(image_path):
                img = ImageReader(image_path)
                img_width, img_height = img.getSize()
                max_width = 300
                max_height = 300
                scale_ratio = min(max_width / img_width, max_height / img_height)
                new_width = img_width * scale_ratio
                new_height = img_height * scale_ratio

                p.setFont("Helvetica-Bold", 14)
                p.drawString(50, y_position, "Analysis Image:")
                y_position -= (new_height + 30)

                p.rect(50, y_position - 10, new_width + 10, new_height + 10)
                p.drawImage(image_path, 55, y_position - 5, width=new_width, height=new_height)
                y_position -= 20
            else:
                raise FileNotFoundError("Image not found.")
        except Exception as e:
            print(f"Error adding image to PDF: {str(e)}")
            p.drawString(70, y_position, "Image could not be included in this report.")
            y_position -= 20

        # Disclaimer
        y_position = max(100, y_position - 40)
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y_position, "Disclaimer:")
        y_position -= 20
        p.setFont("Helvetica", 10)
        disclaimer_lines = [
            "This analysis is provided for informational purposes only and should not",
            "be considered a medical diagnosis. Please consult with a healthcare",
            "professional for proper medical advice and treatment."
        ]
        for line in disclaimer_lines:
            p.drawString(70, y_position, line)
            y_position -= 15

        # Footer
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(width / 2 - 100, 30, "© SkinAI - Advanced Skin Analysis System")

        # Finalize PDF
        p.showPage()
        p.save()
        pdf_data = buffer.getvalue()
        buffer.close()

        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=skin-analysis-report-{analysis_id}.pdf'
        return response

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return jsonify({"error": "Failed to generate report"}), 500


# =======================================
# 🔐 Authentication Routes
# =======================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['logged_in'] = True
            session['user_id'] = user.id
            session['email'] = user.email
            session['name'] = user.fullname
            session.permanent = remember

            print("🎉 Login success for:", email)

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True}), 200

            return redirect(url_for('index'))

        else:
            print("❌ Login failed for:", email)
            msg = 'Invalid email or password'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': msg}), 200
            flash(msg, 'danger')
            return render_template("login.html")

    return render_template("login.html")

# =======================================
#login_google
#========================================
@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    print(f"Redirecting to: {redirect_uri}")  # Debug output
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/authorized')  # This must match exactly
def google_callback():
    try:
        token = google.authorize_access_token()
        user_info = google.parse_id_token(token)
        
        # Extract user info
        email = user_info['email']
        name = user_info.get('name', 'Google User')
        
        print(f"Successfully authenticated: {email}")  # Debug output

        # Check or create user
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(fullname=name, email=email, password=generate_password_hash(secrets.token_hex(8)))
            db.session.add(user)
            db.session.commit()

        session['logged_in'] = True
        session['user_id'] = user.id
        session['email'] = user.email
        session['name'] = user.fullname
        
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Google OAuth Error: {str(e)}")
        traceback.print_exc()  # Print full traceback
        flash("Login with Google failed. Please try again or use email login.", "danger")
        return redirect(url_for('login'))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    print("🔁 [SIGNUP] Request method:", request.method)
    
    if request.method == "POST":
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        password = request.form.get('password')

        print(f"📥 Received - Name: {fullname}, Email: {email}")

        try:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                msg = "Email already registered."
                print("⚠️", msg)
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'message': msg}), 200
                flash(msg, 'danger')
                return render_template("signup.html")

            hashed_password = generate_password_hash(password)
            new_user = User(fullname=fullname, email=email, password=hashed_password)

            db.session.add(new_user)
            db.session.commit()

            msg = "Account created successfully!"
            print("✅", msg)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True}), 200
            flash(msg, 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            print("❌ SIGNUP ERROR:", str(e))
            traceback.print_exc()
            msg = f"Error: {str(e)}"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': msg}), 200
            flash(msg, 'danger')
            return render_template("signup.html")

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# =======================================
# 📸 /analyze Route - MODIFIED TO USE INFERENCE API
# =======================================
@app.route('/analyze', methods=['POST'])
def analyze():
    if not session.get('logged_in'):
        return jsonify({"error": "Not authenticated"}), 401
        
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    user_id = session.get('user_id')
    image_file = request.files['image']
    
    # Save the image
    filename = f"user_{user_id}_{secrets.token_hex(8)}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)
    
    try:
        # Process with Hugging Face Inference API
        # The model is accessed via API instead of loading it locally
        result = skin_client.image_classification(
            filepath, 
            model="Jayanth2002/dinov2-base-finetuned-SkinDisease"
        )
        
        # Extract results
        # The API returns a list of label/score pairs
        scores = [(item['label'], item['score']) for item in result]
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top prediction
        top_prediction = scores[0][0]
        top_confidence = scores[0][1]
        
        # Get top 5 conditions
        top_conditions = scores[:5]
        
        # Get detailed condition description from Mistral model
        description = ""
        recommendations = []
        try:
            # Create prompts for description and recommendations
            description_prompt = f"Describe {top_prediction} in 2-3 simple sentences as if explaining to a patient. Include basic symptoms and general information."
            
            recommendations_prompt = f"Provide 3-4 specific recommendations for someone who might have {top_prediction}. Include advice about when to see a doctor and what self-care might be appropriate. Format as a simple bullet list."
            
            # Get description
            messages_description = [{"role": "user", "content": description_prompt}]
            response_description = client.chat_completion(
                messages=messages_description,
                max_tokens=100
            )
            description = response_description.choices[0]["message"]["content"].strip()
            
            # Get recommendations
            messages_recommendations = [{"role": "user", "content": recommendations_prompt}]
            response_recommendations = client.chat_completion(
                messages=messages_recommendations,
                max_tokens=150
            )
            rec_text = response_recommendations.choices[0]["message"]["content"].strip()
            
            recommendations = [line.strip().replace('- ', '') for line in rec_text.split('\n') if line.strip()]
            if not recommendations:
                recommendations = [
                    "Take a clearer image if unsure.",
                    "Consider visiting a dermatologist.",
                    "Avoid self-diagnosis or self-treatment."
                ]
            
        except Exception as e:
            print(f"Error getting information from model: {str(e)}")
            description = f"{top_prediction} is a skin condition. Please consult a medical professional."
            recommendations = [
                "Take a clearer image if unsure.",
                "Consider visiting a dermatologist.",
                "Avoid self-diagnosis or self-treatment."
            ]

        # Save analysis to database
        new_analysis = Analysis(
            user_id=user_id,
            image_path=os.path.join('uploads', filename),
            prediction=top_prediction,
            confidence=top_confidence
        )
        
        try:
            db.session.add(new_analysis)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Database error: {str(e)}")

        return jsonify({
            "analysis_id": new_analysis.id,
            "prediction": top_prediction,
            "confidence": top_confidence,
            "topConditions": [(name, score) for name, score in top_conditions],
            "description": description,
            "recommendations": recommendations,
            "image_path": os.path.join('static', 'uploads', filename)
        })
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# =======================================
# 💬 /ask Route
# =======================================
@app.route('/ask', methods=['POST'])
def ask():
    if not session.get('logged_in'):
        return jsonify({"error": "Not authenticated"}), 401
        
    data = request.json
    question = data.get("question", "")
    condition = data.get("condition", "")
    language = data.get("language", "en")  # Get the language from the request

    if not question:
        return jsonify({"answer": "Please ask a valid question."}), 400

    # Prompt for multilingual response
    messages = [
        {
            "role": "user",
            "content": f"""A user may have {condition}. They asked in {language}: '{question}'. 
            Keep it professional don't use causal tone.
            Respond like a helpful AI medical assistant. Keep your response focused on the question.Keep
            it 3-4 sentences long.
            Provide a concise answer, avoiding unnecessary details.
            If the question is not clear, ask for clarification.No need to tell the language of the question.
            """
        }
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=300
        )
        answer = response.choices[0]["message"]["content"]
        return jsonify({"answer": answer.strip()})
    except Exception as e:
        return jsonify({"answer": f"Error communicating with Hugging Face: {e}"}), 500

@app.route('/api/placeholder/<width>/<height>')  
def placeholder(width, height):
    # This is a simple implementation - you might want to generate an actual placeholder image
    # For now, we'll just serve a static placeholder
    return send_from_directory('static/images', 'placeholder.jpg')

# =======================================
# 📊 Initialize Database
# =======================================
# =======================================
# 📊 Initialize Database (One-time)
# =======================================
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables initialized!")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
