# ❤️ HeartBot – Heart Disease Prediction Web App  

HeartBot is a **form-based machine learning web application** that predicts the **risk of heart disease** based on user health information.  
It’s built with **Flask (Python backend), HTML, CSS, JavaScript, and Chart.js** to provide an easy-to-use, interactive experience.  

---

## 🚀 Features  
- **Splash Screen (splash.html)** → Welcomes the user with a starting interface.  
- **Form Page (index.html)** → Collects health-related details such as:  
  - Age  
  - Cholesterol level  
  - Blood pressure  
  - Chest pain type, etc.  
- **Results Page (results.html)** → Displays:  
  - Risk percentage in a **donut chart (Chart.js)**  
  - **Tailored health advice** based on results  
  - Button to **make another prediction** or return home  

---

## 🛠️ Tech Stack  

**Frontend:**  
- HTML  
- CSS  
- JavaScript  
- Chart.js (for donut chart visualization)  

**Backend:**  
- Python  
- Flask (for routing)  

**Machine Learning:**  
- Pandas (data preprocessing)  
- NumPy (mathematical operations)  
- Scikit-learn (ML algorithms, training & testing)  
- Pickle (saving & loading trained model)  

---

## 📂 Project Structure  
HeartBot/
│── static/ #CSS file (main.css)
│── templates/ # HTML files (splash.html, index.html, results.html)
│──user_data/ #json files with test user data
│── app.py # Flask application with routes
│──heart.csv #dataset
│── models.pkl # Saved ML model (pickle file)
│── README.md # Documentation
│── requirements.txt # Dependencies
│── train_model.py #


---

## ⚙️ How It Works  
1. User lands on the **Splash Screen** and clicks **Predict**.  
2. Flask routes to the **Form Page**, where the user enters health information.  
3. On submission, Flask passes data into the **trained ML model**.  
4. The **Results Page** shows:  
   - A **donut chart (Chart.js)** with the predicted heart disease risk %  
   - **Custom advice** based on prediction outcome  
   - Options to **try again** or return to the home page.  

---

## 🔧 Installation  

```bash
# Clone repo
git clone https://github.com/YourUsername/HeartBot.git
cd HeartBot

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py


 