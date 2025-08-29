 HeartBot – Heart Disease Prediction Web App

HeartBot is a **machine learning–powered web application** that predicts the **risk of heart disease** based on user health information.
It’s built with **Flask (Python), HTML, CSS, JavaScript, and Chart.js** to deliver an interactive and user-friendly experience.
---

## Features

* **Splash Screen (splash.html)** → Animated welcome page
* **Form Page (index.html)** → Collects user health details:

  * Age, Cholesterol, Blood Pressure, Chest Pain Type, etc.
* **Results Page (results.html)** → Displays:

  * Risk percentage in a **donut chart (Chart.js)**
  * **Tailored health advice**
  * Buttons to **make another prediction** or return home

---

##  Tech Stack

**Frontend:**

* HTML, CSS, JavaScript
* Chart.js → interactive donut chart visualization

**Backend:**

* Python, Flask → web server & routing

**Machine Learning:**

* Pandas → data preprocessing
* NumPy → mathematical operations
* Scikit-learn → ML algorithms (training & testing)
* Pickle → saving & loading trained model

---

##  Project Structure

```
HeartBot/
│── static/           # CSS file (style.css)
│── templates/        # HTML files (splash.html, index.html, results.html)
│── user_data/        # Sample JSON test data
│── app.py            # Flask application with routes
│── heart.csv         # Dataset
│── models.pkl        # Trained ML model (pickle file)
│── train_model.py    # Script for training ML model
│── requirements.txt  # Dependencies
│── README.md         # Documentation
```

---

##  How It Works

1. User lands on the **Splash Screen** and clicks **Predict**.
2. Flask routes to the **Form Page**, where the user enters health data.
3. The inputs are processed and passed into the **trained ML model**.
4. The **Results Page** shows:
   * A **donut chart (Chart.js)** with prediction probability
   * **Personalized health advice**
   * Options to restart or go back home


* Improve model accuracy using larger datasets
* Add authentication & user history tracking
* Deploy on cloud (Heroku/AWS) for public access
* Enhance UI/UX with more interactive elements
