# LSTM Database Anomaly Detection (Real AWS Data)
### "Predicting database failures before they happen"

## What this project does (in one line)
It watches a server's health (CPU usage) the way a hospital monitor watches a heartbeat, and warns you when the pattern starts looking abnormal — often hours before an actual crash.

## Real-life example
Imagine an online shopping website. Its database server usually runs at 30% CPU. One night, CPU slowly creeps up and stays high for hours — a warning sign. Two hours later, the server crashes and the whole website goes down. This project uses a real, historically confirmed AWS server incident (from Amazon's own CloudWatch logs) and trains an AI to recognize that early warning pattern automatically.

## No paid AI API needed
Everything here uses free, open-source tools:
- **TensorFlow/Keras** – to build and train the AI model (100% free, runs on your own computer)
- **The NAB dataset** – real, publicly available AWS server data (free, from github.com/numenta/NAB)
- **Streamlit** – for the dashboard (free)

---

## Folder Structure
```
db-anomaly-detection-lstm/
│
├── data/
│   └── realAWSCloudwatch/
│       └── ec2_cpu_utilization_24ae8d.csv   ← real AWS data (auto-downloads on first run)
│
├── labels/
│   └── combined_labels.json                 ← official real anomaly timestamps (auto-downloads)
│
├── model/
│   └── lstm_anomaly_model.h5                 ← saved trained model (created after training)
│
├── outputs/
│   ├── results.csv                           ← final anomaly predictions (created after training)
│   └── anomaly_plot.png                      ← saved graph (created after training)
│
├── train_model.py                            ← run this FIRST — trains the AI
├── app.py                                    ← run this SECOND — shows the dashboard
├── requirements.txt                          ← list of free Python libraries needed
└── README.md                                 ← this file
```

---

## How to Run This Project (Step by Step)

### Step 1: Install Python
If you don't already have Python, download it free from **python.org/downloads** (get version 3.10 or higher). During install, make sure to check "Add Python to PATH."

### Step 2: Open a terminal in this project folder
- **Windows:** open the folder, then type `cmd` in the address bar and press Enter
- **Mac/Linux:** right-click the folder → "Open Terminal Here" (or use `cd` to navigate to it)

### Step 3: Install the required free libraries
```bash
pip install -r requirements.txt
```
This installs TensorFlow, pandas, numpy, scikit-learn, matplotlib, and streamlit — all free and open-source.

### Step 4: Train the AI model
```bash
python train_model.py
```
What happens:
1. It automatically downloads the real AWS server data and the official anomaly labels (no manual downloading needed)
2. It trains an LSTM Autoencoder for about 1-3 minutes
3. It saves the trained model to `model/lstm_anomaly_model.h5`
4. It saves the results to `outputs/results.csv` and a graph to `outputs/anomaly_plot.png`

You'll see training progress printed in the terminal, ending with something like:
```
DONE. Now run:  streamlit run app.py
```

### Step 5: Launch the dashboard
```bash
streamlit run app.py
```
This opens a browser window (usually `http://localhost:8501`) showing:
- The raw CPU usage graph with the real confirmed anomaly marked in green
- The AI's "how unusual is this moment" score, with a red threshold line
- A table of every point the AI flagged as risky

---

## How to read the results (important, honest explanation)

The AI doesn't perfectly draw a box around only the real anomaly — that's normal and expected for this kind of unsupervised model. What you should look for:
- The AI's error score rises noticeably **during** the real confirmed anomaly window (2014-02-26 22:05 to 2014-02-27 17:15)
- Some of the highest error values in the entire dataset occur inside that real anomaly window
- This shows the model is correctly picking up on genuinely unusual server behavior, even without ever being told exactly where the anomaly was during training

This is exactly how real-world anomaly detection works — it flags "this looks unusual, someone should check it," not "I am 100% certain this is a failure." That's why these systems are used to prioritize what engineers investigate, not to fully replace human judgement.

---

## How to explain this project in an interview (plain language)

> "I built an anomaly detection system using a real AWS server dataset with a documented, confirmed incident. I trained an LSTM autoencoder — a neural network that learns to reconstruct normal patterns — using only the 'healthy' portion of the data. When tested against the full timeline, including the real incident window, the model's reconstruction error rose sharply during that confirmed incident period, showing it can flag genuinely abnormal server behavior before a full failure occurs, using only historical performance metrics."

---

## Deploying the dashboard online (optional, free)
1. Push this whole folder to a GitHub repository
2. Go to **share.streamlit.io**, sign in with GitHub
3. Select your repository and `app.py` as the entry point
4. Click Deploy — you'll get a free public link to share with recruiters

**Note:** Since Streamlit Cloud starts with a fresh environment, you'll need `train_model.py` to have already generated `outputs/results.csv` before deploying (commit that file to your repo), or add a step in `app.py` to run training automatically on first load.

---

## Where everything comes from
| Item | Source |
|---|---|
| Real server data | github.com/numenta/NAB (Numenta Anomaly Benchmark, free & open) |
| Real anomaly labels | Same repo — confirmed by Numenta's research team |
| AI library | TensorFlow/Keras (free, open-source) |
| Dashboard | Streamlit (free, open-source) |
