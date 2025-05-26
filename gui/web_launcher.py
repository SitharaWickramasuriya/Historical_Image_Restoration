from flask import Flask, render_template_string, redirect, url_for
import subprocess
import threading
import os

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Historical Image Restoration</title>
    <style>
        body {
            background-image: url('/static/background.jpg');  /* Make sure image exists in static/ */
            background-size: cover;
            background-position: center;
            font-family: Arial, sans-serif;
            color: white;
            text-align: center;
            padding: 50px;
        }
        .container {
            background: rgba(0, 0, 0, 0.4);
            padding: 30px;
            border-radius: 20px;
            display: inline-block;
            backdrop-filter: blur(5px);
        }
        h1 {
            font-size: 48px;
            margin-bottom: 30px;
            border: 3px solid white;
            padding: 20px;
            border-radius: 25px;
            display: inline-block;
        }
        .info-box {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 20px;
            margin-bottom: 30px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        .info-box h2 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .info-box p {
            font-size: 18px;
        }
        .restore-button {
            background-color: white;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            font-size: 20px;
            padding: 15px 30px;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .restore-button:hover {
            background-color: white;
            background: rgba(255, 255, 255, 0.6);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Historical Image Restoration</h1>
        <div class="info-box">
            <h2>Image Restoration</h2>
            <p>Image restoration refers to the process of digitally repairing and enhancing old, damaged, or degraded photographs, artworks, or documents to restore them to their original or improved visual quality.</p>
        </div>
        <form action="/start">
            <button class="restore-button" type="submit">Let’s Restore Image</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(TEMPLATE)

@app.route('/start')
def start_app():
    def run_app():
        script_path = os.path.abspath('main.py')
        subprocess.Popen(f'start "" "python" "{script_path}"', shell=True)

    threading.Thread(target=run_app).start()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
