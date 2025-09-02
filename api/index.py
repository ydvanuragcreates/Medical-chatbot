from flask import Flask

# The 'app' variable is what Vercel will look for.
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello from a Python backend on Vercel! 🐍'

@app.route('/time')
def current_time():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current server time is: {now}"

# IMPORTANT: Do NOT include the following line
# if __name__ == '__main__':
#     app.run(debug=True)