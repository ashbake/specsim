from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    sin_val = None
    cos_val = None
    
    if request.method == "POST":
        try:
            x = float(request.form.get("x_value"))
            sin_val = np.sin(x)
            cos_val = np.cos(x)
            
            plt.figure()
            X = np.linspace(-2*np.pi, 2*np.pi, 400)
            Y1 = np.sin(X)
            Y2 = np.cos(X)
            plt.plot(X, Y1, label="sin(x)")
            plt.plot(X, Y2, label="cos(x)")
            plt.axvline(x=x, color='r', linestyle='--', label=f'x={x}')
            plt.title("sin(x) and cos(x)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode()
            buf.close()
            
        except ValueError:
            return "Invalid Input. Please enter a number."
    
    return render_template("index.html", sin_val=sin_val, cos_val=cos_val, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
