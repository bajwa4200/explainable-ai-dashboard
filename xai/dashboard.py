"""Flask dashboard for model metrics and feature importance."""

from __future__ import annotations

from flask import Flask, jsonify, render_template_string

from xai.model import build_report, train_classifier

_TEMPLATE = """
<!doctype html>
<html>
<head><title>XAI Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; }
  table { border-collapse: collapse; }
  td, th { border: 1px solid #ccc; padding: 0.5rem 1rem; }
</style>
</head>
<body>
  <h1>Iris classifier explainability</h1>
  <p>Hold-out accuracy: <strong>{{ accuracy | round(3) }}</strong></p>
  <h2>Permutation importance</h2>
  <table>
    <tr><th>Feature</th><th>Mean importance</th></tr>
    {% for name, score in importances %}
    <tr><td>{{ name }}</td><td>{{ '%.4f' | format(score) }}</td></tr>
    {% endfor %}
  </table>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__)
    model, X_test, y_test, names = train_classifier()
    report = build_report(model, X_test, y_test, names)
    app.config["report"] = report

    @app.route("/")
    def index():
        r = app.config["report"]
        return render_template_string(
            _TEMPLATE,
            accuracy=r.accuracy,
            importances=r.importances,
        )

    @app.route("/api/metrics")
    def metrics():
        r = app.config["report"]
        return jsonify(
            {
                "accuracy": r.accuracy,
                "importances": [{"feature": n, "score": s} for n, s in r.importances],
            }
        )

    return app
