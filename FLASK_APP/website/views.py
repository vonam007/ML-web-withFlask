from flask import Blueprint, render_template, request, flash, jsonify , url_for

views = Blueprint('views', __name__)

@views.route('/', methods=['GET'])
def home():
    return render_template("index.html")

