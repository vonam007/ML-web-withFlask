from flask import Flask, Blueprint


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secretkey'
    website_bp = Blueprint('website', __name__, template_folder='templates', static_folder='static')
    
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    return app