"""Flask application factory."""

from pathlib import Path

from flask import Flask

from .admin import init_admin
from .models import db
from .routes import bp as main_bp


def create_app():
    app = Flask(__name__)

    base_dir = Path(__file__).resolve().parents[2]
    db_path = base_dir / "data" / "processed" / "mpg.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    app.config.update(
        SECRET_KEY="change-me",
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path.as_posix()}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    db.init_app(app)
    app.register_blueprint(main_bp)
    init_admin(app, db)

    with app.app_context():
        db.create_all()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
