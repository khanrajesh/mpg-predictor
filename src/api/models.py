"""Database models."""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class MPGRecord(db.Model):
    __tablename__ = "mpg_records"

    id = db.Column(db.Integer, primary_key=True)
    mpg = db.Column(db.Float, nullable=True)
    cylinders = db.Column(db.Integer, nullable=True)
    displacement = db.Column(db.Float, nullable=True)
    horsepower = db.Column(db.Float, nullable=True)
    weight = db.Column(db.Float, nullable=True)
    acceleration = db.Column(db.Float, nullable=True)
    model_year = db.Column(db.Integer, nullable=True)
    origin = db.Column(db.String(50), nullable=True)
    name = db.Column(db.String(120), nullable=True)

    def __repr__(self) -> str:
        return f"<MPGRecord id={self.id} name={self.name!r}>"
