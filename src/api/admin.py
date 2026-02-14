"""Admin panel setup."""

from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

from .models import MPGRecord


def init_admin(app, db):
    admin = Admin(app, name="MPG Admin")
    admin.add_view(ModelView(MPGRecord, db.session, category="Data"))
    return admin
