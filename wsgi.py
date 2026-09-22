"""WSGI entry point: ``gunicorn wsgi:app``."""

from cleartusk.web import create_app

app = create_app()
