import io

import pytest

from cleartusk.services.corpus import ingest_corpus
from cleartusk.services.pipeline import CleaningPipeline
from tests.conftest import EXPECTED_BACKEND


@pytest.mark.parametrize(
    "path",
    ["/", "/studio", "/runs", "/analytics", "/contact", "/healthz", "/api/v1/", "/api/v1/call-types"],
)
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200


def test_legacy_dashboard_redirects(client):
    response = client.get("/dashboard")
    assert response.status_code == 301
    assert response.headers["Location"].endswith("/studio")


def test_health_reports_backend(client):
    payload = client.get("/healthz").get_json()
    assert payload["status"] == "ok"
    assert payload["database"] == EXPECTED_BACKEND


def test_process_endpoint_cleans_an_upload(client, corpus, wav_upload):
    response = client.post(
        "/api/v1/process",
        data={"audio_file": (io.BytesIO(wav_upload.read_bytes()), "field.wav"), "call_type": "rumble"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["call_type"] == "rumble"
    assert payload["metrics"]["target_retention_pct"] > 0
    assert payload["assets"]["cleaned_audio"].startswith("/media/")
    assert payload["links"]["self"].startswith("/runs/")

    detail = client.get(f"/api/v1/runs/{payload['run_id']}")
    assert detail.status_code == 200
    assert detail.get_json()["preset"] == payload["preset"]

    page = client.get(payload["links"]["self"])
    assert page.status_code == 200
    assert b"Call retained" in page.data

    media = client.get(payload["assets"]["cleaned_audio"])
    assert media.status_code == 200


def test_process_requires_a_file(client):
    response = client.post("/api/v1/process", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_process_rejects_an_unknown_call_type(client, wav_upload):
    response = client.post(
        "/api/v1/process",
        data={"audio_file": (io.BytesIO(wav_upload.read_bytes()), "f.wav"), "call_type": "banana"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unknown call type" in response.get_json()["error"]


def test_process_rejects_an_unsupported_extension(client, wav_upload):
    response = client.post(
        "/api/v1/process",
        data={"audio_file": (io.BytesIO(b"not audio"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_runs_listing_paginates(client, corpus, session, wav_upload):
    for _ in range(2):
        CleaningPipeline(session).process_file(wav_upload, "rumble")
    session.commit()

    payload = client.get("/api/v1/runs?page_size=1").get_json()
    assert len(payload["runs"]) == 1
    assert payload["has_next"] is True

    filtered = client.get("/api/v1/runs?call_type=trumpet").get_json()
    assert filtered["runs"] == []


def test_unknown_run_returns_404(client):
    assert client.get("/api/v1/runs/deadbeef").status_code == 404
    assert client.get("/runs/deadbeef").status_code == 404


def test_media_route_blocks_traversal(client):
    assert client.get("/media/../../etc/passwd").status_code == 404


def test_contact_form_stores_a_message(client):
    from cleartusk.db.models import ContactMessage

    response = client.post(
        "/contact",
        data={"name": "Ada", "email": "ada@example.com", "subject": "Hi", "message": "Nice work"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Thanks" in response.data

    with client.application.app_context():
        from cleartusk.db import session_scope

        with session_scope() as session:
            assert session.query(ContactMessage).count() == 1


def test_contact_form_validates_input(client):
    response = client.post("/contact", data={"name": "", "email": "", "subject": "", "message": ""})
    assert b"complete every field" in response.data

    response = client.post("/contact", data={"name": "A", "email": "nope", "subject": "S", "message": "M"})
    assert b"does not look valid" in response.data


def test_analytics_api_matches_the_page(client, corpus, session):
    ingest_corpus(session, corpus)
    session.commit()
    payload = client.get("/api/v1/analytics").get_json()
    assert payload["overview"]["corpus_clips"] == 6
    assert client.get("/analytics").status_code == 200


def test_detector_endpoint_404s_without_a_model(client):
    assert client.get("/api/v1/detector").status_code == 404
