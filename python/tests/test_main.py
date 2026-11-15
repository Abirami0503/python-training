from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_employees():
    response = client.get("/employees")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_employee_success():
    response = client.get("/employees/Arun")
    assert response.status_code == 200
    assert response.json()["name"] == "Arun"


def test_employee_not_found():
    response = client.get("/employees/Ravi")
    assert response.status_code == 404


def test_invalid_salary():
    response = client.post("/employees", json={"name": "Ravi", "salary": -5})
    assert response.status_code == 422


def test_empty_name():
    response = client.post("/employees", json={"name": "", "salary": 30000})
    assert response.status_code == 422
