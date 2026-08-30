from types import SimpleNamespace
from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from classifier.services import ModelUnavailableError


PREDICTION = {
    "categories": ["computer_science"],
    "scores": {
        "biology": 0.01,
        "chemistry": 0.01,
        "computer_science": 0.95,
        "physics": 0.02,
        "social_sciences": 0.01,
    },
    "backend": "pytorch",
}


class ClassificationApiTests(APITestCase):
    @patch("classifier.views.classifier.predict", return_value=[PREDICTION])
    def test_classifies_one_abstract(self, _predict):
        response = self.client.post(
            reverse("classify"),
            {"abstract": "We train a neural network algorithm for image classification."},
            format="json",
        )
        assert response.status_code == status.HTTP_200_OK
        assert "computer_science" in response.data["categories"]
        assert set(response.data["scores"]) == {
            "biology", "chemistry", "computer_science", "physics", "social_sciences"
        }

    @patch("classifier.views.classifier.predict", return_value=[PREDICTION, PREDICTION])
    def test_accepts_batch(self, _predict):
        response = self.client.post(
            reverse("classify"),
            {"abstracts": ["Quantum fields and particles.", "Protein and gene expression in cells."]},
            format="json",
        )
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data["predictions"]) == 2

    @patch(
        "classifier.views.classifier.predict",
        side_effect=ModelUnavailableError("Model artifact is missing"),
    )
    def test_returns_503_when_model_is_unavailable(self, _predict):
        response = self.client.post(
            reverse("classify"), {"abstract": "A valid abstract."}, format="json"
        )
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.data == {"detail": "Model artifact is missing"}

    @patch(
        "classifier.views.classifier",
        new=SimpleNamespace(backend=None, load_error="Model artifact is missing"),
    )
    def test_health_returns_503_when_model_is_unavailable(self):
        response = self.client.get(reverse("health"))
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.data["status"] == "unavailable"
        assert response.data["classifier_backend"] is None

    def test_rejects_empty_batch(self):
        response = self.client.post(reverse("classify"), {"abstracts": []}, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_rejects_ambiguous_payload(self):
        response = self.client.post(
            reverse("classify"), {"abstract": "text", "abstracts": ["text"]}, format="json"
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
