from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase


class ClassificationApiTests(APITestCase):
    def test_classifies_one_abstract(self):
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

    def test_accepts_batch(self):
        response = self.client.post(
            reverse("classify"),
            {"abstracts": ["Quantum fields and particles.", "Protein and gene expression in cells."]},
            format="json",
        )
        assert response.status_code == status.HTTP_200_OK
        assert len(response.data["predictions"]) == 2

    def test_rejects_empty_batch(self):
        response = self.client.post(reverse("classify"), {"abstracts": []}, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_rejects_ambiguous_payload(self):
        response = self.client.post(
            reverse("classify"), {"abstract": "text", "abstracts": ["text"]}, format="json"
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
