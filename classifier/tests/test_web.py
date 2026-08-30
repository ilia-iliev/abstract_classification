from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from classifier.services import ModelUnavailableError


PREDICTION = {
    "categories": ["computer_science"],
    "scores": {
        "biology": 0.01,
        "chemistry": 0.02,
        "computer_science": 0.954,
        "physics": 0.01,
        "social_sciences": 0.006,
    },
    "backend": "pytorch",
}


class ClassifierWebTests(APITestCase):
    def test_home_shows_classification_form(self):
        response = self.client.get(reverse("home"))

        assert response.status_code == status.HTTP_200_OK
        self.assertContains(response, "Paste an arXiv abstract")
        self.assertContains(response, "<textarea", html=False)

    @patch("classifier.views.classifier.predict", return_value=[PREDICTION])
    def test_home_displays_human_readable_prediction(self, predict):
        response = self.client.post(reverse("home"), {"abstract": "A neural network study."})

        assert response.status_code == status.HTTP_200_OK
        self.assertContains(response, "Computer Science")
        self.assertContains(response, "95.4%")
        predict.assert_called_once_with(["A neural network study."])

    @patch(
        "classifier.views.classifier.predict",
        side_effect=ModelUnavailableError("Model artifact is missing"),
    )
    def test_home_explains_when_model_is_unavailable(self, _predict):
        response = self.client.post(reverse("home"), {"abstract": "A valid abstract."})

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        self.assertContains(response, "Classifier unavailable", status_code=503)
        self.assertContains(response, "Model artifact is missing", status_code=503)
