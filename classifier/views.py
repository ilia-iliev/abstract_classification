from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from classifier.forms import ClassificationForm
from classifier.serializers import ClassificationRequestSerializer
from classifier.services import ModelUnavailableError, classifier


def home(request):
    form = ClassificationForm(request.POST or None)
    context = {"form": form}
    response_status = status.HTTP_200_OK
    if request.method == "POST" and form.is_valid():
        try:
            result = classifier.predict([form.cleaned_data["abstract"]])[0]
        except ModelUnavailableError as error:
            context["service_error"] = str(error)
            response_status = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            context["prediction"] = {
                "categories": [label.replace("_", " ").title() for label in result["categories"]],
                "scores": [
                    {
                        "label": label.replace("_", " ").title(),
                        "percent": round(score * 100, 1),
                    }
                    for label, score in result["scores"].items()
                ],
            }
    return render(request, "classifier/home.html", context, status=response_status)


class ClassificationView(APIView):
    def post(self, request):
        serializer = ClassificationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        is_batch = "abstracts" in data
        abstracts = data["abstracts"] if is_batch else [data["abstract"]]
        try:
            predictions = classifier.predict(abstracts)
        except ModelUnavailableError as error:
            return Response(
                {"detail": str(error)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"predictions": predictions} if is_batch else predictions[0], status=status.HTTP_200_OK)


class HealthView(APIView):
    def get(self, request):
        backend = classifier.backend
        if backend is None:
            return Response(
                {"status": "unavailable", "classifier_backend": None, "detail": classifier.load_error},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"status": "ok", "classifier_backend": backend})
