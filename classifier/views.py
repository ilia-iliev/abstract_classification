from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from classifier.serializers import ClassificationRequestSerializer
from classifier.services import classifier


class ClassificationView(APIView):
    def post(self, request):
        serializer = ClassificationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        is_batch = "abstracts" in data
        abstracts = data["abstracts"] if is_batch else [data["abstract"]]
        predictions = classifier.predict(abstracts)
        return Response({"predictions": predictions} if is_batch else predictions[0], status=status.HTTP_200_OK)


class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok", "classifier_backend": classifier.backend})
