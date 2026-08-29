from django.urls import path

from classifier.views import ClassificationView, HealthView

urlpatterns = [
    path("classify/", ClassificationView.as_view(), name="classify"),
    path("health/", HealthView.as_view(), name="health"),
]
