from django.urls import include, path

from classifier.views import home

urlpatterns = [
    path("", home, name="home"),
    path("api/", include("classifier.urls")),
]
