from django.urls import path

from . import views
from .api import api

urlpatterns = [path("", views.home, name="home"), path("api/", api.urls)]
