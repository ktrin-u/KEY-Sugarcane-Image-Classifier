# Create your views here.
from django.http import HttpResponse
from django.template import loader


def home(request) -> HttpResponse:
    template = loader.get_template("qme.html")
    return HttpResponse(template.render(None, request))
