# Create your views here.

from django.http import HttpResponse, HttpRequest
from django.template import loader

def home(request):
    template = loader.get_template("qme.html")
    return HttpResponse(template.render(None, request))