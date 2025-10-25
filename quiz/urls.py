from django.urls import path

from . import views

app_name = "quiz"

urlpatterns = [
    path("", views.home, name="home"),
    path("question/", views.random_question_fragment, name="question_fragment"),
    path("answer/", views.answer_view, name="answer"),
]
