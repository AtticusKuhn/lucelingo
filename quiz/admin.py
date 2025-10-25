from django.contrib import admin

from .models import CorrectResponse, IncorrectResponse, Question


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("id", "part_number", "question_number", "article_number", "question_text")
    search_fields = ("question_text",)


@admin.register(CorrectResponse)
class CorrectResponseAdmin(admin.ModelAdmin):
    list_display = ("id", "question", "response_text")


@admin.register(IncorrectResponse)
class IncorrectResponseAdmin(admin.ModelAdmin):
    list_display = ("id", "question", "response_text")
