from __future__ import annotations

from django.db import models


class Question(models.Model):
    id = models.BigAutoField(primary_key=True)
    question_text = models.TextField()
    part_number = models.IntegerField()
    question_number = models.IntegerField()
    article_number = models.IntegerField()
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    class Meta:
        managed = False  # managed by ingestion script
        db_table = "Question"
        unique_together = ("part_number", "question_number", "article_number")

    def __str__(self) -> str:  # pragma: no cover
        return f"Q{self.id}: P{self.part_number} Q{self.question_number} A{self.article_number}"


class CorrectResponse(models.Model):
    id = models.BigAutoField(primary_key=True)
    question = models.ForeignKey(
        Question, on_delete=models.CASCADE, db_column="question_id", related_name="correct_responses"
    )
    response_text = models.TextField()

    class Meta:
        managed = False
        db_table = "CorrectResponse"

    def __str__(self) -> str:  # pragma: no cover
        return f"Correct for Q{self.question_id}"


class IncorrectResponse(models.Model):
    id = models.BigAutoField(primary_key=True)
    question = models.ForeignKey(
        Question, on_delete=models.CASCADE, db_column="question_id", related_name="incorrect_responses"
    )
    response_text = models.TextField()
    refutation_text = models.TextField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "IncorrectResponse"

    def __str__(self) -> str:  # pragma: no cover
        return f"Incorrect for Q{self.question_id}"

