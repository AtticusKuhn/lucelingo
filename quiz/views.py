from __future__ import annotations

import random
from typing import List, Dict

from django.http import HttpRequest, HttpResponse, Http404
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .models import Question, CorrectResponse, IncorrectResponse


def _get_random_question() -> Question | None:
    qs = Question.objects.all()
    count = qs.count()
    if count == 0:
        return None
    # Efficient random selection by offset
    idx = random.randint(0, count - 1)
    return qs[idx]


def home(request: HttpRequest) -> HttpResponse:
    q = _get_random_question()
    if q is None:
        return render(request, "quiz/empty.html")
    choices: List[Dict] = []
    correct = CorrectResponse.objects.filter(question=q).first()
    if correct:
        choices.append({
            "id": correct.id,
            "text": correct.response_text,
            "kind": "correct",
        })
    incorrect = list(IncorrectResponse.objects.filter(question=q))
    for ir in incorrect:
        choices.append({
            "id": ir.id,
            "text": ir.response_text,
            "kind": "incorrect",
        })
    random.shuffle(choices)
    return render(request, "quiz/home.html", {"question": q, "choices": choices})


@require_GET
def random_question_fragment(request: HttpRequest) -> HttpResponse:
    q = _get_random_question()
    if q is None:
        return render(request, "quiz/empty_fragment.html")
    choices: List[Dict] = []
    correct = CorrectResponse.objects.filter(question=q).first()
    if correct:
        choices.append({
            "id": correct.id,
            "text": correct.response_text,
            "kind": "correct",
        })
    incorrect = list(IncorrectResponse.objects.filter(question=q))
    for ir in incorrect:
        choices.append({
            "id": ir.id,
            "text": ir.response_text,
            "kind": "incorrect",
        })
    random.shuffle(choices)
    return render(
        request,
        "quiz/question_fragment.html",
        {"question": q, "choices": choices},
    )


@require_GET
def answer_view(request: HttpRequest) -> HttpResponse:
    qid = request.GET.get("question_id")
    resp_id = request.GET.get("response_id")
    kind = request.GET.get("kind")
    if not (qid and resp_id and kind):
        raise Http404("Missing parameters")
    try:
        q = Question.objects.get(id=int(qid))
    except Question.DoesNotExist:
        raise Http404("Question not found")

    context: Dict = {"question": q}

    if kind == "correct":
        try:
            cr = CorrectResponse.objects.get(id=int(resp_id), question=q)
        except CorrectResponse.DoesNotExist:
            raise Http404("Response not found")
        context.update(
            {
                "is_correct": True,
                "feedback_title": "Correct!",
                "feedback_text": cr.response_text,
            }
        )
    elif kind == "incorrect":
        try:
            ir = IncorrectResponse.objects.get(id=int(resp_id), question=q)
        except IncorrectResponse.DoesNotExist:
            raise Http404("Response not found")
        context.update(
            {
                "is_correct": False,
                "feedback_title": "Not quite.",
                "feedback_text": ir.refutation_text or "",
            }
        )
    else:
        raise Http404("Invalid kind")

    return render(request, "quiz/answer_fragment.html", context)

