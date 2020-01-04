from api.models import CoqaStory,QA
from rest_framework import viewsets, permissions

from rest_framework import status
from rest_framework.response import Response
from django.http import JsonResponse
from .serializers import StorySerializer,QASerializer
from .tf_api import askCoqaAI
from django.core import serializers
# Story Viewset

class StoryViewSet(viewsets.ModelViewSet):
    permission_classes = [
        permissions.AllowAny,
    ]
    queryset = CoqaStory.objects.all()

    serializer_class = StorySerializer


class QAViewSet(viewsets.ModelViewSet):
    permission_classes = [
        permissions.AllowAny,
    ]

    queryset = QA.objects.all()

    serializer_class = QASerializer

    def create(self, request, *args, **kwargs):

        data =request.data
        story_query = CoqaStory.objects.all()
        story = story_query.filter(id=data['story'])
        story_text=list(story)[0].story
        story_context = {"id":data["story"],"text":story_text}
        qa_query =  QA.objects.all()
        qas = list(qa_query.filter(story=data['story']))
        qas_set=[]
        for i in range(len(qas)):
            qa=qas[i]
            qas_set.append({"turn":qa.turn,"question":qa.question,"answer":qa.answer})

        data['answer'] = askCoqaAI(question=data['question'],turn=data['turn'],story=story_context,history=qas_set)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    # def update(self, request, *args, **kwargs):
    #     instance = self.get_object()
    #     instance.name = request.data.get("name")
    #     instance.save()
    #
    #     serializer = self.get_serializer(instance)
    #     serializer.is_valid(raise_exception=True)
    #     self.perform_update(serializer)
    #
    #     return JsonResponse(serializer.data)